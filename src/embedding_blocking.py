"""
OPTIONAL recall-augmentation stage, off by default (BlockingConfig.
use_embedding_blocking=False).

Why this exists: token/n-gram blocking (blocking.py) cannot find a
candidate pair where the two records are in DIFFERENT SCRIPTS — a
Devanagari name and its romanized counterpart share zero characters, so
no token or char-n-gram key will ever join them. EDA on the real data
found this affects roughly a fifth of Indian records in Source 2/3. A
multilingual sentence embedding model maps semantically-equivalent text
from different scripts/languages into nearby vectors, which is exactly
what's needed here — and unlike a hand-built transliteration table, it
generalizes to French too (also unseen in training).

Model recommendation: sentence-transformers/LaBSE
  - Apache-2.0 licensed, 470.9M parameters (comfortably under the 8B cap) —
    verified directly against the Hugging Face Hub API (not just the model
    card page) on 2026-09-25: license=apache-2.0, downloads=27.9M.
  - Its language tag list explicitly includes every script this dataset
    needs: hi (Devanagari/Hindi), kn (Kannada), ta (Tamil), te (Telugu),
    bn (Bengali), pa (Gurmukhi/Punjabi), plus fr (French) — and it's
    purpose-built for cross-lingual sentence similarity (that's the whole
    point of LaBSE), not a general-purpose model pressed into service.
  - Strong alternative, also verified apache-2.0: Qwen/Qwen3-Embedding-0.6B
    (595.8M params, newer, 89.3M downloads at last check) — worth an A/B
    if you have time. intfloat/multilingual-e5-base (278M, MIT) is a
    smaller/faster third option.
This is a *local* pretrained-model download (weights only, no API calls,
no lookup service), which is different in kind from the "external data
lookup" the brief prohibits (business-identity/geocoding/commercial-ER
APIs). If you want to be extra safe, keep a copy of the model card in your
submission zip to document the license at the version you used.

Before committing to a full run, sanity-check the cross-script hypothesis
cheaply: Hugging Face hosts a live demo Space (e.g. search
"qwen3-embedding" or "sentence-embeddings" under Spaces) where you can
paste a real romanized name and a real Devanagari/Kannada/Tamil name side
by side and see the cosine similarity before installing anything locally.

REQUIRES (not installed by default — heavier deps, and the model weights
~1.9GB need to download from huggingface.co on first run, so this needs
your own machine's internet access; it will NOT work inside a sandboxed
environment with no outbound access to huggingface.co):
    pip install sentence-transformers faiss-cpu torch

The FAISS indexing/query logic below (build_ann_index / query_ann) is
tested in tests/test_embedding_blocking.py using random vectors standing
in for real embeddings — that verifies the wiring (ID mapping, top-k,
cosine-via-inner-product) independent of the model itself, which requires
network access this environment doesn't have.
"""
import numpy as np
import pandas as pd


def build_ann_index(embeddings: np.ndarray, ids: list):
    """embeddings: (n, d) float32, ids: length-n list of entity_id strings
    aligned to embeddings' rows. Returns (index, id_array).
    Uses inner product on L2-normalized vectors == cosine similarity."""
    import faiss
    assert embeddings.shape[0] == len(ids)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, np.array(ids)


def query_ann(index, id_array: np.ndarray, query_embeddings: np.ndarray,
              query_ids: list, top_k: int = 20) -> pd.DataFrame:
    """Returns a candidate-pairs frame: entity_id, candidate_id,
    embedding_score (cosine similarity, higher = more similar)."""
    import faiss
    query_embeddings = query_embeddings.astype("float32").copy()
    faiss.normalize_L2(query_embeddings)
    k = min(top_k, index.ntotal)
    if k == 0:
        return pd.DataFrame(columns=["entity_id", "candidate_id", "embedding_score"])
    scores, idx = index.search(query_embeddings, k)
    rows = []
    for qi, qid in enumerate(query_ids):
        for rank in range(k):
            j = idx[qi, rank]
            if j < 0:
                continue
            rows.append((qid, id_array[j], float(scores[qi, rank])))
    return pd.DataFrame(rows, columns=["entity_id", "candidate_id", "embedding_score"])


def encode_records(df: pd.DataFrame, model, name_col="business_name",
                    addr_col="business_address", batch_size=256) -> np.ndarray:
    """Concatenate name + address (address adds disambiguating context,
    e.g. two 'Sharma General Store' in different cities) and encode.
    `model` is a loaded sentence_transformers.SentenceTransformer."""
    texts = (df[name_col].fillna("") + " " + df[addr_col].fillna("")).tolist()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True)


def merge_embedding_candidates(token_candidates: pd.DataFrame,
                                 embedding_candidates: pd.DataFrame) -> pd.DataFrame:
    """Union blocking.generate_candidates' output with this module's
    embedding-based candidates, so records that share NO token/n-gram
    (cross-script pairs) still enter the candidate pool, and every
    candidate carries an embedding_score feature (NaN where a pair was
    found by tokens/n-grams only and never appeared in the ANN top-k —
    LightGBM handles NaN natively via learned split defaults, so this is
    left as a genuine missing value rather than imputed to some
    arbitrary constant that would misrepresent 'no signal' as 'low
    similarity')."""
    import numpy as np
    if token_candidates.empty:
        merged = embedding_candidates.copy()
        merged["n_keys"] = 1
        merged["match_keys"] = "embedding"
        return merged
    if embedding_candidates.empty:
        merged = token_candidates.copy()
        merged["embedding_score"] = np.nan
        return merged

    merged = token_candidates.merge(
        embedding_candidates, on=["entity_id", "candidate_id"], how="outer")
    from_embedding_only = merged["match_keys"].isna()
    merged.loc[from_embedding_only, "match_keys"] = "embedding"
    merged.loc[from_embedding_only, "n_keys"] = 1
    had_both = ~from_embedding_only & merged["embedding_score"].notna()
    merged.loc[had_both, "match_keys"] = merged.loc[had_both, "match_keys"] + ",embedding"
    merged.loc[had_both, "n_keys"] = merged.loc[had_both, "n_keys"] + 1
    return merged


def generate_embedding_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                                    model_name: str = "sentence-transformers/LaBSE",
                                    top_k: int = 20, batch_size: int = 256) -> pd.DataFrame:
    """Full pipeline: load model -> encode both sides -> ANN search.
    Run this ONCE per country partition (like blocking.generate_candidates)
    to keep the index size manageable and avoid cross-country matches.
    The returned embedding_score is meant to be joined onto the token-based
    candidates as an EXTRA feature (see features.py) — do not replace
    token/n-gram blocking with this, union them, since each catches cases
    the other misses.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    s1_emb = encode_records(s1_df, model, batch_size=batch_size)
    other_emb = encode_records(other_df, model, batch_size=batch_size)

    index, id_array = build_ann_index(other_emb, other_df["entity_id"].tolist())
    return query_ann(index, id_array, s1_emb, s1_df["entity_id"].tolist(), top_k=top_k)
