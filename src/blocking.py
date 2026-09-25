"""
Candidate generation ("blocking").

At full scale (~2.2M Source-1 x ~10.4M Source-2/3 in train, similar order
in test) a naive cross product is ~10^13 pairs — impossible. Everything
here is built as inverted-index LOOKUPS implemented via pandas hash-merges
(vectorized, not Python-level nested loops), which is what makes this
tractable on a single machine.

Strategy: build several independent blocking keys, join each one
separately (S1 rows exploded to one-row-per-key, merged against S2/S3 rows
exploded the same way), then take the UNION of all candidate pairs found
by any key. Recall is the priority here — the model in features.py /
train_model.py is what narrows candidates down for precision.

Blocking keys used:
  1. name_token   — any shared significant token in the normalized name
  2. name_prefix  — first 4 chars of the core (suffix-stripped) name
  3. addr_number  — any shared digit run in the address (house/plot/PIN/ZIP)
  4. name_ngram   — shared character 4-grams of the core name (typo-tolerant,
                     and the main fallback when suffix/token blocking misses)

All blocking is done WITHIN country (S1 record only ever blocks against
S2/S3 records with the same country string) — country is treated as an
opaque label, so this works unchanged for a country never seen in training
(e.g. France), as long as the test data itself provides consistent country
strings between sources.
"""
import pandas as pd

from config import BlockingConfig, ENTITY_ID, COUNTRY
from text_normalize import normalize_record


def build_normalized_frame(df: pd.DataFrame, ngram_n: int) -> pd.DataFrame:
    """Vectorized-ish wrapper: apply normalize_record row-wise and expand
    into columns blocking.py and features.py both consume."""
    records = [
        normalize_record(row[ENTITY_ID], row.get("business_name", ""),
                          row.get("business_address", ""), row[COUNTRY],
                          ngram_n=ngram_n)
        for row in df.to_dict("records")
    ]
    out = pd.DataFrame({
        ENTITY_ID: [r.entity_id for r in records],
        COUNTRY: [r.country for r in records],
        "name_clean": [r.name_clean for r in records],
        "name_core": [r.name_core for r in records],
        "name_suffix": [r.name_suffix for r in records],
        "name_ascii": [r.name_ascii for r in records],
        "name_tokens": [r.name_tokens for r in records],
        "name_ngrams": [r.name_ngrams for r in records],
        "addr_clean": [r.addr_clean for r in records],
        "addr_tokens": [r.addr_tokens for r in records],
        "addr_numbers": [r.addr_numbers for r in records],
        "landmark": [r.landmark for r in records],
    })
    return out


def _explode_key(df: pd.DataFrame, set_col: str, max_postings: int) -> pd.DataFrame:
    """One row per (entity_id, key) from a set-valued column. Drops keys
    that are too common to be useful for blocking (e.g. a name token that
    appears in thousands of records contributes noise, not signal) — capped
    at an ABSOLUTE number of postings (`max_postings`), not a percentage of
    the frame, so the cap stays meaningful whether the frame has 3K or 5M
    rows."""
    exploded = df[[ENTITY_ID, COUNTRY, set_col]].explode(set_col)
    exploded = exploded.dropna(subset=[set_col])
    exploded = exploded[exploded[set_col].astype(str).str.len() > 0]
    if len(exploded) == 0:
        return exploded
    freq = exploded.groupby([COUNTRY, set_col])[ENTITY_ID].transform("count")
    exploded = exploded[freq <= max_postings]
    return exploded


def _join_on_key(s1_exp: pd.DataFrame, other_exp: pd.DataFrame, set_col: str,
                  key_name: str) -> pd.DataFrame:
    if len(s1_exp) == 0 or len(other_exp) == 0:
        return pd.DataFrame(columns=[ENTITY_ID, "candidate_id", "match_key"])
    merged = s1_exp.merge(
        other_exp, on=[COUNTRY, set_col], suffixes=("_s1", "_cand")
    )
    if merged.empty:
        return pd.DataFrame(columns=[ENTITY_ID, "candidate_id", "match_key"])
    out = merged[[f"{ENTITY_ID}_s1", f"{ENTITY_ID}_cand"]].rename(
        columns={f"{ENTITY_ID}_s1": ENTITY_ID, f"{ENTITY_ID}_cand": "candidate_id"}
    )
    out["match_key"] = key_name
    return out.drop_duplicates()


def generate_candidates(s1_norm: pd.DataFrame, other_norm: pd.DataFrame,
                         cfg: BlockingConfig) -> pd.DataFrame:
    """other_norm is the normalized S2 or S3 frame (call once per source and
    concat the results — kept separate so callers can track candidate
    provenance / apply per-source limits if needed).

    Returns columns: entity_id (S1), candidate_id, match_key (list of which
    blocking keys fired, comma-joined), n_keys (how many independent keys
    agreed — a cheap, useful blocking-strength feature for the model).
    """
    pieces = []

    if cfg.use_name_token_blocking:
        s1_tok = s1_norm.assign(
            name_tokens=s1_norm["name_tokens"].apply(
                lambda s: {t for t in s if len(t) >= cfg.min_token_len}))
        other_tok = other_norm.assign(
            name_tokens=other_norm["name_tokens"].apply(
                lambda s: {t for t in s if len(t) >= cfg.min_token_len}))
        s1_exp = _explode_key(s1_tok, "name_tokens", cfg.max_postings_per_key)
        other_exp = _explode_key(other_tok, "name_tokens", cfg.max_postings_per_key)
        pieces.append(_join_on_key(s1_exp, other_exp, "name_tokens", "name_token"))

    if cfg.use_name_prefix_blocking:
        s1_pref = s1_norm.assign(prefix=s1_norm["name_core"].str[:4].apply(
            lambda p: frozenset([p]) if len(p) >= 3 else frozenset()))
        other_pref = other_norm.assign(prefix=other_norm["name_core"].str[:4].apply(
            lambda p: frozenset([p]) if len(p) >= 3 else frozenset()))
        s1_exp = _explode_key(s1_pref, "prefix", cfg.max_postings_per_key)
        other_exp = _explode_key(other_pref, "prefix", cfg.max_postings_per_key)
        pieces.append(_join_on_key(s1_exp, other_exp, "prefix", "name_prefix"))

    if cfg.use_address_number_blocking:
        # Require numbers with >=2 digits BEFORE capping — single digits
        # (apartment/unit numbers) are too common to be a useful key.
        s1_multi = s1_norm.assign(addr_numbers=s1_norm["addr_numbers"].apply(
            lambda s: {n for n in s if len(n) >= 2}))
        other_multi = other_norm.assign(addr_numbers=other_norm["addr_numbers"].apply(
            lambda s: {n for n in s if len(n) >= 2}))
        s1_exp = _explode_key(s1_multi, "addr_numbers", cfg.max_postings_per_key_numbers)
        other_exp = _explode_key(other_multi, "addr_numbers", cfg.max_postings_per_key_numbers)
        pieces.append(_join_on_key(s1_exp, other_exp, "addr_numbers", "addr_number"))

    if cfg.use_char_ngram_blocking:
        s1_exp = _explode_key(s1_norm, "name_ngrams", cfg.max_postings_per_key)
        other_exp = _explode_key(other_norm, "name_ngrams", cfg.max_postings_per_key)
        pieces.append(_join_on_key(s1_exp, other_exp, "name_ngrams", "name_ngram"))

    if not pieces:
        return pd.DataFrame(columns=[ENTITY_ID, "candidate_id", "n_keys", "match_keys"])

    all_candidates = pd.concat(pieces, ignore_index=True)
    if all_candidates.empty:
        return pd.DataFrame(columns=[ENTITY_ID, "candidate_id", "n_keys", "match_keys"])

    agg = (
        all_candidates.groupby([ENTITY_ID, "candidate_id"])["match_key"]
        .agg(lambda keys: ",".join(sorted(set(keys))))
        .reset_index()
        .rename(columns={"match_key": "match_keys"})
    )
    agg["n_keys"] = agg["match_keys"].str.count(",") + 1
    return agg


def cap_candidates(candidates: pd.DataFrame, max_per_entity: int) -> pd.DataFrame:
    """Keep at most `max_per_entity` candidates per S1 entity, preferring
    ones supported by more independent blocking keys (n_keys) as the cheap
    pre-model relevance signal."""
    if candidates.empty or max_per_entity is None:
        return candidates
    candidates = candidates.sort_values("n_keys", ascending=False)
    return (
        candidates.groupby(ENTITY_ID, group_keys=False)
        .head(max_per_entity)
        .reset_index(drop=True)
    )
