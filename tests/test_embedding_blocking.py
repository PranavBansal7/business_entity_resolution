import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from embedding_blocking import build_ann_index, query_ann


def test_ann_recovers_exact_matches():
    """If a query vector IS one of the indexed vectors, it must come back
    as its own top hit with similarity ~1.0 — the most basic correctness
    check for the ID mapping (faiss returns integer positions; a bug here
    would silently attach the WRONG candidate_id to every result)."""
    rng = np.random.default_rng(0)
    n, d = 200, 64
    vecs = rng.normal(size=(n, d)).astype("float32")
    ids = [f"S2-{i:04d}" for i in range(n)]

    index, id_array = build_ann_index(vecs.copy(), ids)
    result = query_ann(index, id_array, vecs.copy(), ids, top_k=5)

    for qid in ids[:20]:  # spot-check
        top = result[result["entity_id"] == qid].sort_values("embedding_score", ascending=False).iloc[0]
        assert top["candidate_id"] == qid, f"{qid} should recover itself as top-1"
        assert top["embedding_score"] > 0.999, f"self-similarity should be ~1.0, got {top['embedding_score']}"


def test_ann_ranks_nearby_vectors_higher():
    """A vector nudged slightly from a target should rank that target
    above unrelated random vectors — checks the similarity ORDERING is
    sane, not just that self-matches work."""
    rng = np.random.default_rng(1)
    d = 32
    target = rng.normal(size=(d,)).astype("float32")
    near = (target + rng.normal(scale=0.05, size=(d,)).astype("float32"))
    far_ones = rng.normal(size=(50, d)).astype("float32")

    corpus = np.vstack([target[None, :], far_ones])
    ids = ["TARGET"] + [f"FAR-{i}" for i in range(len(far_ones))]
    index, id_array = build_ann_index(corpus, ids)

    result = query_ann(index, id_array, near[None, :], ["QUERY"], top_k=5)
    top = result.sort_values("embedding_score", ascending=False).iloc[0]
    assert top["candidate_id"] == "TARGET", "the nudged vector should still rank its near-neighbor first"


def test_top_k_respected_and_capped_at_index_size():
    rng = np.random.default_rng(2)
    vecs = rng.normal(size=(10, 16)).astype("float32")
    ids = [f"S3-{i}" for i in range(10)]
    index, id_array = build_ann_index(vecs, ids)

    result = query_ann(index, id_array, vecs[:1], ["Q"], top_k=1000)  # ask for more than exist
    assert len(result) == 10, "top_k should be silently capped at index size, not error"


def test_empty_index_returns_empty_frame_not_crash():
    index, id_array = build_ann_index(np.zeros((0, 8), dtype="float32"), [])
    result = query_ann(index, id_array, np.zeros((2, 8), dtype="float32"), ["Q1", "Q2"], top_k=5)
    assert len(result) == 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
