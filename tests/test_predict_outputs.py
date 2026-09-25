import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from predict import build_outputs


def test_every_entity_gets_exactly_one_row_including_singletons():
    all_s1 = ["S1-1", "S1-2", "S1-3"]  # S1-3 will get zero candidates entirely
    candidates = pd.DataFrame({
        "entity_id": ["S1-1", "S1-1", "S1-2"],
        "candidate_id": ["S2-1", "S2-2", "S3-1"],
    })
    scored = pd.DataFrame({
        "entity_id": ["S1-1", "S1-1", "S1-2"],
        "candidate_id": ["S2-1", "S2-2", "S3-1"],
        "score": [0.9, 0.3, 0.9],  # S2-2 below threshold
    })
    cand_df, match_df = build_outputs(
        all_s1, candidates, scored, threshold=0.5,
        valid_s2_ids={"S2-1", "S2-2"}, valid_s3_ids={"S3-1"})

    assert len(cand_df) == 3 and len(match_df) == 3
    assert set(cand_df["source1_entity_id"]) == set(all_s1)

    row3_cand = cand_df.loc[cand_df["source1_entity_id"] == "S1-3", "candidate_entity_ids"].iloc[0]
    row3_match = match_df.loc[match_df["source1_entity_id"] == "S1-3", "matched_entity_ids"].iloc[0]
    assert row3_cand == "" and row3_match == "", "entity with zero candidates must get empty string, not be dropped"

    row1_match = match_df.loc[match_df["source1_entity_id"] == "S1-1", "matched_entity_ids"].iloc[0]
    assert row1_match == "S2-1", "only the above-threshold candidate should survive into matching_results"

    row1_cand = cand_df.loc[cand_df["source1_entity_id"] == "S1-1", "candidate_entity_ids"].iloc[0]
    assert set(row1_cand.split(",")) == {"S2-1", "S2-2"}, "candidate_pairs must keep BOTH candidates regardless of score"


def test_matched_is_subset_of_candidates():
    all_s1 = ["S1-1"]
    candidates = pd.DataFrame({"entity_id": ["S1-1"], "candidate_id": ["S2-1"]})
    scored = pd.DataFrame({"entity_id": ["S1-1"], "candidate_id": ["S2-1"], "score": [0.99]})
    cand_df, match_df = build_outputs(all_s1, candidates, scored, 0.5, {"S2-1"}, set())
    matched = set(match_df["matched_entity_ids"].iloc[0].split(","))
    cands = set(cand_df["candidate_entity_ids"].iloc[0].split(","))
    assert matched <= cands


def test_invalid_id_raises():
    """A matched id that isn't a real test S2/S3 id must fail loudly, not
    silently write a submission that gets rejected by the real validator."""
    all_s1 = ["S1-1"]
    candidates = pd.DataFrame({"entity_id": ["S1-1"], "candidate_id": ["S2-999"]})
    scored = pd.DataFrame({"entity_id": ["S1-1"], "candidate_id": ["S2-999"], "score": [0.99]})
    try:
        build_outputs(all_s1, candidates, scored, 0.5, valid_s2_ids=set(), valid_s3_ids=set())
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected an AssertionError for an id not in the valid test set"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
