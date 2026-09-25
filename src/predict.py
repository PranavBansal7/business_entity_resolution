"""
Run the trained model on the test set and produce both required outputs:

  output/candidate_pairs.tsv   <- the blocking output, LAST stage before
                                   the model scores it (exactly what
                                   compute_features ran on)
  output/matching_results.tsv  <- candidate_pairs.tsv filtered by the
                                   tuned threshold

Usage:
    python src/predict.py --data-root /path/to/student_resource/dataset \
        --work-dir ./work --output-dir ./output

Guarantees enforced here (matching the brief's validation rules exactly):
  - every test source1 entity gets exactly one output row, including ones
    blocking found zero candidates for (empty matched_entity_ids)
  - no duplicate entity IDs within an ID list
  - matched_entity_ids is a subset of candidate_entity_ids for every row
  - only S2-/S3- ids that actually exist in the test set ever appear
    (guaranteed by construction: candidates are only ever drawn from the
    test_source2/test_source3 frames, so nothing else can leak in — the
    assertions below just make that explicit and fail loudly if it's ever
    violated instead of silently submitting something invalid)
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from config import Paths, BlockingConfig, COUNTRY, ENTITY_ID
from io_utils import load_all_sources, write_tsv
from blocking import build_normalized_frame, generate_candidates, cap_candidates
from features import compute_features_parallel, FEATURE_COLUMNS


def ids_to_str(ids) -> str:
    """`ids` arrives as a pandas Series when used via groupby(...).apply —
    `if ids:` on a Series raises (ambiguous truth value), so check length
    explicitly instead of relying on truthiness."""
    ids = list(ids)
    return ",".join(sorted(set(ids))) if len(ids) > 0 else ""


def build_outputs(all_s1_ids, all_candidates: pd.DataFrame, all_scored: pd.DataFrame,
                   threshold: float, valid_s2_ids: set, valid_s3_ids: set):
    """Pure function (no file I/O) so it's directly unit-testable — see
    tests/test_predict_outputs.py. Returns (candidate_pairs_df,
    matching_results_df); raises AssertionError on any format violation
    described in the module docstring."""
    cand_grouped = (all_candidates.groupby(ENTITY_ID)["candidate_id"]
                     .apply(ids_to_str).to_dict()) if not all_candidates.empty else {}
    candidate_pairs_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "candidate_entity_ids": [cand_grouped.get(sid, "") for sid in all_s1_ids],
    })

    kept = all_scored[all_scored["score"] >= threshold] if not all_scored.empty else all_scored
    match_grouped = (kept.groupby(ENTITY_ID)["candidate_id"]
                      .apply(ids_to_str).to_dict()) if not kept.empty else {}
    matching_results_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": [match_grouped.get(sid, "") for sid in all_s1_ids],
    })

    assert len(candidate_pairs_df) == len(all_s1_ids)
    assert len(matching_results_df) == len(all_s1_ids)
    assert not candidate_pairs_df["source1_entity_id"].duplicated().any()
    assert not matching_results_df["source1_entity_id"].duplicated().any()

    for ids_str in match_grouped.values():
        if not ids_str:
            continue
        ids = ids_str.split(",")
        assert len(ids) == len(set(ids)), "duplicate id within a matched list"
        for i in ids:
            assert (i.startswith("S2-") and i in valid_s2_ids) or \
                   (i.startswith("S3-") and i in valid_s3_ids), \
                   f"matched id {i} not a valid test S2/S3 id"

    matched_set_per_entity = {k: set(v.split(",")) if v else set()
                               for k, v in match_grouped.items()}
    cand_set_per_entity = {k: set(v.split(",")) if v else set()
                            for k, v in cand_grouped.items()}
    for sid, matched in matched_set_per_entity.items():
        assert matched <= cand_set_per_entity.get(sid, set()), \
            f"{sid}: matched id(s) not present in its own candidate set — pipeline bug"

    return candidate_pairs_df, matching_results_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset")
    ap.add_argument("--work-dir", default="./work")
    ap.add_argument("--output-dir", default="./output")
    ap.add_argument("--threshold", type=float, default=None,
                     help="override the tuned threshold from run_metadata.json")
    args = ap.parse_args()

    paths = Paths(root=Path(args.data_root), work_dir=Path(args.work_dir),
                  output_dir=Path(args.output_dir))
    bcfg = BlockingConfig()

    model = joblib.load(paths.work_dir / "model.joblib")
    with open(paths.work_dir / "run_metadata.json") as f:
        meta = json.load(f)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    print(f"using threshold={threshold}")

    s1, s2, s3, _ = load_all_sources(paths, split="test")
    valid_s2_ids = set(s2[ENTITY_ID])
    valid_s3_ids = set(s3[ENTITY_ID])
    all_s1_ids = list(s1[ENTITY_ID])

    countries = sorted(s1[COUNTRY].unique())
    print(f"countries in test data: {countries}")

    candidate_parts, scored_parts = [], []
    for country in countries:
        s1_c = s1[s1[COUNTRY] == country]
        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]

        s1n = build_normalized_frame(s1_c, bcfg.ngram_n)
        s2n = build_normalized_frame(s2_c, bcfg.ngram_n)
        s3n = build_normalized_frame(s3_c, bcfg.ngram_n)
        other_norm = pd.concat([s2n, s3n], ignore_index=True)

        cand2 = generate_candidates(s1n, s2n, bcfg)
        cand3 = generate_candidates(s1n, s3n, bcfg)
        cand = cap_candidates(pd.concat([cand2, cand3], ignore_index=True),
                               bcfg.max_candidates_per_entity)
        print(f"  [{country}] {len(s1_c):,} S1 rows -> {len(cand):,} candidate pairs")
        if cand.empty:
            continue

        feats = compute_features_parallel(cand, s1n, other_norm)
        scores = model.predict_proba(feats[FEATURE_COLUMNS])[:, 1]
        scored = feats[["entity_id", "candidate_id"]].copy()
        scored["score"] = scores

        candidate_parts.append(cand[[ENTITY_ID, "candidate_id"]])
        scored_parts.append(scored)

    all_candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else \
        pd.DataFrame(columns=[ENTITY_ID, "candidate_id"])
    all_scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else \
        pd.DataFrame(columns=[ENTITY_ID, "candidate_id", "score"])

    candidate_pairs_df, matching_results_df = build_outputs(
        all_s1_ids, all_candidates, all_scored, threshold, valid_s2_ids, valid_s3_ids)

    write_tsv(candidate_pairs_df, paths.output_dir / "candidate_pairs.tsv")
    write_tsv(matching_results_df, paths.output_dir / "matching_results.tsv")

    n_with_match = (matching_results_df["matched_entity_ids"] != "").sum()
    print(f"\n{n_with_match:,} / {len(all_s1_ids):,} test entities got >=1 match "
          f"({n_with_match/len(all_s1_ids):.1%})")


if __name__ == "__main__":
    main()
