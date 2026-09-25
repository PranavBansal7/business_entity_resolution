"""
diag_blocking_full.py — blocking-only diagnostic on the REAL full dataset.

Measures PRE-CAP and POST-CAP blocking recall (pair-level and entity-full-
level) WITHOUT running feature extraction or the classifier. Answers one
question before any expensive model training: does the blocker actually
find the true matches, on the real data?

This is NOT the tests/chunked_sample fixture, which is missing 17,335 of
its own 17,359 referenced GT target IDs from its S2/S3 files and cannot
measure recall at all. This script loads the full train_source1/2/3 +
train_ground_truth files directly.

Design note: S1 is processed in chunks (for progress visibility and, in
the non-pilot case, to bound peak memory of the candidate/label frames),
but S2/S3 are normalized ONCE per country and reused across every chunk —
each S1 chunk's candidates must be generated against the FULL S2/S3 pool
for that country, never a subsample of it. Subsampling S2/S3 is exactly
the chunked_sample mistake being avoided here.

Usage (from the business_entity_resolution/ directory):
    python .\\diag_blocking_full.py

Edit PILOT_ONLY / CHUNK_SIZE / DATA_ROOT below, then re-run.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from config import Paths, BlockingConfig, COUNTRY, ENTITY_ID, GT_S1
from io_utils import load_all_sources
from blocking import build_normalized_frame, generate_candidates, cap_candidates
from build_pairs import label_candidates, report_blocking_recall

# ---------------------------------------------------------------------
# EDIT THESE
DATA_ROOT = Path(r".\tests\real_aligned_sample")
CHUNK_SIZE = 25_000
PILOT_ONLY = True   # True: only the first CHUNK_SIZE S1 rows per country.
                     # False: the full dataset, chunked, all countries.
# ---------------------------------------------------------------------


def _empty_totals():
    return dict(total_true_matches=0, matches_found_in_candidates=0,
                entities_with_matches=0, entities_fully_recalled=0)


def _accumulate(totals, report):
    for k in totals:
        totals[k] += report[k]


def _finalize(totals):
    pair_recall = (totals["matches_found_in_candidates"] / totals["total_true_matches"]
                   if totals["total_true_matches"] else float("nan"))
    entity_recall = (totals["entities_fully_recalled"] / totals["entities_with_matches"]
                      if totals["entities_with_matches"] else float("nan"))
    return dict(totals, pair_level_recall=pair_recall,
                entity_level_full_recall=entity_recall)


def main():
    paths = Paths(root=DATA_ROOT)
    bcfg = BlockingConfig()  # current defaults, unmodified — per the plan

    print(f"loading full dataset from {DATA_ROOT.resolve()} ...")
    t0 = time.time()
    s1, s2, s3, gt = load_all_sources(paths, split="train")
    print(f"loaded in {time.time() - t0:.1f}s")

    countries = sorted(s1[COUNTRY].unique())
    print(f"countries: {countries}  |  PILOT_ONLY={PILOT_ONLY}  CHUNK_SIZE={CHUNK_SIZE:,}\n")

    precap_totals = _empty_totals()
    postcap_totals = _empty_totals()
    run_start = time.time()

    for country in countries:
        s1_c = s1[s1[COUNTRY] == country].reset_index(drop=True)
        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]

        print(f"=== {country}: {len(s1_c):,} S1 rows, {len(s2_c):,} S2 rows, "
              f"{len(s3_c):,} S3 rows ===")

        t0 = time.time()
        s2n = build_normalized_frame(s2_c, bcfg.ngram_n)
        s3n = build_normalized_frame(s3_c, bcfg.ngram_n)
        print(f"  normalized S2/S3 for {country} in {time.time() - t0:.1f}s "
              f"(done once, reused across every chunk below)")

        if PILOT_ONLY:
            chunks = [s1_c.head(CHUNK_SIZE)]
        else:
            chunks = [s1_c.iloc[i:i + CHUNK_SIZE] for i in range(0, len(s1_c), CHUNK_SIZE)]

        for ci, s1_chunk in enumerate(chunks):
            if s1_chunk.empty:
                continue
            try:
                t0 = time.time()
                s1n = build_normalized_frame(s1_chunk, bcfg.ngram_n)
                gt_chunk = gt[gt[GT_S1].isin(s1_chunk[ENTITY_ID])]

                cand2 = generate_candidates(s1n, s2n, bcfg)
                cand3 = generate_candidates(s1n, s3n, bcfg)
                cand = pd.concat([cand2, cand3], ignore_index=True)

                labeled_precap = label_candidates(cand, gt_chunk)
                precap_report = report_blocking_recall(labeled_precap, gt_chunk)
                _accumulate(precap_totals, precap_report)

                capped = cap_candidates(cand, bcfg.max_candidates_per_entity)
                labeled_postcap = label_candidates(capped, gt_chunk)
                postcap_report = report_blocking_recall(labeled_postcap, gt_chunk)
                _accumulate(postcap_totals, postcap_report)

                elapsed_total = time.time() - run_start
                print(f"  [{country} chunk {ci + 1}/{len(chunks)}] {len(s1_chunk):,} S1 rows, "
                      f"{len(cand):,} candidates PRE-CAP -> {len(capped):,} POST-CAP "
                      f"({time.time() - t0:.1f}s this chunk, {elapsed_total / 60:.1f}min total)")
                print(f"    PRE-CAP  pair_recall={precap_report['pair_level_recall']:.4f}  "
                      f"entity_full_recall={precap_report['entity_level_full_recall']:.4f}  "
                      f"(true_matches_in_chunk={precap_report['total_true_matches']:,})")
                print(f"    POST-CAP pair_recall={postcap_report['pair_level_recall']:.4f}  "
                      f"entity_full_recall={postcap_report['entity_level_full_recall']:.4f}")
            except Exception:
                print(f"  [{country} chunk {ci + 1}/{len(chunks)}] FAILED — skipping, "
                      f"totals below EXCLUDE this chunk:")
                traceback.print_exc()
        print()

    print("=" * 70)
    print(f"OVERALL  (PILOT_ONLY={PILOT_ONLY})")
    print("=" * 70)
    final_pre = _finalize(precap_totals)
    final_post = _finalize(postcap_totals)
    print("PRE-CAP :")
    for k, v in final_pre.items():
        print(f"   {k}: {v}")
    print("POST-CAP:")
    for k, v in final_post.items():
        print(f"   {k}: {v}")
    print(f"\ntotal wall time: {(time.time() - run_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
