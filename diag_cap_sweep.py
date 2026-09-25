from pathlib import Path
import time

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Paths, BlockingConfig, COUNTRY, ENTITY_ID, GT_S1
from io_utils import load_all_sources
from blocking import build_normalized_frame, generate_candidates, cap_candidates
from build_pairs import label_candidates, report_blocking_recall


DATA_ROOT = Path(r".\tests\real_aligned_sample")
CHUNK_SIZE = 25_000

CAPS = [60, 80, 100, 120, 150, 200, 300]


def main():
    paths = Paths(root=DATA_ROOT)
    bcfg = BlockingConfig()

    print("Loading aligned real-data sample...")
    t0 = time.time()
    s1, s2, s3, gt = load_all_sources(paths, split="train")
    print(f"Loaded in {time.time() - t0:.1f}s")

    print()
    print("=" * 90)
    print("CANDIDATE CAP SWEEP")
    print("=" * 90)
    print(f"Caps: {CAPS}")
    print()

    rows = []

    for country in sorted(s1[COUNTRY].unique()):
        s1_c = s1[s1[COUNTRY] == country].reset_index(drop=True)
        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]
        gt_c = gt[gt[GT_S1].isin(s1_c[ENTITY_ID])]

        print(
            f"\n=== {country}: "
            f"S1={len(s1_c):,}, "
            f"S2={len(s2_c):,}, "
            f"S3={len(s3_c):,} ==="
        )

        t0 = time.time()
        s2n = build_normalized_frame(s2_c, bcfg.ngram_n)
        s3n = build_normalized_frame(s3_c, bcfg.ngram_n)

        s1n = build_normalized_frame(s1_c, bcfg.ngram_n)

        cand2 = generate_candidates(s1n, s2n, bcfg)
        cand3 = generate_candidates(s1n, s3n, bcfg)
        cand = pd.concat([cand2, cand3], ignore_index=True)

        print(
            f"Raw candidates: {len(cand):,} "
            f"({time.time() - t0:.1f}s)"
        )

        precap_labeled = label_candidates(cand, gt_c)
        precap = report_blocking_recall(precap_labeled, gt_c)

        print(
            f"PRE-CAP | candidates={len(cand):,} | "
            f"pair={precap['pair_level_recall']:.4f} | "
            f"entity-full={precap['entity_level_full_recall']:.4f}"
        )

        # Evaluate every cap against exactly the same raw candidates.
        for cap in CAPS:
            capped = cap_candidates(cand, cap)
            labeled = label_candidates(capped, gt_c)
            report = report_blocking_recall(labeled, gt_c)

            avg_candidates = len(capped) / len(s1_c)

            row = {
                "country": country,
                "cap": cap,
                "candidates": len(capped),
                "avg_candidates_per_s1": avg_candidates,
                "matches_found": report["matches_found_in_candidates"],
                "total_true": report["total_true_matches"],
                "pair_recall": report["pair_level_recall"],
                "entities_fully_recalled": report["entities_fully_recalled"],
                "entities_with_matches": report["entities_with_matches"],
                "entity_full_recall": report["entity_level_full_recall"],
            }

            rows.append(row)

            print(
                f"  cap={cap:3d} | "
                f"candidates={len(capped):,} | "
                f"avg/S1={avg_candidates:.1f} | "
                f"pair={report['pair_level_recall']:.4f} | "
                f"entity-full={report['entity_level_full_recall']:.4f}"
            )

    result = pd.DataFrame(rows)
    out = Path(".\cap_sweep.csv")
    result.to_csv(out, index=False)

    print()
    print("=" * 90)
    print("CAP SWEEP RESULTS")
    print("=" * 90)

    print(
        result[
            [
                "country",
                "cap",
                "candidates",
                "avg_candidates_per_s1",
                "pair_recall",
                "entity_full_recall",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
