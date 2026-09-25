from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Paths, BlockingConfig, COUNTRY, ENTITY_ID, GT_S1
from io_utils import load_all_sources
from blocking import build_normalized_frame, generate_candidates
from build_pairs import label_candidates


DATA_ROOT = Path(r".\tests\real_aligned_sample_b")

CAPS = [60, 100, 150, 200, 300, 500, 1000]


def main():
    paths = Paths(root=DATA_ROOT)
    bcfg = BlockingConfig()

    print("Loading aligned real-data sample...")
    s1, s2, s3, gt = load_all_sources(paths, split="train")

    all_rows = []

    for country in sorted(s1[COUNTRY].unique()):
        print()
        print("=" * 80)
        print(f"COUNTRY: {country}")
        print("=" * 80)

        s1_c = s1[s1[COUNTRY] == country].reset_index(drop=True)
        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]
        gt_c = gt[gt[GT_S1].isin(s1_c[ENTITY_ID])]

        t0 = time.time()

        s1n = build_normalized_frame(s1_c, bcfg.ngram_n)
        s2n = build_normalized_frame(s2_c, bcfg.ngram_n)
        s3n = build_normalized_frame(s3_c, bcfg.ngram_n)

        cand2 = generate_candidates(s1n, s2n, bcfg)
        cand3 = generate_candidates(s1n, s3n, bcfg)

        cand = pd.concat([cand2, cand3], ignore_index=True)

        print(f"Raw candidates: {len(cand):,}")
        print(f"Generation time: {time.time() - t0:.1f}s")

        labeled = label_candidates(cand, gt_c)

        # This reproduces the EXACT ordering currently used by cap_candidates().
        ranked = labeled.sort_values(
            ["entity_id", "n_keys"],
            ascending=[True, False],
            kind="mergesort",
        ).copy()

        ranked["rank"] = ranked.groupby("entity_id").cumcount() + 1

        positives = ranked[ranked["label"] == 1].copy()

        total_true = 0
        for value in gt_c["matched_entity_ids"]:
            if value:
                total_true += len(
                    [x for x in value.split(",") if x.strip()]
                )

        found_true = len(positives)

        print()
        print(f"True matches in GT: {total_true:,}")
        print(f"True matches surfaced pre-cap: {found_true:,}")
        print(
            f"Blocking ceiling: "
            f"{found_true / total_true:.4f}"
        )

        if not positives.empty:
            print()
            print("TRUE-MATCH n_keys distribution")
            print(
                positives["n_keys"]
                .value_counts()
                .sort_index()
                .to_string()
            )

            print()
            print("TRUE-MATCH rank quantiles")
            print(
                positives["rank"]
                .quantile([0.50, 0.75, 0.90, 0.95, 0.99])
                .to_string()
            )

            print()
            print("TRUE-MATCH rank summary")
            print(positives["rank"].describe().to_string())

            print()
            print("RECALL IF WE RANK ONLY BY CURRENT n_keys")
            for cap in CAPS:
                found = int((positives["rank"] <= cap).sum())
                recall_vs_gt = found / total_true if total_true else 0.0
                recall_vs_pre = found / found_true if found_true else 0.0

                avg_candidates = min(
                    cap,
                    int(
                        ranked.groupby("entity_id").size().mean()
                    )
                )

                print(
                    f"  cap={cap:4d} | "
                    f"true={found:,}/{total_true:,} | "
                    f"recall_vs_GT={recall_vs_gt:.4f} | "
                    f"recall_vs_PRECAP={recall_vs_pre:.4f} | "
                    f"avg_cap={avg_candidates}"
                )

            # How many true matches are supported by >=2 blocking keys?
            multi_key = int((positives["n_keys"] >= 2).sum())
            single_key = int((positives["n_keys"] == 1).sum())

            print()
            print(
                f"True matches with n_keys=1:  {single_key:,} "
                f"({single_key / found_true:.2%})"
            )
            print(
                f"True matches with n_keys>=2: {multi_key:,} "
                f"({multi_key / found_true:.2%})"
            )

        ranked.to_parquet(
            Path(f".ranked_{country}.parquet"),
            index=False,
        )

        all_rows.append(
            {
                "country": country,
                "total_true": total_true,
                "found_pre_cap": found_true,
                "blocking_ceiling": (
                    found_true / total_true
                    if total_true else np.nan
                ),
            }
        )

    result = pd.DataFrame(all_rows)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()


