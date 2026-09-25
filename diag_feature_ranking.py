from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Paths, BlockingConfig, COUNTRY, ENTITY_ID, GT_S1
from io_utils import load_all_sources
from blocking import build_normalized_frame, generate_candidates
from build_pairs import label_candidates, report_blocking_recall
from features import compute_features


DATA_ROOT = Path(r".\tests\real_aligned_sample")
CAPS = [60, 100, 150]


def evaluate_rank(pairs, gt, score_name, score):
    x = pairs[["entity_id", "candidate_id"]].copy()
    x["rank_score"] = score.to_numpy()

    # Stable descending sort: reproducible tie handling.
    x = x.sort_values(
        ["entity_id", "rank_score"],
        ascending=[True, False],
        kind="mergesort",
    )

    for cap in CAPS:
        selected = (
            x.groupby("entity_id", group_keys=False)
            .head(cap)
        )

        labeled = label_candidates(selected, gt)
        report = report_blocking_recall(labeled, gt)

        print(
            f"  {score_name:28s} cap={cap:3d} | "
            f"pair={report['pair_level_recall']:.4f} | "
            f"entity-full={report['entity_level_full_recall']:.4f}"
        )


def main():
    paths = Paths(root=DATA_ROOT)
    bcfg = BlockingConfig()

    s1, s2, s3, gt = load_all_sources(paths, split="train")

    print()
    print("=" * 90)
    print("FEATURE-AWARE PRE-CAP RANKING EXPERIMENT")
    print("=" * 90)

    for country in sorted(s1[COUNTRY].unique()):
        print()
        print("=" * 90)
        print(f"COUNTRY: {country}")
        print("=" * 90)

        s1_c = s1[s1[COUNTRY] == country].reset_index(drop=True)
        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]
        gt_c = gt[gt[GT_S1].isin(s1_c[ENTITY_ID])].copy()

        t0 = time.time()

        s1n = build_normalized_frame(s1_c, bcfg.ngram_n)
        s2n = build_normalized_frame(s2_c, bcfg.ngram_n)
        s3n = build_normalized_frame(s3_c, bcfg.ngram_n)

        # Candidate generation is exactly the existing blocker.
        cand2 = generate_candidates(s1n, s2n, bcfg)
        cand3 = generate_candidates(s1n, s3n, bcfg)
        cand = pd.concat([cand2, cand3], ignore_index=True)

        print(f"Raw candidates: {len(cand):,}")
        print(f"Blocking time: {time.time() - t0:.1f}s")

        # ------------------------------------------------------------
        # Compute the existing production features.
        # These are NOT labels and are therefore valid for ranking.
        # ------------------------------------------------------------
        t1 = time.time()

        other_norm = pd.concat([s2n, s3n], ignore_index=True)

        feats = compute_features(
            cand,
            s1n,
            other_norm,
        )

        print(
            f"Feature computation: "
            f"{time.time() - t1:.1f}s"
        )

        # Sanity check alignment.
        assert len(feats) == len(cand)

        # ------------------------------------------------------------
        # Existing baseline.
        # ------------------------------------------------------------
        print()
        print("BASELINE: current n_keys ranking")

        evaluate_rank(
            cand,
            gt_c,
            "n_keys",
            cand["n_keys"].astype(float),
        )

        # ------------------------------------------------------------
        # Ranking 1: prioritize exact name equality first.
        # ------------------------------------------------------------
        print()
        print("RANKING 1: exact names + similarities")

        score_exact = (
            1000.0 * feats["name_exact_clean"]
            + 900.0 * feats["name_exact_core"]
            + 800.0 * feats["name_exact_ascii"]
            + 100.0 * feats["name_token_jaccard"]
            + 100.0 * feats["name_char_jaccard"]
            + 50.0 * feats["name_levenshtein_ratio"]
            + 40.0 * feats["addr_token_jaccard"]
            + 30.0 * feats["addr_number_jaccard"]
            + 20.0 * feats["addr_number_any_shared"]
            + 10.0 * feats["n_keys"]
        )

        evaluate_rank(
            cand,
            gt_c,
            "exact+name+address",
            score_exact,
        )

        # ------------------------------------------------------------
        # Ranking 2: name-heavy but retain n_keys strongly.
        # ------------------------------------------------------------
        print()
        print("RANKING 2: n_keys + name similarity")

        score_name = (
            100.0 * feats["n_keys"]
            + 40.0 * feats["name_exact_clean"]
            + 35.0 * feats["name_exact_core"]
            + 30.0 * feats["name_exact_ascii"]
            + 25.0 * feats["name_token_jaccard"]
            + 25.0 * feats["name_char_jaccard"]
            + 20.0 * feats["name_levenshtein_ratio"]
            + 15.0 * feats["name_token_sort_ratio"]
            + 10.0 * feats["name_partial_ratio"]
            + 10.0 * feats["addr_token_jaccard"]
            + 10.0 * feats["addr_number_jaccard"]
            + 5.0 * feats["addr_number_any_shared"]
        )

        evaluate_rank(
            cand,
            gt_c,
            "n_keys+name_similarity",
            score_name,
        )

        # ------------------------------------------------------------
        # Ranking 3: emphasize exact/address evidence.
        # ------------------------------------------------------------
        print()
        print("RANKING 3: strong exact/address evidence")

        score_precise = (
            100.0 * feats["n_keys"]
            + 150.0 * feats["name_exact_clean"]
            + 140.0 * feats["name_exact_core"]
            + 130.0 * feats["name_exact_ascii"]
            + 40.0 * feats["addr_number_any_shared"]
            + 35.0 * feats["addr_number_jaccard"]
            + 30.0 * feats["name_token_jaccard"]
            + 30.0 * feats["name_char_jaccard"]
            + 25.0 * feats["addr_token_jaccard"]
            + 20.0 * feats["name_levenshtein_ratio"]
        )

        evaluate_rank(
            cand,
            gt_c,
            "precise_exact_address",
            score_precise,
        )

        del feats, cand, cand2, cand3
        del s1n, s2n, s3n, other_norm

    print()
    print("Done.")


if __name__ == "__main__":
    main()
