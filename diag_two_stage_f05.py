from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Paths, BlockingConfig, TrainConfig, COUNTRY, ENTITY_ID, GT_S1
from io_utils import load_all_sources
from blocking import build_normalized_frame
from build_pairs import (
    label_candidates,
    entity_level_split,
    sample_training_pairs,
)
from features import compute_features, FEATURE_COLUMNS
from metric import load_gt_dict, f_beta_for_entity


DATA_ROOT = Path(r".\tests\real_aligned_sample_b")

TOP_K = 60
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def fit_matcher(train_pairs, s1n, other_norm, tcfg, label):
    sampled = sample_training_pairs(
        train_pairs,
        negatives_per_positive=tcfg.negatives_per_positive,
        random_seed=tcfg.random_seed,
    )

    X = compute_features(
        sampled,
        s1n,
        other_norm,
    )[FEATURE_COLUMNS]

    y = sampled["label"].astype(np.uint8).to_numpy()

    print(
        f"  {label}: training rows={len(X):,}, "
        f"positives={int(y.sum()):,}"
    )

    params = dict(tcfg.lgbm_params)
    params["n_estimators"] = 300

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    return model


def choose_threshold(
    model,
    val_pairs,
    val_gt,
    val_s1_ids,
    s1n,
    other_norm,
):
    feats = compute_features(
        val_pairs,
        s1n,
        other_norm,
    )[FEATURE_COLUMNS]

    scores = model.predict_proba(feats)[:, 1]

    scored = val_pairs[["entity_id", "candidate_id"]].copy()
    scored["score"] = scores

    gt_dict = load_gt_dict(val_gt)

    score_by_threshold = []

    for threshold in THRESHOLDS:
        score_sum = 0.0

        grouped = scored.groupby("entity_id")

        for eid in val_s1_ids:
            true_ids = gt_dict.get(eid, set())

            if eid in grouped.groups:
                group = grouped.get_group(eid)
                pred_ids = set(
                    group.loc[
                        group["score"] >= threshold,
                        "candidate_id",
                    ]
                )
            else:
                pred_ids = set()

            score_sum += f_beta_for_entity(
                true_ids,
                pred_ids,
                beta=0.5,
            )

        macro = score_sum / len(val_s1_ids)

        score_by_threshold.append(
            (threshold, macro)
        )

    best_threshold, best_score = max(
        score_by_threshold,
        key=lambda x: x[1],
    )

    return best_threshold, best_score


def main():
    paths = Paths(root=DATA_ROOT)
    bcfg = BlockingConfig()
    tcfg = TrainConfig()

    s1, s2, s3, gt = load_all_sources(paths, split="train")

    all_results = []

    # -------------------------------------------------------------
    # PREPARE RAW CANDIDATES + PRE-RANKER TRAINING DATA
    # -------------------------------------------------------------
    country_data = []
    pretrain_X = []
    pretrain_y = []

    print("=" * 90)
    print("PREPARING TWO-STAGE EXPERIMENT")
    print("=" * 90)

    for country in sorted(s1[COUNTRY].unique()):
        print()
        print(f"COUNTRY: {country}")

        s1_c = s1[
            s1[COUNTRY] == country
        ].reset_index(drop=True)

        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]

        gt_c = gt[
            gt[GT_S1].isin(s1_c[ENTITY_ID])
        ].copy()

        train_ids, val_ids = entity_level_split(
            s1_c[ENTITY_ID],
            tcfg.val_fraction,
            tcfg.random_seed,
        )

        ranked_path = Path(
            f".ranked_{country}.parquet"
        )

        if not ranked_path.exists():
            raise FileNotFoundError(
                f"Missing {ranked_path}. "
                "Run diag_rank_analysis.py for Sample B first."
            )

        raw_candidates = pd.read_parquet(
            ranked_path
        )

        print(
            f"Raw candidates: "
            f"{len(raw_candidates):,}"
        )

        s1n = build_normalized_frame(
            s1_c,
            bcfg.ngram_n,
        )

        s2n = build_normalized_frame(
            s2_c,
            bcfg.ngram_n,
        )

        s3n = build_normalized_frame(
            s3_c,
            bcfg.ngram_n,
        )

        other_norm = pd.concat(
            [s2n, s3n],
            ignore_index=True,
        )

        train_raw = raw_candidates[
            raw_candidates[ENTITY_ID].isin(train_ids)
        ].copy()

        val_raw = raw_candidates[
            raw_candidates[ENTITY_ID].isin(val_ids)
        ].copy()

        # Pre-ranker training data.
        sampled = sample_training_pairs(
            train_raw,
            negatives_per_positive=tcfg.negatives_per_positive,
            random_seed=tcfg.random_seed,
        )

        print(
            f"Pre-ranker training pairs: "
            f"{len(sampled):,} | "
            f"positives={int(sampled['label'].sum()):,}"
        )

        feats = compute_features(
            sampled,
            s1n,
            other_norm,
        )

        pretrain_X.append(
            feats[FEATURE_COLUMNS]
        )

        pretrain_y.append(
            sampled["label"].astype(np.uint8).to_numpy()
        )

        country_data.append(
            {
                "country": country,
                "s1n": s1n,
                "other_norm": other_norm,
                "gt": gt_c,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_raw": train_raw,
                "val_raw": val_raw,
            }
        )

    # -------------------------------------------------------------
    # TRAIN PRE-RANKER
    # -------------------------------------------------------------
    print()
    print("=" * 90)
    print("TRAINING PRE-RANKER")
    print("=" * 90)

    X_pre = pd.concat(
        pretrain_X,
        ignore_index=True,
    )

    y_pre = np.concatenate(pretrain_y)

    params = dict(tcfg.lgbm_params)
    params["n_estimators"] = 300

    pre_model = lgb.LGBMClassifier(**params)
    pre_model.fit(X_pre, y_pre)

    print(
        f"Pre-ranker rows={len(X_pre):,} | "
        f"positives={int(y_pre.sum()):,}"
    )

    # -------------------------------------------------------------
    # EVALUATE BOTH ARCHITECTURES
    # -------------------------------------------------------------
    print()
    print("=" * 90)
    print("END-TO-END F0.5 COMPARISON")
    print("=" * 90)

    for data in country_data:
        country = data["country"]
        s1n = data["s1n"]
        other_norm = data["other_norm"]
        gt_c = data["gt"]
        train_ids = data["train_ids"]
        val_ids = data["val_ids"]
        train_raw = data["train_raw"]
        val_raw = data["val_raw"]

        print()
        print(f"COUNTRY: {country}")

        # =========================================================
        # BASELINE: current n_keys -> top 60 -> final matcher
        # =========================================================
        baseline_train = (
            train_raw
            .sort_values(
                ["entity_id", "n_keys"],
                ascending=[True, False],
                kind="mergesort",
            )
            .groupby(
                "entity_id",
                group_keys=False,
            )
            .head(TOP_K)
            .reset_index(drop=True)
        )

        baseline_val = (
            val_raw
            .sort_values(
                ["entity_id", "n_keys"],
                ascending=[True, False],
                kind="mergesort",
            )
            .groupby(
                "entity_id",
                group_keys=False,
            )
            .head(TOP_K)
            .reset_index(drop=True)
        )

        baseline_model = fit_matcher(
            baseline_train,
            s1n,
            other_norm,
            tcfg,
            "BASELINE",
        )

        val_gt = gt_c[
            gt_c[GT_S1].isin(val_ids)
        ].copy()

        base_t, base_f = choose_threshold(
            baseline_model,
            baseline_val,
            val_gt,
            val_ids,
            s1n,
            other_norm,
        )

        print(
            f"  BASELINE final F0.5={base_f:.4f} "
            f"threshold={base_t:.2f}"
        )

        # =========================================================
        # TWO-STAGE:
        # raw blocking -> learned pre-ranker -> top 60 -> matcher
        # =========================================================

        def rerank(raw):
            # KEEP ALL ORIGINAL COLUMNS, especially `label`.
            # The label is not used for scoring/ranking; it is needed later
            # by sample_training_pairs() when training the final matcher.
            feats = compute_features(
                raw,
                s1n,
                other_norm,
            )

            scores = pre_model.predict_proba(
                feats[FEATURE_COLUMNS]
            )[:, 1]

            ranked = raw.copy()
            ranked["rank_score"] = scores

            ranked = ranked.sort_values(
                ["entity_id", "rank_score"],
                ascending=[True, False],
                kind="mergesort",
            )

            return (
                ranked
                .groupby(
                    "entity_id",
                    group_keys=False,
                )
                .head(TOP_K)
                .reset_index(drop=True)
            )

        print("  Reranking training candidates...")
        reranked_train = rerank(train_raw)

        print("  Reranking validation candidates...")
        reranked_val = rerank(val_raw)

        reranked_model = fit_matcher(
            reranked_train,
            s1n,
            other_norm,
            tcfg,
            "TWO-STAGE",
        )

        two_t, two_f = choose_threshold(
            reranked_model,
            reranked_val,
            val_gt,
            val_ids,
            s1n,
            other_norm,
        )

        print(
            f"  TWO-STAGE final F0.5={two_f:.4f} "
            f"threshold={two_t:.2f}"
        )

        all_results.append(
            {
                "country": country,
                "baseline_f05": base_f,
                "baseline_threshold": base_t,
                "two_stage_f05": two_f,
                "two_stage_threshold": two_t,
            }
        )

    result = pd.DataFrame(all_results)

    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(result.to_string(index=False))

    result.to_csv(
        "two_stage_f05_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

