"""
End-to-end training on the REAL local dataset. Run from
code/business_entity_resolution/ after editing config.Paths.root (or pass
--data-root).

    python src/train_model.py --data-root /path/to/student_resource/dataset

What this does, per country found in train_source1 (country is read from
the data, never hardcoded — a new country just works):
  1. normalize source1/2/3 subset to that country
  2. block source1 against source2 and against source3
  3. label candidates against ground truth (-> hard negatives + positives)
  4. accumulate a blocking-recall report (the ceiling on final recall)
Then, over the whole (all-country) candidate pool:
  5. split S1 entities into train/val (grouped, no leakage)
  6. sample a training set (all positives + hard-negative sample)
  7. train LightGBM with early stopping
  8. score the held-out val candidates, grid-search the F_0.5-optimal
     threshold
  9. save the model + threshold + feature list to work_dir/

Memory note: this holds one country's candidate/feature frames at a time
rather than all three countries at once, which is the single biggest lever
if you're on a memory-constrained machine. If even one country is too much,
the natural next cut is to also chunk by S1-entity_id ranges within a
country — the blocking/features/labeling functions all operate on whatever
frame you hand them, so that requires no code changes, just an outer loop.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import Paths, BlockingConfig, TrainConfig, COUNTRY, ENTITY_ID
from io_utils import load_all_sources, read_tsv_cached
from blocking import build_normalized_frame, generate_candidates, cap_candidates
from features import compute_features_parallel, FEATURE_COLUMNS
from build_pairs import (label_candidates, report_blocking_recall,
                          sample_training_pairs, entity_level_split)
from metric import load_gt_dict, find_best_threshold


def process_country(s1_c, s2_c, s3_c, gt_df, cfg: BlockingConfig, country: str):
    t0 = time.time()
    s1n = build_normalized_frame(s1_c, cfg.ngram_n)
    s2n = build_normalized_frame(s2_c, cfg.ngram_n)
    s3n = build_normalized_frame(s3_c, cfg.ngram_n)

    cand2 = generate_candidates(s1n, s2n, cfg)
    cand3 = generate_candidates(s1n, s3n, cfg)
    cand = pd.concat([cand2, cand3], ignore_index=True)
    cand = cap_candidates(cand, cfg.max_candidates_per_entity)

    labeled = label_candidates(cand, gt_df)
    recall = report_blocking_recall(labeled, gt_df)
    print(f"  [{country}] {len(s1_c):,} S1 rows -> {len(cand):,} candidates "
          f"in {time.time()-t0:.1f}s | pair recall={recall['pair_level_recall']:.3f} "
          f"entity-full-recall={recall['entity_level_full_recall']:.3f}")

    other_norm = pd.concat([s2n, s3n], ignore_index=True)
    return s1n, other_norm, labeled, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset")
    ap.add_argument("--work-dir", default="./work")
    args = ap.parse_args()

    paths = Paths(root=Path(args.data_root), work_dir=Path(args.work_dir))
    bcfg = BlockingConfig()
    tcfg = TrainConfig()

    s1, s2, s3, gt = load_all_sources(paths, split="train")
    countries = sorted(s1[COUNTRY].unique())
    print(f"countries in training data: {countries}")

    all_train_ids, all_val_ids = set(), set()
    train_feat_parts, val_feat_parts = [], []
    recall_reports = {}

    for country in countries:
        s1_c = s1[s1[COUNTRY] == country]
        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]
        gt_c = gt[gt["source1_entity_id"].isin(s1_c[ENTITY_ID])]

        s1n, other_norm, labeled, recall = process_country(s1_c, s2_c, s3_c, gt_c, bcfg, country)
        recall_reports[country] = recall

        train_ids, val_ids = entity_level_split(s1n[ENTITY_ID], tcfg.val_fraction, tcfg.random_seed)
        all_train_ids |= train_ids
        all_val_ids |= val_ids

        labeled_train = labeled[labeled[ENTITY_ID].isin(train_ids)]
        labeled_val = labeled[labeled[ENTITY_ID].isin(val_ids)]

        sampled_train = sample_training_pairs(
            labeled_train, tcfg.negatives_per_positive, random_seed=tcfg.random_seed)

        f_train = compute_features_parallel(sampled_train, s1n, other_norm)
        f_train["label"] = sampled_train["label"].values
        train_feat_parts.append(f_train)

        f_val = compute_features_parallel(labeled_val, s1n, other_norm)
        f_val["label"] = labeled_val["label"].values
        val_feat_parts.append(f_val)

    feats_train = pd.concat(train_feat_parts, ignore_index=True)
    feats_val = pd.concat(val_feat_parts, ignore_index=True)
    print(f"\ntotal training pairs: {len(feats_train):,} "
          f"(positives: {int(feats_train['label'].sum()):,})")
    print(f"total validation candidates: {len(feats_val):,} "
          f"(true positives available: {int(feats_val['label'].sum()):,})")

    # Small internal split off the TRAIN pairs for early stopping — the
    # `val` split above stays completely untouched until threshold search,
    # so the reported macro F_0.5 isn't inflated by any peeking.
    rng = np.random.default_rng(tcfg.random_seed)
    shuffle_idx = rng.permutation(len(feats_train))
    n_es = max(1, int(0.1 * len(feats_train)))
    es_idx, fit_idx = shuffle_idx[:n_es], shuffle_idx[n_es:]

    model = lgb.LGBMClassifier(**{**tcfg.lgbm_params, "verbosity": -1})
    model.fit(
        feats_train.iloc[fit_idx][FEATURE_COLUMNS], feats_train.iloc[fit_idx]["label"],
        eval_set=[(feats_train.iloc[es_idx][FEATURE_COLUMNS], feats_train.iloc[es_idx]["label"])],
        callbacks=[lgb.early_stopping(tcfg.early_stopping_rounds, verbose=False)],
    )
    print(f"trained with {model.best_iteration_} boosting rounds (early stopping)")

    val_scores = model.predict_proba(feats_val[FEATURE_COLUMNS])[:, 1]
    scored = pd.concat([f[["entity_id", "candidate_id"]] for f in
                         [feats_val]], ignore_index=True)
    scored["score"] = val_scores

    gt_dict = load_gt_dict(gt)
    best_t, best_f, curve = find_best_threshold(scored, gt_dict, list(all_val_ids))
    print(f"\nbest threshold: {best_t:.2f}   held-out macro F_0.5: {best_f:.4f}")
    print(curve.to_string(index=False))

    importances = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1])
    print("\nfeature importances:")
    for name, imp in importances:
        print(f"  {name}: {imp}")

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths.work_dir / "model.joblib")
    with open(paths.work_dir / "run_metadata.json", "w") as f:
        json.dump(dict(
            threshold=best_t, held_out_macro_f_beta=best_f,
            feature_columns=FEATURE_COLUMNS, blocking_config=vars(bcfg),
            recall_reports=recall_reports,
        ), f, indent=2, default=str)
    print(f"\nsaved model + metadata to {paths.work_dir}/")


if __name__ == "__main__":
    main()
