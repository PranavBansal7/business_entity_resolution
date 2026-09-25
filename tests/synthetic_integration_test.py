"""
NOT a performance benchmark. This proves the pipeline is WIRED CORRECTLY —
normalize -> block -> feature -> label -> train -> threshold -> score --
by perturbing real Source-1 records with the exact noise transformations
observed in the actual Source-2/3 data during EDA (suffix reshuffling,
abbreviation, typos, missing address, diacritic changes, DBA-pipe trailers,
landmark insertion, word-order shuffling). True negatives are genuine,
unrelated real records from the source2/3 samples.

The resulting F_0.5 number is a sanity check that the code path works, NOT
an estimate of real leaderboard performance — real performance depends on
the actual noise distribution in Source 2/3, which can only be measured by
running this same code on the full local dataset.
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import BlockingConfig, TrainConfig
from blocking import build_normalized_frame, generate_candidates, cap_candidates
from features import compute_features, FEATURE_COLUMNS
from build_pairs import label_candidates, report_blocking_recall, sample_training_pairs, entity_level_split
from metric import load_gt_dict, macro_f_beta, find_best_threshold, predictions_from_scored_pairs

RNG = random.Random(7)

SUFFIXES = ["LLC", "Inc", "Corp", "Pvt Ltd", "Private Limited", "SARL", "SAS", "Ltd", "LLP"]
STREET_ABBREV = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"),
                  ("Boulevard", "Blvd"), ("Drive", "Dr")]


def typo(word: str) -> str:
    if len(word) < 4:
        return word
    i = RNG.randrange(1, len(word) - 1)
    kind = RNG.choice(["swap", "dup", "drop"])
    if kind == "swap":
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if kind == "dup":
        return word[:i] + word[i] + word[i:]
    return word[:i] + word[i + 1:]


def perturb_name(name: str) -> str:
    words = name.split()
    ops = RNG.sample(
        ["suffix", "amp", "typo", "case", "dba"],
        k=RNG.randint(1, 3),
    )
    if "suffix" in ops:
        suf = RNG.choice(SUFFIXES)
        words = ([suf] + words) if RNG.random() < 0.5 else (words + [suf])
    if "amp" in ops and "and" in [w.lower() for w in words]:
        words = [("&" if w.lower() == "and" else w) for w in words]
    if "typo" in ops and words:
        i = RNG.randrange(len(words))
        words[i] = typo(words[i])
    if "case" in ops:
        words = [w.upper() if RNG.random() < 0.5 else w.lower() for w in words]
    name = " ".join(words)
    if "dba" in ops:
        name = f"{name} | {RNG.choice(['www.', ''])}{words[0].lower() if words else 'biz'}.com"
    return name


def perturb_address(addr: str) -> str:
    if not addr or RNG.random() < 0.08:  # ~missing-address rate observed in EDA
        return ""
    text = addr
    for full, abbr in STREET_ABBREV:
        if RNG.random() < 0.6:
            text = text.replace(full, abbr).replace(full.lower(), abbr)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) > 1 and RNG.random() < 0.4:
        RNG.shuffle(parts)
    if RNG.random() < 0.15:
        parts.append(f"Near {RNG.choice(['SBI ATM', 'City Mall', 'Central Park', 'Metro Station'])}")
    if len(parts) > 2 and RNG.random() < 0.2:
        parts.pop(RNG.randrange(len(parts)))
    return ", ".join(parts)


def build_synthetic_dataset(s1_sample: pd.DataFrame, real_other: pd.DataFrame, n_entities: int = 400):
    picked = s1_sample.sample(n=min(n_entities, len(s1_sample)), random_state=7).reset_index(drop=True)
    s1_rows, other_rows, gt_rows = [], [], []
    singleton_rate = 0.054  # matches the ~5.4% observed in the real ground-truth sample

    for i, row in picked.iterrows():
        s1_id = row["entity_id"]
        s1_rows.append(row.to_dict())
        if RNG.random() < singleton_rate:
            gt_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
            continue
        n_matches = RNG.choice([1, 1, 2, 2, 2, 3, 3, 4])
        match_ids = []
        for j in range(n_matches):
            src_prefix = RNG.choice(["S2", "S3"])
            new_id = f"{src_prefix}-SYN{i:04d}{j}"
            other_rows.append({
                "entity_id": new_id,
                "business_name": perturb_name(row["business_name"]),
                "business_address": perturb_address(row["business_address"]),
                "country": row["country"],
            })
            match_ids.append(new_id)
        gt_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(match_ids)})

    synth_other = pd.DataFrame(other_rows)
    # Mix in real, unrelated records as genuine (non-synthetic) true negatives.
    real_negs = real_other.sample(n=min(3000, len(real_other)), random_state=11)
    other_df = pd.concat([synth_other, real_negs], ignore_index=True).drop_duplicates("entity_id")
    gt_df = pd.DataFrame(gt_rows)
    return picked, other_df, gt_df


def main():
    s1 = pd.read_csv("train_source1_sample.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv("train_source2_sample.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv("train_source3_sample.tsv", sep="\t", dtype=str, keep_default_na=False)
    real_other = pd.concat([s2, s3], ignore_index=True)

    s1_synth, other_synth, gt_synth = build_synthetic_dataset(s1, real_other, n_entities=500)
    print(f"synthetic S1 entities: {len(s1_synth)}, other-source records "
          f"(synthetic matches + real negatives): {len(other_synth)}")

    cfg = BlockingConfig()
    s1n = build_normalized_frame(s1_synth, cfg.ngram_n)
    othern = build_normalized_frame(other_synth, cfg.ngram_n)

    cand = generate_candidates(s1n, othern, cfg)
    cand = cap_candidates(cand, cfg.max_candidates_per_entity)
    print(f"candidate pairs: {len(cand)}")

    labeled = label_candidates(cand, gt_synth)
    recall_report = report_blocking_recall(labeled, gt_synth)
    print("\n--- blocking recall (ceiling on everything downstream) ---")
    for k, v in recall_report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    train_cfg = TrainConfig()
    train_ids, val_ids = entity_level_split(s1n["entity_id"], train_cfg.val_fraction, train_cfg.random_seed)

    labeled_train = labeled[labeled["entity_id"].isin(train_ids)]
    labeled_val = labeled[labeled["entity_id"].isin(val_ids)]

    sampled_train = sample_training_pairs(labeled_train, train_cfg.negatives_per_positive,
                                            random_seed=train_cfg.random_seed)
    print(f"\ntraining pairs after sampling: {len(sampled_train)} "
          f"(positives: {int(sampled_train['label'].sum())})")

    feats_train = compute_features(sampled_train, s1n, othern)
    feats_train["label"] = sampled_train["label"].values

    feats_val = compute_features(labeled_val, s1n, othern)
    feats_val["label"] = labeled_val["label"].values

    model = lgb.LGBMClassifier(**{**train_cfg.lgbm_params, "n_estimators": 300, "verbosity": -1})
    model.fit(feats_train[FEATURE_COLUMNS], feats_train["label"])

    val_scores = model.predict_proba(feats_val[FEATURE_COLUMNS])[:, 1]
    scored = labeled_val[["entity_id", "candidate_id"]].copy()
    scored["score"] = val_scores

    gt_dict = load_gt_dict(gt_synth)
    val_s1_ids = list(val_ids)
    best_t, best_f, curve = find_best_threshold(scored, gt_dict, val_s1_ids)

    print(f"\n--- validation (synthetic — plumbing check, NOT a real performance estimate) ---")
    print(f"best threshold: {best_t}  macro F_0.5: {best_f:.4f}")
    print(curve.to_string(index=False))

    importances = sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
                          key=lambda x: -x[1])
    print("\ntop 10 feature importances:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp}")

    return best_f


if __name__ == "__main__":
    score = main()
    assert score > 0.5, f"Sanity check failed — synthetic macro F0.5 too low ({score}), pipeline likely has a bug"
    print("\nPASS: pipeline runs end-to-end and clearly separates synthetic matches from real negatives.")
