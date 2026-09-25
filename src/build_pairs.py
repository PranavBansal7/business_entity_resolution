"""
Turn (normalized sources + ground truth) into a labeled training table.

Key design choice: positives AND negatives are both drawn from the
blocking output, not generated separately. A candidate pair is:
  label=1  if candidate_id is in the S1 entity's true match set
  label=0  otherwise (it was blocked — i.e. looks superficially similar
           enough to be a candidate — but isn't a true match)

This means the classifier is trained on exactly the distribution it will
see at inference time (things that already passed blocking), and the
negatives are automatically "hard negatives" — near-miss look-alikes,
not random unrelated pairs, which is exactly what a precision-weighted
metric like F_0.5 needs to learn to reject.

IMPORTANT CONSEQUENCE: a true match that blocking never surfaces as a
candidate cannot appear here as a positive example, and can never be
predicted at inference either. Blocking recall is a hard ceiling on
everything downstream — see the recall report this module prints.
"""
import numpy as np
import pandas as pd

from config import GT_S1, GT_MATCHES
from metric import ids_string_to_set


def label_candidates(candidates: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `label` column (1/0) to the candidate-pairs frame."""
    gt_map = {row[GT_S1]: ids_string_to_set(row[GT_MATCHES])
              for row in gt_df.to_dict("records")}
    true_sets = candidates["entity_id"].map(gt_map)
    labels = [
        int(cid in ts) if isinstance(ts, set) else 0
        for cid, ts in zip(candidates["candidate_id"], true_sets)
    ]
    out = candidates.copy()
    out["label"] = labels
    return out


def report_blocking_recall(labeled_candidates: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """How much of the ground truth did blocking actually surface? This is
    the recall CEILING — the matching model can never exceed it, no matter
    how good the classifier is."""
    gt_map = {row[GT_S1]: ids_string_to_set(row[GT_MATCHES])
              for row in gt_df.to_dict("records")}
    total_true = sum(len(v) for v in gt_map.values())
    found_true = int(labeled_candidates["label"].sum())
    n_entities_with_matches = sum(1 for v in gt_map.values() if v)
    entities_fully_recalled = 0
    by_entity = labeled_candidates.groupby("entity_id")["label"].sum()
    for sid, true_set in gt_map.items():
        if not true_set:
            continue
        if by_entity.get(sid, 0) >= len(true_set):
            entities_fully_recalled += 1
    return dict(
        total_true_matches=total_true,
        matches_found_in_candidates=found_true,
        pair_level_recall=found_true / total_true if total_true else float("nan"),
        entities_with_matches=n_entities_with_matches,
        entities_fully_recalled=entities_fully_recalled,
        entity_level_full_recall=(entities_fully_recalled / n_entities_with_matches
                                   if n_entities_with_matches else float("nan")),
    )


def sample_training_pairs(labeled_candidates: pd.DataFrame,
                           negatives_per_positive: int = 6,
                           min_negatives_for_singleton: int = 6,
                           random_seed: int = 42) -> pd.DataFrame:
    """Per S1 entity: keep every positive, then keep a sample of negatives
    sized off the positive count (or `min_negatives_for_singleton` when
    there are none), weighted toward higher n_keys — the near-miss
    candidates are the ones the model most needs to see."""
    rng = np.random.default_rng(random_seed)
    keep_parts = []
    for eid, grp in labeled_candidates.groupby("entity_id"):
        pos = grp[grp["label"] == 1]
        neg = grp[grp["label"] == 0]
        keep_parts.append(pos)
        n_neg_keep = max(len(pos), 1) * negatives_per_positive
        n_neg_keep = max(n_neg_keep, min_negatives_for_singleton if len(pos) == 0 else 0)
        n_neg_keep = min(n_neg_keep, len(neg))
        if n_neg_keep > 0:
            w = neg["n_keys"].astype(float) + 1.0
            w = w / w.sum()
            idx = rng.choice(neg.index.to_numpy(), size=n_neg_keep, replace=False,
                              p=w.to_numpy())
            keep_parts.append(neg.loc[idx])
    if not keep_parts:
        return labeled_candidates.iloc[0:0]
    return pd.concat(keep_parts, ignore_index=True)


def entity_level_split(s1_ids, val_fraction: float = 0.15, random_seed: int = 42):
    """Split at the S1-ENTITY level, not the pair level — a pair-level
    split would leak (two candidates for the same entity could land on
    opposite sides, letting the model implicitly see validation-entity
    context during training)."""
    ids = np.array(sorted(set(s1_ids)))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids
