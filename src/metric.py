"""
The exact scoring metric from the challenge brief:

    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R),  beta = 0.5

computed PER Source-1 entity, then macro-averaged over every S1 entity in
the evaluation set (singletons included: an entity with no true matches
scores 1.0 for an empty prediction, 0.0 for any non-empty prediction).

This module is deliberately dependency-light (just Python sets) so it can
be trusted as the ground truth for "did I implement the metric right" —
see tests/test_metric.py, which reproduces the brief's own worked example
number-for-number.
"""
from collections import defaultdict

import numpy as np
import pandas as pd


def f_beta_for_entity(true_ids: set, pred_ids: set, beta: float = 0.5) -> float:
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    denom = (beta ** 2) * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta ** 2) * precision * recall / denom


def macro_f_beta(gt: dict, pred: dict, all_s1_ids, beta: float = 0.5) -> float:
    """gt / pred: {source1_entity_id: set(matched_ids)}. Every id in
    all_s1_ids is scored, defaulting to an empty set on either side if
    missing — mirrors what actually happens on the real leaderboard, where
    a missing row would be scored as "no prediction", not skipped."""
    scores = [
        f_beta_for_entity(gt.get(sid, set()), pred.get(sid, set()), beta)
        for sid in all_s1_ids
    ]
    return float(np.mean(scores)) if scores else 0.0


def ids_string_to_set(s) -> set:
    if not isinstance(s, str) or not s:
        return set()
    return {x for x in s.split(",") if x}


def load_gt_dict(gt_df: pd.DataFrame, s1_col: str = "source1_entity_id",
                  match_col: str = "matched_entity_ids") -> dict:
    return {row[s1_col]: ids_string_to_set(row[match_col])
            for row in gt_df.to_dict("records")}


def predictions_from_scored_pairs(scored: pd.DataFrame, threshold: float,
                                   entity_col: str = "entity_id",
                                   cand_col: str = "candidate_id",
                                   score_col: str = "score") -> dict:
    """Turn a (entity_id, candidate_id, score) frame into
    {entity_id: set(candidate_ids above threshold)}."""
    kept = scored[scored[score_col] >= threshold]
    out = defaultdict(set)
    for eid, cid in zip(kept[entity_col], kept[cand_col]):
        out[eid].add(cid)
    return dict(out)


def find_best_threshold(scored: pd.DataFrame, gt: dict, all_s1_ids,
                         thresholds=None, beta: float = 0.5,
                         entity_col: str = "entity_id",
                         cand_col: str = "candidate_id",
                         score_col: str = "score") -> tuple:
    """Grid-search the score threshold that maximizes macro F_beta on a
    validation split. Returns (best_threshold, best_score, full_curve_df).
    Precision is weighted 2x over recall in F_0.5, so don't assume 0.5 is
    anywhere near the optimum — in practice this tends to land high
    (often 0.6-0.85) because false merges are so heavily penalized."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = predictions_from_scored_pairs(
            scored, t, entity_col, cand_col, score_col)
        score = macro_f_beta(gt, pred, all_s1_ids, beta)
        rows.append((t, score))
    curve = pd.DataFrame(rows, columns=["threshold", "macro_f_beta"])
    best_row = curve.loc[curve["macro_f_beta"].idxmax()]
    return float(best_row["threshold"]), float(best_row["macro_f_beta"]), curve
