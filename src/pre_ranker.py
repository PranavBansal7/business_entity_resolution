"""
Lightweight learned pre-ranker for candidate selection.

Purpose:
    Raw blocking can surface a large candidate set. The existing pipeline
    currently sorts only by n_keys and then keeps the top K. This module
    provides a lightweight learned ranking stage BEFORE the expensive final
    matcher.

Design:
    - Uses only cheap/important features.
    - Trains on candidate pairs that already passed blocking.
    - Scores candidates before the final K cap.
    - Never uses ground-truth labels at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


PRE_RANK_FEATURES = [
    "n_keys",
    "matched_by_token",
    "matched_by_prefix",
    "matched_by_number",
    "matched_by_ngram",
    "name_exact_clean",
    "name_exact_core",
    "name_exact_ascii",
    "name_token_jaccard",
    "name_char_jaccard",
    "addr_number_any_shared",
]


@dataclass
class PreRankConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 30
    random_seed: int = 42


def _jaccard(a, b) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def compute_pre_rank_features(
    pairs: pd.DataFrame,
    s1_norm: pd.DataFrame,
    cand_norm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute only the inexpensive features needed by the pre-ranker.

    IMPORTANT:
        No label is used here.
    """
    s1_idx = s1_norm.set_index("entity_id")
    cand_idx = cand_norm.set_index("entity_id")

    left = pairs.join(
        s1_idx,
        on="entity_id",
        rsuffix="_s1_unused",
    )

    left = left.join(
        cand_idx,
        on="candidate_id",
        rsuffix="_cand",
    )

    # Explicit names after the two joins.
    left = left.rename(
        columns={
            "name_clean": "name_clean_s1",
            "name_core": "name_core_s1",
            "name_ascii": "name_ascii_s1",
            "name_tokens": "name_tokens_s1",
            "name_ngrams": "name_ngrams_s1",
            "addr_numbers": "addr_numbers_s1",
        }
    )

    records = []

    for row in left.itertuples(index=False):
        r = row._asdict()

        name_clean_s1 = r["name_clean_s1"]
        name_clean_c = r["name_clean_cand"]

        name_core_s1 = r["name_core_s1"]
        name_core_c = r["name_core_cand"]

        name_ascii_s1 = r["name_ascii_s1"]
        name_ascii_c = r["name_ascii_cand"]

        name_tok_s1 = r["name_tokens_s1"]
        name_tok_c = r["name_tokens_cand"]

        ngram_s1 = r["name_ngrams_s1"]
        ngram_c = r["name_ngrams_cand"]

        nums_s1 = r["addr_numbers_s1"]
        nums_c = r["addr_numbers_cand"]

        match_keys = r.get("match_keys") or ""

        records.append(
            {
                "n_keys": float(r.get("n_keys", 0) or 0),

                "matched_by_token": int(
                    "name_token" in match_keys
                ),
                "matched_by_prefix": int(
                    "name_prefix" in match_keys
                ),
                "matched_by_number": int(
                    "addr_number" in match_keys
                ),
                "matched_by_ngram": int(
                    "name_ngram" in match_keys
                ),

                "name_exact_clean": int(
                    bool(name_clean_s1)
                    and name_clean_s1 == name_clean_c
                ),

                "name_exact_core": int(
                    bool(name_core_s1)
                    and name_core_s1 == name_core_c
                ),

                "name_exact_ascii": int(
                    bool(name_ascii_s1)
                    and name_ascii_s1 == name_ascii_c
                ),

                "name_token_jaccard": _jaccard(
                    name_tok_s1,
                    name_tok_c,
                ),

                "name_char_jaccard": _jaccard(
                    ngram_s1,
                    ngram_c,
                ),

                "addr_number_any_shared": int(
                    bool(nums_s1 & nums_c)
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def train_pre_ranker(
    X: pd.DataFrame,
    y: np.ndarray,
    config: PreRankConfig | None = None,
):
    """
    Train a binary LightGBM model whose score is used ONLY for ranking.
    """
    config = config or PreRankConfig()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        random_state=config.random_seed,
        n_jobs=-1,
    )

    model.fit(
        X[PRE_RANK_FEATURES],
        y,
    )

    return model


def score_pre_ranker(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    return model.predict_proba(
        X[PRE_RANK_FEATURES]
    )[:, 1]


def rerank_and_cap(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    max_candidates_per_entity: int,
) -> pd.DataFrame:
    """
    Keep the best K candidates for each S1 entity.

    All original columns are retained so downstream code can still access
    label / n_keys / match_keys.
    """
    ranked = pairs.copy()
    ranked["_pre_rank_score"] = np.asarray(scores)

    ranked = ranked.sort_values(
        ["entity_id", "_pre_rank_score", "n_keys"],
        ascending=[True, False, False],
        kind="mergesort",
    )

    ranked = (
        ranked.groupby(
            "entity_id",
            group_keys=False,
        )
        .head(max_candidates_per_entity)
        .reset_index(drop=True)
    )

    return ranked


def save_pre_ranker(model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_pre_ranker(path: Path):
    return joblib.load(path)
