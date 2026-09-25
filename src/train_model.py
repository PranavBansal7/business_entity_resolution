"""
Memory-safe, chunked end-to-end training on the real local dataset.

Run from the project root:

    python src/train_model.py --data-root ./dataset --work-dir ./work_chunked

For the first real-data test, use the prepared 5K sample dataset:

    python src/train_model.py \
        --data-root ./tests/chunked_sample \
        --work-dir ./work_chunked_sample \
        --chunk-size 25000 \
        --clean-artifacts

Methodology preserved from the original train_model.py:
  1. country-wise normalization
  2. S1 -> S2 and S1 -> S3 blocking with the existing blocking module
  3. concatenate S2/S3 candidates
  4. cap candidates AFTER combining, using the configured cap
  5. ground-truth labeling
  6. blocking-recall accounting
  7. one entity-level train/validation split per country
  8. all positives + existing weighted negative sampling
  9. existing compute_features() (NOT multiprocessing)
 10. disk-backed feature chunks
 11. LightGBM configuration unchanged
 12. chunkwise validation prediction
 13. exact macro F_0.5 threshold evaluation using metric.f_beta_for_entity()

Important memory design:
  - only one S1 chunk is normalized/blocked/featured at a time
  - S2/S3 normalized frames are reused within a country
  - training/validation feature frames are never accumulated in RAM
  - validation scores are never concatenated in RAM
  - training features are assembled into float32 disk-backed memmaps
  - validation threshold search is streamed from score chunks
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import Paths, BlockingConfig, TrainConfig, COUNTRY, ENTITY_ID
from io_utils import load_all_sources
from blocking import build_normalized_frame, generate_candidates, cap_candidates
from features import compute_features, FEATURE_COLUMNS
from build_pairs import (
    label_candidates,
    report_blocking_recall,
    sample_training_pairs,
    entity_level_split,
)
from metric import load_gt_dict, f_beta_for_entity


DEFAULT_CHUNK_SIZE = 25_000
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _country_dirname(country) -> str:
    text = str(country)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return safe or "UNKNOWN"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except Exception:
        return False


PARQUET_AVAILABLE = _parquet_available()


def _write_frame(df: pd.DataFrame, directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    if PARQUET_AVAILABLE:
        path = directory / f"{stem}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = directory / f"{stem}.pkl.gz"
        df.to_pickle(path, compression="gzip")

    return path


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".pkl.gz"):
        return pd.read_pickle(path, compression="gzip")
    raise ValueError(f"Unsupported feature file: {path}")


def _feature_files(directory: Path):
    files = list(directory.rglob("*.parquet"))
    files += list(directory.rglob("*.pkl.gz"))
    return sorted(files)


def _clear_if_requested(paths, clean: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not clean:
        names = ", ".join(str(p) for p in existing)
        raise RuntimeError(
            "Chunk artifacts already exist. This run refuses to mix stale and "
            f"new chunks. Re-run with --clean-artifacts.\nExisting: {names}"
        )

    if clean:
        for p in existing:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


def _new_recall_accumulator() -> dict:
    return {
        "total_true_matches": 0,
        "matches_found_in_candidates": 0,
        "entities_with_matches": 0,
        "entities_fully_recalled": 0,
    }


def _accumulate_recall(acc: dict, report: dict) -> None:
    for key in acc:
        acc[key] += int(report.get(key, 0))


def _finalize_recall(acc: dict) -> dict:
    total_true = acc["total_true_matches"]
    entities_with_matches = acc["entities_with_matches"]

    pair_recall = (
        acc["matches_found_in_candidates"] / total_true
        if total_true
        else 0.0
    )
    entity_full_recall = (
        acc["entities_fully_recalled"] / entities_with_matches
        if entities_with_matches
        else 0.0
    )

    return {
        **acc,
        "pair_level_recall": float(pair_recall),
        "entity_level_full_recall": float(entity_full_recall),
    }


def _prepare_feature_subset(
    features: pd.DataFrame,
    include_ids: bool,
    include_label: bool,
) -> pd.DataFrame:
    required = list(FEATURE_COLUMNS)
    if include_ids:
        required = ["entity_id", "candidate_id"] + required
    if include_label:
        required = required + ["label"]

    missing = [col for col in required if col not in features.columns]
    if missing:
        raise KeyError(
            "compute_features() did not return expected columns: "
            + ", ".join(missing)
        )

    return features.loc[:, required].copy()


def _process_country(
    s1_c: pd.DataFrame,
    s2_c: pd.DataFrame,
    s3_c: pd.DataFrame,
    gt_c: pd.DataFrame,
    bcfg: BlockingConfig,
    tcfg: TrainConfig,
    country: str,
    train_root: Path,
    val_root: Path,
    chunk_size: int,
    chunk_offset: int,
):
    """
    Process one country in fixed global S1 chunks.

    The train/validation split is computed ONCE for the entire country before
    chunking. This is the critical leakage-prevention invariant.
    """
    country_name = _country_dirname(country)
    train_country_root = train_root / country_name
    val_country_root = val_root / country_name
    train_country_root.mkdir(parents=True, exist_ok=True)
    val_country_root.mkdir(parents=True, exist_ok=True)

    # Global entity-level split for this country: DO NOT recompute per chunk.
    all_country_ids = s1_c[ENTITY_ID]
    train_ids, val_ids = entity_level_split(
        all_country_ids,
        tcfg.val_fraction,
        tcfg.random_seed,
    )

    print(
        f"\n[{country}] S1={len(s1_c):,}, S2={len(s2_c):,}, S3={len(s3_c):,} | "
        f"global train IDs={len(train_ids):,}, val IDs={len(val_ids):,}"
    )

    # S2/S3 are normalized once per country and reused by every S1 chunk.
    s2n = build_normalized_frame(s2_c, bcfg.ngram_n)
    s3n = build_normalized_frame(s3_c, bcfg.ngram_n)
    other_norm = pd.concat([s2n, s3n], ignore_index=True)

    recall_acc = _new_recall_accumulator()
    train_rows = 0
    val_rows = 0
    chunk_id = chunk_offset

    # Preserve the input ordering; only membership in the fixed split decides
    # train vs validation. Chunking must not affect the split.
    for start in range(0, len(s1_c), chunk_size):
        stop = min(start + chunk_size, len(s1_c))
        t0 = time.time()

        s1_chunk = s1_c.iloc[start:stop].copy()
        s1n = build_normalized_frame(s1_chunk, bcfg.ngram_n)

        cand2 = generate_candidates(s1n, s2n, bcfg)
        cand3 = generate_candidates(s1n, s3n, bcfg)

        # IMPORTANT: cap AFTER combining S2/S3 candidates.
        cand = pd.concat([cand2, cand3], ignore_index=True)
        cand = cap_candidates(cand, bcfg.max_candidates_per_entity)

        labeled = label_candidates(cand, gt_c)
        recall = report_blocking_recall(labeled, gt_c)
        _accumulate_recall(recall_acc, recall)

        # Fixed global split assignment; never split this chunk independently.
        labeled_train = labeled[labeled[ENTITY_ID].isin(train_ids)]
        labeled_val = labeled[labeled[ENTITY_ID].isin(val_ids)]

        sampled_train = sample_training_pairs(
            labeled_train,
            tcfg.negatives_per_positive,
            random_seed=tcfg.random_seed,
        )

        f_train = compute_features(sampled_train, s1n, other_norm)
        f_train["label"] = sampled_train["label"].to_numpy()

        f_val = compute_features(labeled_val, s1n, other_norm)
        f_val["label"] = labeled_val["label"].to_numpy()

        train_to_save = _prepare_feature_subset(
            f_train,
            include_ids=False,
            include_label=True,
        )
        val_to_save = _prepare_feature_subset(
            f_val,
            include_ids=True,
            include_label=True,
        )

        _write_frame(
            train_to_save,
            train_country_root,
            f"chunk_{chunk_id:05d}",
        )
        _write_frame(
            val_to_save,
            val_country_root,
            f"chunk_{chunk_id:05d}",
        )

        train_rows += len(train_to_save)
        val_rows += len(val_to_save)

        elapsed = time.time() - t0
        pair_recall = recall.get("pair_level_recall", 0.0)
        full_recall = recall.get("entity_level_full_recall", 0.0)

        print(
            f"  chunk {chunk_id:05d} | "
            f"S1 {start:,}:{stop:,} ({len(s1_chunk):,}) | "
            f"candidates={len(cand):,} | "
            f"train={len(train_to_save):,} | val={len(val_to_save):,} | "
            f"chunk pair recall={pair_recall:.3f} | "
            f"chunk entity-full={full_recall:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Release all large chunk-local objects before the next chunk.
        del (
            s1_chunk,
            s1n,
            cand2,
            cand3,
            cand,
            labeled,
            labeled_train,
            labeled_val,
            sampled_train,
            f_train,
            f_val,
            train_to_save,
            val_to_save,
        )
        gc.collect()

        chunk_id += 1

    # Country-level aggregate is computed from the chunk counts, not averaged
    # across chunks (which would weight small and large chunks equally).
    country_recall = _finalize_recall(recall_acc)

    print(
        f"  [{country}] aggregate blocking recall | "
        f"pair={country_recall['pair_level_recall']:.4f} | "
        f"entity-full={country_recall['entity_level_full_recall']:.4f} | "
        f"train rows={train_rows:,} | val rows={val_rows:,}"
    )

    del s2n, s3n, other_norm
    gc.collect()

    return {
        "recall": country_recall,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "next_chunk_id": chunk_id,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }


def _materialize_training_memmaps(
    train_files,
    work_dir: Path,
):
    if not train_files:
        raise RuntimeError("No training feature chunks were produced.")

    row_counts = []
    for path in train_files:
        df = _read_frame(path)
        row_counts.append(len(df))
        del df
        gc.collect()

    n_rows = int(sum(row_counts))
    n_features = len(FEATURE_COLUMNS)

    x_path = work_dir / "train_features.float32.memmap"
    y_path = work_dir / "train_labels.uint8.memmap"

    # Remove stale files from a previous run.
    for path in (x_path, y_path):
        if path.exists():
            path.unlink()

    x_mm = np.memmap(
        x_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_rows, n_features),
    )
    y_mm = np.memmap(
        y_path,
        dtype=np.uint8,
        mode="w+",
        shape=(n_rows,),
    )

    offset = 0
    for i, path in enumerate(train_files):
        df = _read_frame(path)

        end = offset + len(df)
        x_mm[offset:end] = df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        y_mm[offset:end] = df["label"].to_numpy(
            dtype=np.uint8,
            copy=False,
        )

        offset = end
        print(
            f"  loaded train chunk {i + 1}/{len(train_files)} "
            f"into memmap ({offset:,}/{n_rows:,} rows)"
        )

        del df
        gc.collect()

    x_mm.flush()
    y_mm.flush()

    return x_mm, y_mm, n_rows


def _make_shuffled_memmaps(
    x_mm,
    y_mm,
    n_rows: int,
    work_dir: Path,
    random_seed: int,
    block_rows: int = 100_000,
):
    """
    Reproduce the original random-permutation internal early-stopping split
    while keeping the large feature matrix disk-backed.

    The shuffled copy makes the 10% early-stopping set contiguous, so LightGBM
    can receive memmap slices instead of a giant boolean-indexed RAM copy.
    """
    rng = np.random.default_rng(random_seed)
    permutation = rng.permutation(n_rows)

    x_shuf_path = work_dir / "train_features_shuffled.float32.memmap"
    y_shuf_path = work_dir / "train_labels_shuffled.uint8.memmap"

    for path in (x_shuf_path, y_shuf_path):
        if path.exists():
            path.unlink()

    x_shuf = np.memmap(
        x_shuf_path,
        dtype=np.float32,
        mode="w+",
        shape=x_mm.shape,
    )
    y_shuf = np.memmap(
        y_shuf_path,
        dtype=np.uint8,
        mode="w+",
        shape=y_mm.shape,
    )

    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        idx = permutation[start:stop]

        x_shuf[start:stop] = x_mm[idx]
        y_shuf[start:stop] = y_mm[idx]

    del permutation
    gc.collect()

    x_shuf.flush()
    y_shuf.flush()

    n_es = max(1, int(0.1 * n_rows))
    return x_shuf, y_shuf, n_es, x_shuf_path, y_shuf_path


def _train_model(
    train_files,
    work_dir: Path,
    tcfg: TrainConfig,
):
    x_mm, y_mm, n_rows = _materialize_training_memmaps(
        train_files,
        work_dir,
    )

    x_shuf, y_shuf, n_es, x_shuf_path, y_shuf_path = _make_shuffled_memmaps(
        x_mm,
        y_mm,
        n_rows,
        work_dir,
        tcfg.random_seed,
    )

    model = lgb.LGBMClassifier(
        **{**tcfg.lgbm_params, "verbosity": -1}
    )

    print(
        f"\ntraining LightGBM on {n_rows:,} sampled training pairs "
        f"({int(y_mm.sum()):,} positives)"
    )

    model.fit(
        x_shuf[n_es:],
        y_shuf[n_es:],
        eval_set=[
            (x_shuf[:n_es], y_shuf[:n_es])
        ],
        callbacks=[
            lgb.early_stopping(
                tcfg.early_stopping_rounds,
                verbose=False,
            )
        ],
    )

    print(
        f"trained with {model.best_iteration_} boosting rounds "
        f"(early stopping)"
    )

    # Close memmaps before returning. File-backed artifacts remain on disk.
    del x_shuf, y_shuf, x_mm, y_mm
    gc.collect()

    return model, x_shuf_path, y_shuf_path


def _score_validation_chunks(
    model,
    val_files,
    score_root: Path,
):
    if not val_files:
        raise RuntimeError("No validation feature chunks were produced.")

    score_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for i, path in enumerate(val_files):
        df = _read_frame(path)

        x = df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        scores = model.predict_proba(x)[:, 1]

        scored = df.loc[:, ["entity_id", "candidate_id"]].copy()
        scored["score"] = scores

        # Match predictions_from_scored_pairs() semantics exactly: candidate
        # IDs are sets per entity, so remove duplicate pair rows before scoring.
        scored = scored.drop_duplicates(
            subset=["entity_id", "candidate_id"],
            keep="last",
        ).reset_index(drop=True)

        out = _write_frame(
            scored,
            score_root,
            f"score_{i:05d}",
        )
        total += len(scored)

        print(
            f"  validation score chunk {i + 1}/{len(val_files)} | "
            f"rows={len(scored):,} -> {out.name}"
        )

        del df, x, scores, scored
        gc.collect()

    print(f"total validation scored pairs: {total:,}")


def _stream_find_best_threshold(
    score_files,
    gt_dict,
    all_val_ids,
    beta: float = 0.5,
    thresholds=None,
):
    """
    Exact metric evaluation without concatenating all validation pairs.

    For each validation entity, its candidates are sorted by model score once.
    The predicted set for each threshold is then read as the prefix whose score
    is >= threshold. f_beta_for_entity() from metric.py is used unchanged.

    Entities absent from all score files are explicitly scored as empty
    predictions, matching macro_f_beta().
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    thresholds = [float(t) for t in thresholds]
    score_sums = np.zeros(len(thresholds), dtype=np.float64)
    seen_entities = set()

    for file_idx, path in enumerate(score_files):
        df = _read_frame(path)

        if df.empty:
            del df
            continue

        df = df.drop_duplicates(
            subset=["entity_id", "candidate_id"],
            keep="last",
        )

        for eid, group in df.groupby("entity_id", sort=False):
            seen_entities.add(eid)

            true_ids = gt_dict.get(eid, set())

            scores = group["score"].to_numpy(dtype=np.float64, copy=False)
            cids = group["candidate_id"].to_numpy(copy=False)

            order = np.argsort(-scores, kind="mergesort")
            sorted_scores = scores[order]
            sorted_cids = cids[order]

            for j, threshold in enumerate(thresholds):
                k = int(np.searchsorted(
                    -sorted_scores,
                    -threshold,
                    side="left",
                ))
                pred_ids = (
                    set(sorted_cids[:k].tolist())
                    if k
                    else set()
                )
                score_sums[j] += f_beta_for_entity(
                    true_ids,
                    pred_ids,
                    beta=beta,
                )

        print(
            f"  threshold-search processed score chunk "
            f"{file_idx + 1}/{len(score_files)}"
        )

        del df
        gc.collect()

    # Include S1 validation entities with no candidate rows at all.
    unseen = set(all_val_ids) - seen_entities
    for eid in unseen:
        score_sums += f_beta_for_entity(
            gt_dict.get(eid, set()),
            set(),
            beta=beta,
        )

    denominator = len(all_val_ids)
    if denominator == 0:
        curve = pd.DataFrame(
            {
                "threshold": thresholds,
                "macro_f_beta": np.zeros(len(thresholds)),
            }
        )
        return 0.05, 0.0, curve

    macro_scores = score_sums / denominator

    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "macro_f_beta": macro_scores,
        }
    )
    best_idx = int(curve["macro_f_beta"].idxmax())

    return (
        float(curve.loc[best_idx, "threshold"]),
        float(curve.loc[best_idx, "macro_f_beta"]),
        curve,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset")
    ap.add_argument("--work-dir", default="./work_chunked")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of S1 entities per chunk (default: 25000).",
    )
    ap.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="Delete prior chunk/score artifacts before this run.",
    )
    args = ap.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    paths = Paths(
        root=Path(args.data_root),
        work_dir=Path(args.work_dir),
    )
    bcfg = BlockingConfig()
    tcfg = TrainConfig()

    train_root = paths.work_dir / "train_features"
    val_root = paths.work_dir / "val_features"
    score_root = paths.work_dir / "val_scores"

    _clear_if_requested(
        [train_root, val_root, score_root],
        clean=args.clean_artifacts,
    )

    paths.work_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"feature storage: "
        f"{'Parquet' if PARQUET_AVAILABLE else 'gzip-compressed pickle'}"
    )
    print(f"S1 chunk size: {args.chunk_size:,}")

    s1, s2, s3, gt = load_all_sources(paths, split="train")
    countries = sorted(s1[COUNTRY].unique())

    print(f"countries in training data: {countries}")

    recall_reports = {}
    overall_recall_acc = _new_recall_accumulator()

    all_train_ids = set()
    all_val_ids = set()

    chunk_offset = 0
    total_train_rows = 0
    total_val_rows = 0

    for country in countries:
        s1_c = s1[s1[COUNTRY] == country]
        s2_c = s2[s2[COUNTRY] == country]
        s3_c = s3[s3[COUNTRY] == country]
        gt_c = gt[gt["source1_entity_id"].isin(s1_c[ENTITY_ID])]

        result = _process_country(
            s1_c=s1_c,
            s2_c=s2_c,
            s3_c=s3_c,
            gt_c=gt_c,
            bcfg=bcfg,
            tcfg=tcfg,
            country=country,
            train_root=train_root,
            val_root=val_root,
            chunk_size=args.chunk_size,
            chunk_offset=chunk_offset,
        )

        chunk_offset = result["next_chunk_id"]
        total_train_rows += result["train_rows"]
        total_val_rows += result["val_rows"]

        recall_reports[str(country)] = result["recall"]
        _accumulate_recall(overall_recall_acc, result["recall"])

        all_train_ids |= set(result["train_ids"])
        all_val_ids |= set(result["val_ids"])

        del s1_c, s2_c, s3_c, gt_c
        gc.collect()

    # Release the full source tables before the model-training phase.
    del s1, s2, s3, gt
    gc.collect()

    overall_recall = _finalize_recall(overall_recall_acc)

    print("\n" + "=" * 72)
    print("AGGREGATE BLOCKING RECALL")
    print("=" * 72)
    print(
        f"pair-level recall:          "
        f"{overall_recall['pair_level_recall']:.4f}"
    )
    print(
        f"entity full recall:          "
        f"{overall_recall['entity_level_full_recall']:.4f}"
    )
    print(
        f"true matches:                "
        f"{overall_recall['total_true_matches']:,}"
    )
    print(
        f"matches found in candidates: "
        f"{overall_recall['matches_found_in_candidates']:,}"
    )
    print(
        f"entities with true matches:  "
        f"{overall_recall['entities_with_matches']:,}"
    )
    print(
        f"entities fully recalled:     "
        f"{overall_recall['entities_fully_recalled']:,}"
    )

    train_files = _feature_files(train_root)
    val_files = _feature_files(val_root)

    if not train_files or not val_files:
        raise RuntimeError(
            "Chunk feature generation produced no train or validation files."
        )

    print(
        f"\nfeature chunks: train={len(train_files)}, "
        f"val={len(val_files)}"
    )
    print(
        f"training rows: {total_train_rows:,} | "
        f"validation rows: {total_val_rows:,}"
    )

    model, shuffled_x_path, shuffled_y_path = _train_model(
        train_files,
        paths.work_dir,
        tcfg,
    )

    _score_validation_chunks(
        model,
        val_files,
        score_root,
    )

    score_files = _feature_files(score_root)

    # Reload GT only if the first phase was large enough that it was deleted.
    # The helper itself only needs the dictionary, and load_all_sources() is
    # intentionally not called again; read the GT directly from the train path.
    gt_path = paths.root / "train" / "train_ground_truth.tsv"
    gt_df = pd.read_csv(gt_path, sep="\t")
    gt_dict = load_gt_dict(gt_df)

    best_t, best_f, curve = _stream_find_best_threshold(
        score_files=score_files,
        gt_dict=gt_dict,
        all_val_ids=all_val_ids,
        beta=0.5,
        thresholds=THRESHOLDS,
    )

    print(f"\nbest threshold: {best_t:.2f}")
    print(f"held-out macro F_0.5: {best_f:.4f}")
    print(curve.to_string(index=False))

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: -x[1],
    )

    print("\nfeature importances:")
    for name, importance in importances:
        print(f"  {name}: {importance}")

    metadata = dict(
        threshold=best_t,
        held_out_macro_f_beta=best_f,
        feature_columns=FEATURE_COLUMNS,
        blocking_config=vars(bcfg),
        recall_reports=recall_reports,
        overall_blocking_recall=overall_recall,
        chunk_size=args.chunk_size,
        parquet_available=PARQUET_AVAILABLE,
        train_rows=total_train_rows,
        validation_rows=total_val_rows,
        train_entity_count=len(all_train_ids),
        validation_entity_count=len(all_val_ids),
        best_iteration=int(model.best_iteration_),
    )

    joblib.dump(
        model,
        paths.work_dir / "model.joblib",
    )

    with open(
        paths.work_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            default=str,
        )

    curve.to_csv(
        paths.work_dir / "threshold_curve.csv",
        index=False,
    )

    # Shuffled memmaps are only training intermediates. Keep the original
    # feature chunks and compact score chunks as the useful reproducibility
    # artifacts.
    for path in (shuffled_x_path, shuffled_y_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    print(f"\nsaved model + metadata to {paths.work_dir}/")


if __name__ == "__main__":
    main()
