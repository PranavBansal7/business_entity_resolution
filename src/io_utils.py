"""
Loading for the real dataset files (hundreds of MB each). Two things this
handles that matter at this scale:

1. Parquet caching. Re-parsing a 480MB TSV every time you re-run a script
   during development is wasteful — we cache a parquet copy next to it on
   first load and reuse it after that (parquet is both smaller on disk and
   much faster to re-read than TSV).
2. Reading as `dtype=str` with `keep_default_na=False` throughout. These
   are identifier/text columns — letting pandas guess types or convert
   things like a business named "NA" or "NULL" into a real NaN would
   silently corrupt data. Every read in this project goes through here.
"""
from pathlib import Path

import pandas as pd


def read_tsv_cached(path: Path, columns=None) -> pd.DataFrame:
    path = Path(path)
    cache_path = path.with_suffix(".parquet")
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache_path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                      usecols=columns)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:  # pragma: no cover - caching is best-effort
        print(f"  (parquet cache write skipped for {path.name}: {e})")
    return df


def load_all_sources(paths, split: str = "train"):
    """split: 'train' or 'test'. Returns (s1, s2, s3, gt_or_None)."""
    s1 = read_tsv_cached(getattr(paths, f"{split}_source1"))
    s2 = read_tsv_cached(getattr(paths, f"{split}_source2"))
    s3 = read_tsv_cached(getattr(paths, f"{split}_source3"))
    gt = None
    if split == "train":
        gt = read_tsv_cached(paths.train_ground_truth)
    print(f"[{split}] source1={len(s1):,}  source2={len(s2):,}  "
          f"source3={len(s3):,}" + (f"  ground_truth={len(gt):,}" if gt is not None else ""))
    return s1, s2, s3, gt


def write_tsv(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"wrote {path}  ({len(df):,} rows)")
