from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"..\dataset\train")
OUT = Path(r".\tests\real_aligned_sample\train")
OUT.mkdir(parents=True, exist_ok=True)

N_S1 = 5000
N_DISTRACTORS = 10000
SEED = 42

gt_path = ROOT / "train_ground_truth.tsv"
s1_path = ROOT / "train_source1.tsv"
s2_path = ROOT / "train_source2.tsv"
s3_path = ROOT / "train_source3.tsv"

print("1/5 Loading first S1 rows...")

s1 = pd.read_csv(
    s1_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=N_S1,
)

s1_ids = set(s1["entity_id"])

print(f"S1 rows selected: {len(s1):,}")

print("2/5 Finding matching GT rows by S1 entity_id...")

gt_parts = []

for chunk_no, chunk in enumerate(
    pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=200_000,
    )
):
    matched = chunk[chunk["source1_entity_id"].isin(s1_ids)]

    if not matched.empty:
        gt_parts.append(matched)

    print(
        f"  GT chunk {chunk_no + 1:02d} | "
        f"matching rows found={sum(len(x) for x in gt_parts):,}",
        end="\r",
    )

    # Because each S1 entity should have exactly one GT row,
    # we can stop once all selected S1 IDs are found.
    if sum(len(x) for x in gt_parts) >= len(s1_ids):
        break

print()

if not gt_parts:
    raise RuntimeError("No GT rows matched the selected S1 IDs.")

gt = pd.concat(gt_parts, ignore_index=True)

gt_ids = set(gt["source1_entity_id"])

missing_gt_s1 = s1_ids - gt_ids

if missing_gt_s1:
    raise RuntimeError(
        f"{len(missing_gt_s1):,} selected S1 IDs have no GT row. "
        f"Examples: {list(sorted(missing_gt_s1))[:10]}"
    )

print(f"GT rows selected: {len(gt):,}")

target_s2 = set()
target_s3 = set()

for value in gt["matched_entity_ids"]:
    if not value:
        continue

    for entity_id in value.split(","):
        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            target_s2.add(entity_id)
        elif entity_id.startswith("S3-"):
            target_s3.add(entity_id)

print(f"Required S2 matches: {len(target_s2):,}")
print(f"Required S3 matches: {len(target_s3):,}")


def collect_source(path, required_ids, n_distractors, seed):
    required = set(required_ids)
    found = {}
    distractors = []
    rng = np.random.default_rng(seed)

    print(f"  Scanning {path.name}...")

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=200_000,
        )
    ):
        # Preserve ALL required GT-linked rows.
        matches = chunk[chunk["entity_id"].isin(required)]

        for row in matches.to_dict("records"):
            found[row["entity_id"]] = row

        # Add deterministic distractors.
        if len(distractors) < n_distractors:
            remaining = chunk[~chunk["entity_id"].isin(required)]

            if not remaining.empty:
                take = min(
                    1000,
                    n_distractors - len(distractors),
                    len(remaining),
                )

                indices = rng.choice(
                    len(remaining),
                    size=take,
                    replace=False,
                )

                distractors.extend(
                    remaining.iloc[indices].to_dict("records")
                )

        print(
            f"    chunk {chunk_no + 1:02d} | "
            f"required found={len(found):,}/{len(required):,} | "
            f"distractors={len(distractors):,}",
            end="\r",
        )

        if (
            len(found) == len(required)
            and len(distractors) >= n_distractors
        ):
            break

    print()

    missing = required - set(found)

    if missing:
        raise RuntimeError(
            f"{len(missing):,} required GT IDs were not found in "
            f"{path.name}. Examples: {list(sorted(missing))[:10]}"
        )

    required_rows = pd.DataFrame(list(found.values()))
    distractor_rows = pd.DataFrame(
        distractors[:n_distractors]
    )

    result = pd.concat(
        [required_rows, distractor_rows],
        ignore_index=True,
    ).drop_duplicates(subset=["entity_id"])

    return result


print("3/5 Building aligned S2 sample...")

s2_sample = collect_source(
    s2_path,
    target_s2,
    N_DISTRACTORS,
    SEED,
)

print(f"S2 sample rows: {len(s2_sample):,}")

print("4/5 Building aligned S3 sample...")

s3_sample = collect_source(
    s3_path,
    target_s3,
    N_DISTRACTORS,
    SEED + 1,
)

print(f"S3 sample rows: {len(s3_sample):,}")

print("5/5 Writing and verifying sample...")

s1.to_csv(
    OUT / "train_source1.tsv",
    sep="\t",
    index=False,
)

s2_sample.to_csv(
    OUT / "train_source2.tsv",
    sep="\t",
    index=False,
)

s3_sample.to_csv(
    OUT / "train_source3.tsv",
    sep="\t",
    index=False,
)

gt.to_csv(
    OUT / "train_ground_truth.tsv",
    sep="\t",
    index=False,
)

s2_ids = set(s2_sample["entity_id"])
s3_ids = set(s3_sample["entity_id"])

missing_s2 = target_s2 - s2_ids
missing_s3 = target_s3 - s3_ids

print()
print("=" * 70)
print("ALIGNED SAMPLE")
print("=" * 70)
print(f"S1 rows: {len(s1):,}")
print(f"S2 rows: {len(s2_sample):,}")
print(f"S3 rows: {len(s3_sample):,}")
print(f"GT rows: {len(gt):,}")
print(f"GT S2 targets: {len(target_s2):,}")
print(f"GT S3 targets: {len(target_s3):,}")
print(f"Missing S2 targets: {len(missing_s2):,}")
print(f"Missing S3 targets: {len(missing_s3):,}")
print(f"Output: {OUT.resolve()}")

assert len(gt) == len(s1)
assert not missing_s2
assert not missing_s3

print()
print("SUCCESS: sample is ground-truth consistent.")
