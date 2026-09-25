from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"..\dataset\train")
OUT = Path(r".\tests\real_aligned_sample_b\train")
OUT.mkdir(parents=True, exist_ok=True)

N_S1 = 5000
N_DISTRACTORS = 10000
SEED = 12345

s1_path = ROOT / "train_source1.tsv"
gt_path = ROOT / "train_ground_truth.tsv"
s2_path = ROOT / "train_source2.tsv"
s3_path = ROOT / "train_source3.tsv"

print("1/6 Selecting random S1 IDs...")

rng = np.random.default_rng(SEED)

# Reservoir sample entity IDs without loading the full S1 table.
sample_ids = []
seen = 0

for chunk in pd.read_csv(
    s1_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["entity_id"],
    chunksize=200_000,
):
    for entity_id in chunk["entity_id"]:
        seen += 1

        if len(sample_ids) < N_S1:
            sample_ids.append(entity_id)
        else:
            j = rng.integers(0, seen)
            if j < N_S1:
                sample_ids[j] = entity_id

s1_ids = set(sample_ids)

print(f"Selected random S1 IDs: {len(s1_ids):,}")

print("2/6 Collecting those full S1 rows...")

s1_parts = []

for chunk in pd.read_csv(
    s1_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=200_000,
):
    part = chunk[chunk["entity_id"].isin(s1_ids)]

    if not part.empty:
        s1_parts.append(part)

s1 = pd.concat(s1_parts, ignore_index=True)

if len(s1) != N_S1:
    raise RuntimeError(
        f"Expected {N_S1} S1 rows but found {len(s1)}"
    )

print(f"S1 rows collected: {len(s1):,}")

print("3/6 Collecting matching GT rows...")

gt_parts = []

for chunk in pd.read_csv(
    gt_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=200_000,
):
    part = chunk[
        chunk["source1_entity_id"].isin(s1_ids)
    ]

    if not part.empty:
        gt_parts.append(part)

gt = pd.concat(gt_parts, ignore_index=True)

if len(gt) != N_S1:
    raise RuntimeError(
        f"Expected {N_S1} GT rows but found {len(gt)}"
    )

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

print(f"Required S2 targets: {len(target_s2):,}")
print(f"Required S3 targets: {len(target_s3):,}")


def collect_source(path, required_ids, n_distractors, seed):
    required_ids = set(required_ids)
    found = {}
    distractors = []
    rng = np.random.default_rng(seed)

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=200_000,
        )
    ):
        required = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for row in required.to_dict("records"):
            found[row["entity_id"]] = row

        if len(distractors) < n_distractors:
            remaining = chunk[
                ~chunk["entity_id"].isin(required_ids)
            ]

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
            f"  {path.name}: chunk {chunk_no + 1:02d} | "
            f"required={len(found):,}/{len(required_ids):,} | "
            f"distractors={len(distractors):,}",
            end="\r",
        )

        if (
            len(found) == len(required_ids)
            and len(distractors) >= n_distractors
        ):
            break

    print()

    missing = required_ids - set(found)

    if missing:
        raise RuntimeError(
            f"{len(missing):,} required IDs missing from {path.name}"
        )

    required_rows = pd.DataFrame(list(found.values()))
    distractor_rows = pd.DataFrame(
        distractors[:n_distractors]
    )

    return pd.concat(
        [required_rows, distractor_rows],
        ignore_index=True,
    ).drop_duplicates("entity_id")


print("4/6 Building S2 sample...")

s2 = collect_source(
    s2_path,
    target_s2,
    N_DISTRACTORS,
    SEED,
)

print(f"S2 rows: {len(s2):,}")

print("5/6 Building S3 sample...")

s3 = collect_source(
    s3_path,
    target_s3,
    N_DISTRACTORS,
    SEED + 1,
)

print(f"S3 rows: {len(s3):,}")

print("6/6 Writing and verifying...")

s1.to_csv(
    OUT / "train_source1.tsv",
    sep="\t",
    index=False,
)

s2.to_csv(
    OUT / "train_source2.tsv",
    sep="\t",
    index=False,
)

s3.to_csv(
    OUT / "train_source3.tsv",
    sep="\t",
    index=False,
)

gt.to_csv(
    OUT / "train_ground_truth.tsv",
    sep="\t",
    index=False,
)

assert len(s1) == len(gt)

s2_ids = set(s2["entity_id"])
s3_ids = set(s3["entity_id"])

assert target_s2 <= s2_ids
assert target_s3 <= s3_ids

print()
print("=" * 70)
print("RANDOM ALIGNED SAMPLE B")
print("=" * 70)
print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")
print(f"GT: {len(gt):,}")
print(f"Required S2: {len(target_s2):,}")
print(f"Required S3: {len(target_s3):,}")
print(f"Missing S2: {len(target_s2 - s2_ids):,}")
print(f"Missing S3: {len(target_s3 - s3_ids):,}")
print(f"Output: {OUT.resolve()}")
print()
print("SUCCESS")
