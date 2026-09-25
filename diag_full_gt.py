from pathlib import Path
import pandas as pd

ROOT = Path(r"..\dataset\train")

gt_path = ROOT / "train_ground_truth.tsv"
s2_path = ROOT / "train_source2.tsv"
s3_path = ROOT / "train_source3.tsv"

# Read a manageable sample of GT rows.
gt = pd.read_csv(
    gt_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=2000,
)

target_ids = set()

for value in gt["matched_entity_ids"]:
    if value:
        target_ids.update(x.strip() for x in value.split(",") if x.strip())

print(f"GT rows sampled: {len(gt):,}")
print(f"Distinct target IDs sampled: {len(target_ids):,}")

remaining_s2 = {x for x in target_ids if x.startswith("S2-")}
remaining_s3 = {x for x in target_ids if x.startswith("S3-")}

print(f"Need to find in S2: {len(remaining_s2):,}")
print(f"Need to find in S3: {len(remaining_s3):,}")

def scan_ids(path, remaining, chunk_size=500_000):
    found = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype={"entity_id": str},
        usecols=["entity_id"],
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        ids = set(chunk["entity_id"])
        found.update(remaining.intersection(ids))

        if found == remaining:
            break

    return found

found_s2 = scan_ids(s2_path, remaining_s2)
found_s3 = scan_ids(s3_path, remaining_s3)

print()
print("RESULT")
print(f"S2 found: {len(found_s2):,} / {len(remaining_s2):,}")
print(f"S3 found: {len(found_s3):,} / {len(remaining_s3):,}")
print(f"Missing S2: {len(remaining_s2 - found_s2):,}")
print(f"Missing S3: {len(remaining_s3 - found_s3):,}")
