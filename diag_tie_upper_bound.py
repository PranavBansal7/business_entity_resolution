from pathlib import Path
import pandas as pd

CAPS = [60, 100, 150, 200, 300]


def analyze(path):
    country = path.stem.replace(".ranked_", "").replace(".parquet", "")
    df = pd.read_parquet(path)

    print()
    print("=" * 80)
    print(f"{country}")
    print("=" * 80)

    total_true = int(df["label"].sum())

    print(f"Candidates: {len(df):,}")
    print(f"True matches surfaced: {total_true:,}")

    print()
    print("Candidate counts by n_keys:")
    print(
        df["n_keys"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("True matches by n_keys:")
    positives = df[df["label"] == 1]

    print(
        positives["n_keys"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # -------------------------------------------------------------
    # Current ranking:
    #   n_keys DESC
    #
    # Best possible tie-break:
    #   n_keys DESC
    #   label DESC
    #
    # label is only used here to establish a theoretical UPPER BOUND.
    # It is NOT available at inference time.
    # -------------------------------------------------------------

    best = df.sort_values(
        ["entity_id", "n_keys", "label"],
        ascending=[True, False, False],
        kind="mergesort",
    ).copy()

    print()
    print("THEORETICAL BEST RECALL IF n_keys TIES WERE PERFECTLY RESOLVED")

    for cap in CAPS:
        selected = (
            best.groupby("entity_id", group_keys=False)
            .head(cap)
        )

        found = int(selected["label"].sum())
        recall = found / total_true if total_true else 0.0

        print(
            f"  cap={cap:3d} | "
            f"true={found:,}/{total_true:,} | "
            f"recall={recall:.4f}"
        )

    # -------------------------------------------------------------
    # How many true matches are actually tied with other candidates?
    # -------------------------------------------------------------

    print()
    print("TRUE-MATCH POSITION WITHIN ITS n_keys TIE GROUP")

    tmp = df.copy()

    tmp["nkeys_group_size"] = (
        tmp.groupby(["entity_id", "n_keys"])["candidate_id"]
        .transform("size")
    )

    # This gives each row's position inside its entity+n_keys group.
    tmp["tie_position"] = (
        tmp.groupby(["entity_id", "n_keys"])
        .cumcount() + 1
    )

    pos = tmp[tmp["label"] == 1]

    for k in sorted(pos["n_keys"].unique()):
        p = pos[pos["n_keys"] == k]

        print(
            f"  n_keys={k}: "
            f"true={len(p):,} | "
            f"tie_size_median="
            f"{p['nkeys_group_size'].median():.1f} | "
            f"tie_position_median="
            f"{p['tie_position'].median():.1f} | "
            f"tie_position_90%="
            f"{p['tie_position'].quantile(.90):.1f}"
        )


for path in [
    Path(".ranked_India.parquet"),
    Path(".ranked_US.parquet"),
]:
    if path.exists():
        analyze(path)
    else:
        print(f"Missing: {path}")
