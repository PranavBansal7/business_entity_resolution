"""
Central configuration for the business entity resolution pipeline.
Edit paths here to point at your local `student_resource/dataset` folder.
Nothing else in the pipeline should hardcode a path, a country name, or a
column name â€” it should all flow from here.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    # ---- EDIT THESE to match your local checkout ----
    root: Path = Path("./dataset")
    train_source1: Path = None
    train_source2: Path = None
    train_source3: Path = None
    train_ground_truth: Path = None
    test_source1: Path = None
    test_source2: Path = None
    test_source3: Path = None

    output_dir: Path = Path("./output")
    work_dir: Path = Path("./work")  # cached parquet / intermediate artifacts

    def __post_init__(self):
        self.train_source1 = self.root / "train" / "train_source1.tsv"
        self.train_source2 = self.root / "train" / "train_source2.tsv"
        self.train_source3 = self.root / "train" / "train_source3.tsv"
        self.train_ground_truth = self.root / "train" / "train_ground_truth.tsv"
        self.test_source1 = self.root / "test" / "test_source1.tsv"
        self.test_source2 = self.root / "test" / "test_source2.tsv"
        self.test_source3 = self.root / "test" / "test_source3.tsv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)


# Columns exactly as specified in the challenge brief. Do not change unless
# the brief changes â€” everything downstream assumes these names.
ENTITY_ID = "entity_id"
BUSINESS_NAME = "business_name"
BUSINESS_ADDRESS = "business_address"
COUNTRY = "country"

GT_S1 = "source1_entity_id"
GT_MATCHES = "matched_entity_ids"


@dataclass
class BlockingConfig:
    # Max candidates kept per S1 entity after blocking, BEFORE the model
    # scores them. This is a recall/compute trade-off: higher = safer for
    # recall, slower/more memory for feature extraction + inference.
    max_candidates_per_entity: int = 150

    # Minimum token length to use as a blocking key (drop single letters).
    min_token_len: int = 3

    # Which blocking keys to use (union of all â€” recall-oriented).
    use_name_token_blocking: bool = True
    use_name_prefix_blocking: bool = True
    use_address_number_blocking: bool = True
    use_char_ngram_blocking: bool = True
    ngram_n: int = 4

    # Absolute (not percentage!) cap on how many records a single blocking-
    # key value may match before it's dropped as "too common to be useful".
    # Must be an absolute number, not a fraction of the dataset â€” at 5M+
    # rows a percentage cap would still let a common token through with a
    # six-figure posting list and blow up the join.
    max_postings_per_key: int = 400
    max_postings_per_key_numbers: int = 100  # numbers are noisier in bulk

    # Optional embedding-based ANN augmentation (ch)eaper on recall for
    # cross-script / unseen-language records; heavier dependency, off by
    # default. See src/embedding_blocking.py.
    use_embedding_blocking: bool = False
    embedding_top_k: int = 20


@dataclass
class TrainConfig:
    negatives_per_positive: int = 6          # hard-negative sampling ratio
    val_fraction: float = 0.15               # held out at the S1-entity level
    random_seed: int = 42
    lgbm_params: dict = field(default_factory=lambda: dict(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=30,
        n_estimators=2000,
        n_jobs=-1,
    ))
    early_stopping_rounds: int = 50
