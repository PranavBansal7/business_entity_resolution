# Business Entity Resolution — Pipeline

Matches Source-1 (reference) business records against Source-2/3 records
across US, India, and France, for the ML Challenge 2026 "Business Entity
Resolution" challenge. Precision-weighted (F_0.5) macro-averaged scoring.

## How this works, in one paragraph

Blocking (`src/blocking.py`) finds *candidate* matches for every Source-1
entity using vectorized inverted-index joins on name tokens, name
prefixes, address numbers, and character n-grams — fast enough to avoid
the ~10^13-pair brute-force cross product entirely. `src/features.py`
turns each (S1, candidate) pair into ~28 similarity features. A LightGBM
classifier (`src/train_model.py`) is trained on those features, using the
blocking output itself as training data (true matches = positives,
everything else blocking found = hard negatives). `src/predict.py` runs
the same blocking + features + trained model over the test set, tunes a
decision threshold that maximizes the exact macro F_0.5 metric from the
brief (`src/metric.py`), and writes both required output files.

## Setup

```bash
pip install -r requirements.txt
```

Edit the data path when you run (`--data-root`), or edit the default in
`src/config.py`. Expected layout (matches the brief's own structure):

```
<data-root>/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Run it

```bash
# 1. Run the test suite first (seconds, no data needed) — these caught
#    three real bugs during development, worth 30 seconds of your time.
python3 tests/test_metric.py
python3 tests/test_predict_outputs.py
python3 tests/test_embedding_blocking.py
python3 tests/synthetic_integration_test.py

# 2. Train (processes train data country-by-country; first run also
#    builds a parquet cache of each TSV, which speeds up every re-run
#    after that considerably)
python3 src/train_model.py --data-root /path/to/dataset --work-dir ./work

# 3. Predict on the test set
python3 src/predict.py --data-root /path/to/dataset --work-dir ./work --output-dir ./output

# 4. Validate against the brief's own checker before uploading
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching ./output/matching_results.tsv \
    --candidate ./output/candidate_pairs.tsv \
    --test-dir /path/to/dataset/test
```

Step 2 prints, per country, blocking pair-recall and entity-level full-
recall — read this before trusting anything downstream: it's the ceiling
on what the model can possibly achieve. It also prints the tuned
threshold, held-out macro F_0.5, and LightGBM feature importances.

## What's real vs. what still needs your machine

Everything in `src/` has been exercised against real samples of the
actual dataset (pulled via a filesystem connector, since the full files
are 200MB-500MB each) and/or an automated test. Three real bugs were
found and fixed this way — see `tests/` for what's actually verified:
Unicode combining-mark corruption on Devanagari/Kannada/Tamil text,
a blocking frequency-cap that only worked at toy scale, and a
pandas-Series-truthiness crash in the output-writing code that would
have failed partway through a full test-set run.

What hasn't been run: the pipeline against the FULL dataset (only small
samples were available for development — a few thousand rows per file
via head/tail reads). Real blocking recall, real F_0.5, and real runtime
at ~2.2M/~5M/~5.3M row scale can only be measured by actually running
`train_model.py` on your machine. Budget real time for this — see the
memory/runtime notes below.

## Memory / runtime notes

- Peak memory is dominated by holding one country's normalized S1 + S2 +
  S3 frames, plus their candidate/feature frames, at once — processing
  is already split by country in both `train_model.py` and `predict.py`
  for exactly this reason. If even one country is too much for your
  machine's RAM, the natural next cut is chunking by entity_id ranges
  within a country; none of `blocking.py`/`features.py` need to change
  for that, it's purely an outer-loop change in the two orchestration
  scripts.
- Feature engineering measured ~27K pairs/sec single-threaded in this
  environment. `features.compute_features_parallel` splits work across
  `os.cpu_count()` processes automatically — on a multi-core machine this
  is the single biggest speed lever after blocking's own cost.
- `BlockingConfig.max_candidates_per_entity` (default 60) and
  `max_postings_per_key` (default 400) trade recall for speed/memory —
  raise them if your blocking-recall report shows real misses and you
  have compute to spare; the brief explicitly rewards this ("no matter
  the cost" territory is mostly spent here and on the embedding stage
  below, not on the classifier, which is comparatively cheap).

## The optional embedding stage (src/embedding_blocking.py)

Off by default. Token/n-gram blocking cannot find a pair where the two
records are in *different scripts* (a Devanagari name and its romanized
counterpart share zero characters) — EDA on the real data found this
affects roughly a fifth of Indian records in Source 2/3. This module
uses a multilingual sentence-embedding model (sentence-transformers/
LaBSE recommended — Apache-2.0, 470.9M params, license and language
coverage verified directly against the Hugging Face Hub API) plus FAISS
for approximate nearest-neighbor search, to catch exactly those pairs.
The FAISS indexing/search logic itself is unit-tested
(`tests/test_embedding_blocking.py`); the model download requires
internet access this development environment didn't have, so the actual
embedding quality on this dataset is untested — sanity-check it cheaply
first (a live HF demo Space, or a two-line local script) before
committing a full run's time to it. See the module docstring for exact
setup and a slightly-more-current alternative model.

To turn it on: set `BlockingConfig.use_embedding_blocking = True`, then
in your blocking step call `generate_embedding_candidates(...)` per
country and merge with `merge_embedding_candidates(token_cand,
embedding_cand)` before capping — `embedding_score` then flows through
`features.py` automatically (it's already in `FEATURE_COLUMNS`).

## Directory layout (matches the required submission zip)

```
business_entity_resolution/
├── README.md              <- you are here
├── requirements.txt
├── src/                   <- all pipeline code
└── tests/                 <- automated tests + the synthetic integration check
```
