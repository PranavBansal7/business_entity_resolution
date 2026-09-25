# Methodology Draft — Business Entity Resolution

NOTE: this is a standalone draft written because the filesystem connector
to your local `Documentation_template.md` wasn't available this session.
The content below covers exactly the four things the brief requires
("Methodology used", "Candidate generation/blocking strategy", "Model
architecture and feature engineering", "Any other relevant information")
— paste it into the real template's headings once you can reconnect, and
fill in the bracketed [ACTUAL RUN NUMBERS] once you've run this on the
full dataset, since everything below that isn't from the brief's own
worked example is either from small-sample EDA or from tests on
synthetic/sampled data, not the real leaderboard.

## 1. Methodology used

Three-stage pipeline: (1) normalize every record's name/address, (2) block
— generate a bounded candidate set of plausible S2/S3 matches per S1
entity using multiple cheap, high-recall signals, (3) score each candidate
pair with a supervised binary classifier trained on the blocking output
itself (true ground-truth matches = positives, every other candidate
blocking surfaced = a hard negative), then pick a probability threshold
that maximizes the brief's own macro F_0.5 metric on a held-out,
entity-level validation split (no pair-level leakage).

Normalization deliberately avoids hardcoding to the two countries present
in training (US, India), since the test set adds France with zero
training examples: no per-country branching logic, only per-country
*data* (e.g. the address-abbreviation dictionary has entries for US/India/
France street vocabulary, but the code path is identical regardless of
which country string appears). Country is treated as an opaque label
throughout blocking and features.

## 2. Candidate generation / blocking strategy

Implemented as vectorized inverted-index joins (pandas hash-merges), not
per-record loops — this is what makes blocking tractable at the dataset's
real scale (~2.2M Source-1 records against ~5.06M/~5.31M Source-2/3
records in training; similar order of magnitude in test). A naive cross
product is ~10^13 pairs.

Four blocking keys, unioned (a candidate needs to match on only ONE key
to be retrieved — recall-oriented by design):
1. **name_token** — any shared significant token (>=3 chars) in the
   normalized, suffix-stripped business name
2. **name_prefix** — first 4 characters of the core name
3. **addr_number** — any shared digit run (house/plot/PIN/ZIP number) of
   >=2 digits in the address
4. **name_ngram** — shared 4-character n-grams of the core name (the
   typo-tolerant fallback when tokens don't align exactly)

Each blocking key drops values that are too common to be useful, capped
at an ABSOLUTE posting-list size (400 for name/ngram keys, 100 for
address numbers) rather than a percentage of the dataset — a percentage
cap is fine at prototype scale but would let a common token through with
a six-figure posting list at the real ~5M-row scale, defeating the point
of blocking. This was caught and fixed during development (see Section 4).

Candidates are capped per S1 entity (default 60, configurable) after
blocking, keeping the candidates supported by the most independent
blocking keys when a cut is needed.

**Known blocking gap, addressed but unverified at scale**: token/n-gram
blocking cannot connect two records in different scripts — a Devanagari
name and its romanized counterpart share zero characters. Small-sample
EDA found this affects roughly a fifth of Indian records in Source 2/3
(multiple Indic scripts: Devanagari, Kannada, Tamil, Telugu, Bengali,
Gurmukhi — not just Hindi). An optional module
(`src/embedding_blocking.py`) adds a multilingual sentence-embedding +
FAISS approximate-nearest-neighbor stage specifically to catch these; off
by default because it requires extra dependencies and a model download
this development environment couldn't perform (no internet access to
huggingface.co). [FILL IN: whether you enabled it for the final run, and
its measured effect on blocking recall for India specifically.]

**Blocking-recall report**: `train_model.py` prints, per country,
pair-level recall (what fraction of true matches survived blocking) and
entity-level full-recall (what fraction of S1 entities got ALL their true
matches through blocking) before any model training happens. This is the
hard ceiling on achievable recall — [FILL IN ACTUAL NUMBERS from your run;
on synthetic test data with mild perturbations this was measured at
100%, which should NOT be taken as a real-data estimate].

## 3. Model architecture and feature engineering

**Model**: LightGBM binary classifier (`lightgbm.LGBMClassifier`, MIT
license, gradient-boosted decision trees — a few thousand to low-millions
of numeric split parameters depending on tree count/depth, trivially
within the 8B-parameter cap). Trained with early stopping on a held-out
10% slice of the training pairs; the F_0.5 threshold search afterward
uses a completely separate, untouched validation split at the S1-entity
level (not just pair level, to avoid leaking candidate context for a
validation entity into training).

**~28 pairwise features** per (S1, candidate) pair, computed via
`src/features.py` (RapidFuzz for edit-distance metrics — MIT license):
- Blocking provenance: `n_keys` (how many independent blocking keys
  agreed), one-hot flags for which key(s) fired
- Exact-match flags on three name normalizations (full-clean, suffix-
  stripped core, ASCII-diacritic-folded)
- Name similarity: token Jaccard, character 4-gram Jaccard, Levenshtein
  ratio, token-sort ratio (word-order-invariant — handles the brief's
  documented "word-order transposition" noise), partial ratio (substring
  containment, e.g. DBA-style trailing text), first-token match, length
  ratio/difference
- Legal-suffix signal: has-suffix flags per side + suffix-equality (never
  a hard filter — a missing/wrong suffix only loses one feature's worth
  of signal, never excludes a candidate)
- Address similarity: token Jaccard, shared-number Jaccard/any-shared,
  Levenshtein ratio, token-sort ratio, length ratio, both-have-address flag
- Script signals: both-Indic-script flag, script-mismatch flag
- Embedding channel (optional, NaN when the embedding stage isn't run —
  LightGBM handles this as a genuine missing value via learned split
  defaults, not an imputed constant): cosine similarity from the
  embedding-blocking module

**Training data construction**: positives and hard negatives are BOTH
drawn from blocking's own output (a candidate is labeled 1 if it's a true
ground-truth match, 0 otherwise), so the classifier trains on exactly the
distribution it sees at inference — and negatives are automatically
"hard" near-miss look-alikes rather than random unrelated pairs, which
matters for a precision-weighted metric. Per-entity negative sampling is
weighted toward higher `n_keys` (the more-confusable candidates). This
also means: a true match blocking never surfaces can never become a
training positive OR a correct prediction — reinforcing that blocking
recall is the systemic ceiling, not a detail.

**Threshold selection**: grid search directly against the brief's macro
F_0.5 formula (implemented and unit-tested against the brief's own worked
example — see Section 4), not against a generic classification metric
like accuracy or F1. [FILL IN: final chosen threshold and held-out macro
F_0.5 from your actual run.]

## 4. Other relevant information

**Fair-play / license compliance**: every dependency is MIT, BSD, or
Apache-2.0 (pandas, numpy, LightGBM, RapidFuzz, FAISS, and the
recommended embedding model LaBSE — license verified directly against
the Hugging Face Hub API, not just a cached webpage). No external
business-identity, geocoding, or lookup API is called anywhere in the
pipeline; the only network access anywhere in this codebase is the
one-time pretrained-embedding-weights download in the optional
`embedding_blocking.py` module, which is a model download, not a
per-record lookup against a business database.

**Testing approach**: given development happened against small samples
of the real files (pulled via a filesystem connector — the full files are
200-500MB each) rather than the full dataset, correctness was validated
three ways: (a) unit tests against hand-computable cases, including the
brief's own worked F_0.5 example reproduced exactly (0.714); (b) a
synthetic integration test that perturbs real Source-1 records with the
exact noise patterns found in small-sample EDA (suffix reshuffling,
typos, missing addresses, landmark insertion, word reordering) to verify
the full pipeline wires together correctly end-to-end; (c) direct
inspection against real noisy Source-2/3 records during development.

Three real bugs were caught this way before ever running on full data:
1. A Unicode-`\w`-based punctuation filter was silently stripping
   Devanagari/Kannada/Tamil combining vowel signs, fragmenting Indic-
   script names into isolated consonants — would have corrupted matching
   on a meaningful fraction of Indian records.
2. Blocking's common-token frequency cap was percentage-of-dataset-based,
   which only behaves correctly at small scale — fixed to an absolute cap.
3. A pandas Series-truthiness bug in the output-writing code
   (`if ids:` on a groupby-apply Series raises `ValueError`) would have
   crashed a full test-set run partway through — caught by a unit test
   before ever touching real data.

**Known limitations / where a real run would sharpen this document**:
- Real blocking recall and real macro F_0.5 are unmeasured — only
  reachable by running on the full dataset, which this development
  environment couldn't do (read-only sampled access only).
- The embedding-blocking module's real effect on Indic cross-script
  recall is unmeasured for the same reason.
- [ADD YOUR OWN: hyperparameter tuning notes, ablations, error analysis
  on the validation set, anything you changed after your first real run.]
