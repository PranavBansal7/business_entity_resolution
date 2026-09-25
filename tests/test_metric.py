import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metric import f_beta_for_entity, macro_f_beta


def test_brief_worked_example():
    """From the challenge brief, verbatim:
      predicted: [S2-00047, S2-00193, S3-00812]
      truth:     [S2-00047, S3-00812]
      Precision = 2/3, Recall = 2/2 = 1.0
      F_0.5 = (1.25 x 0.667 x 1.0) / (0.25 x 0.667 + 1.0) = 0.714
    """
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    score = f_beta_for_entity(true, pred, beta=0.5)
    assert abs(score - 0.714) < 0.001, f"expected ~0.714, got {score}"


def test_singleton_correct_is_perfect():
    assert f_beta_for_entity(set(), set(), beta=0.5) == 1.0


def test_singleton_false_merge_is_zero():
    assert f_beta_for_entity(set(), {"S2-1"}, beta=0.5) == 0.0


def test_missed_everything_is_zero():
    assert f_beta_for_entity({"S2-1", "S2-2"}, set(), beta=0.5) == 0.0


def test_perfect_match_is_one():
    assert f_beta_for_entity({"S2-1", "S3-2"}, {"S2-1", "S3-2"}, beta=0.5) == 1.0


def test_precision_weighted_over_recall():
    """F_0.5 should penalize a false positive more than an equivalent false
    negative — that's the whole point of beta=0.5."""
    true = {"S2-1", "S2-2", "S2-3", "S2-4"}
    one_false_positive = f_beta_for_entity(true, true | {"S2-5"}, beta=0.5)
    one_false_negative = f_beta_for_entity(true, true - {"S2-4"}, beta=0.5)
    assert one_false_positive < one_false_negative


def test_macro_average_includes_singletons():
    gt = {"S1-1": {"S2-1"}, "S1-2": set()}
    pred_perfect = {"S1-1": {"S2-1"}, "S1-2": set()}
    assert macro_f_beta(gt, pred_perfect, ["S1-1", "S1-2"]) == 1.0

    pred_false_merge_on_singleton = {"S1-1": {"S2-1"}, "S1-2": {"S2-9"}}
    # one perfect (1.0) + one false merge on a singleton (0.0), averaged
    assert macro_f_beta(gt, pred_false_merge_on_singleton, ["S1-1", "S1-2"]) == 0.5


def test_missing_prediction_row_scored_as_empty():
    """An S1 entity present in gt/all_s1_ids but absent from `pred` must be
    scored as an empty prediction, not skipped — mirrors what the real
    validator/leaderboard would see if a row were missing."""
    gt = {"S1-1": {"S2-1"}}
    pred = {}  # entity missing entirely
    assert macro_f_beta(gt, pred, ["S1-1"]) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
