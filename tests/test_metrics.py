from bank_recs.metrics import apk, coverage_at_k, hit_rate_at_k, mapk, precision_at_k, recall_at_k


def test_apk_exact_order():
    assert apk(["a", "b"], ["a", "b", "c"], k=3) == 1.0


def test_ranking_metrics_basic():
    actual = {1: ["a", "b"], 2: ["c"]}
    pred = {1: ["a", "x", "b"], 2: ["x", "c"]}
    assert mapk(actual, pred, k=3) > 0
    assert precision_at_k(actual, pred, k=3) > 0
    assert recall_at_k(actual, pred, k=3) > 0
    assert hit_rate_at_k(actual, pred, k=3) == 1.0
    assert coverage_at_k(pred, ["a", "b", "c", "x"], k=3) == 1.0
