"""Ranking metrics for bank product recommendations."""

from __future__ import annotations

from collections.abc import Iterable


def apk(actual: list[str], predicted: list[str], k: int = 7) -> float:
    """Average precision at k for one user."""
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    hits = 0.0
    seen = set()
    actual_set = set(actual)
    for i, product in enumerate(predicted, start=1):
        if product in actual_set and product not in seen:
            hits += 1.0
            score += hits / i
            seen.add(product)
    return score / min(len(actual), k)


def mapk(actual: dict[int, list[str]], predicted: dict[int, list[str]], k: int = 7) -> float:
    """Mean average precision at k."""
    if not actual:
        return 0.0
    return sum(apk(actual.get(user, []), predicted.get(user, []), k) for user in actual) / len(actual)


def precision_at_k(actual: dict[int, list[str]], predicted: dict[int, list[str]], k: int = 7) -> float:
    """Mean precision@k over users with at least one actual product."""
    scores = []
    for user, actual_products in actual.items():
        if not actual_products:
            continue
        pred = predicted.get(user, [])[:k]
        scores.append(len(set(actual_products) & set(pred)) / k)
    return float(sum(scores) / len(scores)) if scores else 0.0


def recall_at_k(actual: dict[int, list[str]], predicted: dict[int, list[str]], k: int = 7) -> float:
    """Mean recall@k over users with at least one actual product."""
    scores = []
    for user, actual_products in actual.items():
        if not actual_products:
            continue
        pred = predicted.get(user, [])[:k]
        scores.append(len(set(actual_products) & set(pred)) / len(set(actual_products)))
    return float(sum(scores) / len(scores)) if scores else 0.0


def hit_rate_at_k(actual: dict[int, list[str]], predicted: dict[int, list[str]], k: int = 7) -> float:
    """Share of users for whom at least one recommended product is relevant."""
    users = [user for user, products in actual.items() if products]
    if not users:
        return 0.0
    hits = 0
    for user in users:
        if set(actual[user]) & set(predicted.get(user, [])[:k]):
            hits += 1
    return hits / len(users)


def coverage_at_k(predicted: dict[int, list[str]], all_products: Iterable[str], k: int = 7) -> float:
    """Share of available products that appear in top-k recommendations."""
    all_products = set(all_products)
    if not all_products:
        return 0.0
    recommended = set()
    for products in predicted.values():
        recommended.update(products[:k])
    return len(recommended & all_products) / len(all_products)


def ranking_metrics(
    actual: dict[int, list[str]],
    predicted: dict[int, list[str]],
    all_products: Iterable[str],
    k: int = 7,
) -> dict[str, float]:
    """Collect all ranking metrics in one dict."""
    return {
        f"map_at_{k}": mapk(actual, predicted, k),
        f"precision_at_{k}": precision_at_k(actual, predicted, k),
        f"recall_at_{k}": recall_at_k(actual, predicted, k),
        f"hitrate_at_{k}": hit_rate_at_k(actual, predicted, k),
        f"coverage_at_{k}": coverage_at_k(predicted, all_products, k),
    }
