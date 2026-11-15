import pandas as pd
import pytest

from filmantic.evaluation import (
    build_relevant_sets,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_build_relevant_sets():
    df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "genres": [["Action"], ["Action"], ["Comedy"]],
        }
    )

    relevant = build_relevant_sets(df)

    assert relevant[1] == {2}
    assert relevant[2] == {1}
    assert relevant[3] == set()


def test_precision_at_k():
    assert precision_at_k([1, 2, 3, 4], {2, 4}, k=4) == 0.5


def test_recall_at_k():
    assert recall_at_k([1, 2, 3], {2, 3, 4}, k=3) == pytest.approx(2 / 3)


def test_hit_rate_at_k():
    assert hit_rate_at_k([1, 2, 3], {2}, k=3) == 1.0
    assert hit_rate_at_k([1, 2, 3], {4}, k=3) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k([2, 3], {2, 3}, k=2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], {2}, k=2) < 1.0
