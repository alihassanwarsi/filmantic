import numpy as np
import pandas as pd
import pytest

from filmantic.recommender import (
    find_movie_ids,
    recommend_embedding,
    recommend_hybrid,
    recommend_movies,
)


@pytest.fixture
def tiny():
    df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "title": ["Alpha", "Beta", "Gamma", "Delta"],
        }
    )
    similarity = np.array(
        [
            [1.0, 0.9, 0.1, 0.5],
            [0.9, 1.0, 0.2, 0.3],
            [0.1, 0.2, 1.0, 0.4],
            [0.5, 0.3, 0.4, 1.0],
        ],
        dtype="float32",
    )
    return df, similarity


def test_results_are_sorted_and_exclude_the_movie_itself(tiny):
    df, sim = tiny
    result = recommend_movies(1, df, sim, n=3)
    assert result["title"].tolist() == ["Beta", "Delta", "Gamma"]


def test_n_limits_results(tiny):
    df, sim = tiny
    assert len(recommend_movies(1, df, sim, n=2)) == 2


def test_unknown_id_raises_keyerror(tiny):
    df, sim = tiny
    with pytest.raises(KeyError):
        recommend_movies(999, df, sim)


def test_find_movie_ids_is_case_insensitive(tiny):
    df, _ = tiny
    assert find_movie_ids("alpha", df) == [1]


def test_find_movie_ids_returns_all_duplicates():
    df = pd.DataFrame({"movie_id": [268, 2661], "title": ["Batman", "Batman"]})
    assert find_movie_ids("batman", df) == [268, 2661]


@pytest.fixture
def tiny_embeddings(tiny):
    df, _ = tiny
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype="float32",
    )
    return df, embeddings


def test_recommend_embedding(tiny_embeddings):
    df, emb = tiny_embeddings
    result = recommend_embedding(1, df, emb, n=2)
    assert result["title"].tolist() == ["Beta", "Delta"]
    assert len(result) == 2


def test_recommend_embedding_unknown_id_raises_keyerror(tiny_embeddings):
    df, emb = tiny_embeddings
    with pytest.raises(KeyError):
        recommend_embedding(999, df, emb)


def test_find_movie_ids_returns_empty_list_when_not_found(tiny):
    df, _ = tiny
    assert find_movie_ids("Nonexistent Movie", df) == []


@pytest.fixture
def tiny_hybrid(tiny):
    df, similarity = tiny

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.9, np.sqrt(0.19)],
            [0.0, 1.0],
        ],
        dtype="float32",
    )

    return df, similarity, embeddings


def test_recommend_hybrid_combines_rankings(tiny_hybrid):
    df, similarity, embeddings = tiny_hybrid

    result = recommend_hybrid(1, df, similarity, embeddings, n=3)

    assert result["title"].tolist() == ["Beta", "Gamma", "Delta"]
    assert 1 not in result["movie_id"].tolist()


def test_recommend_hybrid_limits_results(tiny_hybrid):
    df, similarity, embeddings = tiny_hybrid

    result = recommend_hybrid(1, df, similarity, embeddings, n=2)

    assert len(result) == 2


def test_recommend_hybrid_unknown_id_raises_keyerror(tiny_hybrid):
    df, similarity, embeddings = tiny_hybrid

    with pytest.raises(KeyError):
        recommend_hybrid(999, df, similarity, embeddings)
