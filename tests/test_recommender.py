import numpy as np
import pandas as pd
import pytest

from filmantic.recommender import recommend_movies


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
    result = recommend_movies("Alpha", df, sim, n=3)
    assert result["title"].tolist() == ["Beta", "Delta", "Gamma"]


def test_n_limits_results(tiny):
    df, sim = tiny
    assert len(recommend_movies("Alpha", df, sim, n=2)) == 2


def test_title_match_is_case_insensitive(tiny):
    df, sim = tiny
    assert recommend_movies("alpha", df, sim, n=1)["title"].iloc[0] == "Beta"


def test_unknown_title_raises_keyerror(tiny):
    df, sim = tiny
    with pytest.raises(KeyError):
        recommend_movies("Nope", df, sim)
