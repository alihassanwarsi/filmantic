import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from .config import PROCESSED_DATA_PATH
from .data import load_movies
from .features import build_tags, fit_tfidf

MOVIES_CACHE = PROCESSED_DATA_PATH / "movies.pkl"
SIMILARITY_CACHE = PROCESSED_DATA_PATH / "similarity.npy"


def build_recommender(rebuild=False):
    """Build a recommender system based on movie features."""

    if not rebuild and MOVIES_CACHE.exists() and SIMILARITY_CACHE.exists():
        df = pd.read_pickle(MOVIES_CACHE)
        similarity = np.load(SIMILARITY_CACHE)
        return df, similarity

    df = load_movies()
    tags = build_tags(df)
    _, vectors = fit_tfidf(tags)
    similarity = linear_kernel(vectors, vectors).astype("float32")

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    df.to_pickle(MOVIES_CACHE)
    np.save(SIMILARITY_CACHE, similarity)

    return df, similarity


def recommend_movies(title, df, similarity, n=5):
    """Find and return the most similar movies based on a given title."""

    matched = df[df["title"].str.lower() == title.lower()]

    if matched.empty:
        raise KeyError(f"Movie '{title}' not found in the dataset.")

    index = matched.index[0]

    result = df[["movie_id", "title"]].copy()
    result["score"] = similarity[index]
    result = result.drop(index)
    result = result.sort_values("score", ascending=False).head(n)
    return result


if __name__ == "__main__":
    df, similarity = build_recommender()
    print(recommend_movies("Interstellar", df, similarity))
