import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from .config import PROCESSED_DATA_PATH
from .data import load_movies
from .embeddings import build_embeddings
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


def find_movie_ids(title, df):
    """Find and return movie id based on a given title."""

    return df.loc[df["title"].str.lower() == title.lower(), "movie_id"].tolist()


def recommend_movies(movie_id, df, similarity, n=5):
    """Find and return the most similar movies for a given movie id."""

    matched = df.index[df["movie_id"] == movie_id]

    if matched.empty:
        raise KeyError(f"Movie id {movie_id} not found in the dataset.")

    index = matched[0]

    result = df[["movie_id", "title"]].copy()
    result["score"] = similarity[index]
    result = result.drop(index)
    result = result.sort_values("score", ascending=False).head(n)
    return result


def recommend_embedding(movie_id, df, embeddings, n=5):
    """Recommend movies using embedding similarity."""

    matched = df.index[df["movie_id"] == movie_id]

    if matched.empty:
        raise KeyError(f"Movie id {movie_id} not found.")

    index = matched[0]

    scores = embeddings @ embeddings[index]

    ranked_indices = np.argsort(scores)[::-1]
    ranked_indices = [i for i in ranked_indices if i != index][:n]

    result = df.iloc[ranked_indices][["movie_id", "title"]].copy()

    result["score"] = scores[ranked_indices]

    return result


def recommend_hybrid(movie_id, df, tfidf_similarity, embeddings, n=5, k=60):
    """Recommend movies using Reciprocal Rank Fusion."""

    matched = df.index[df["movie_id"] == movie_id]

    if matched.empty:
        raise KeyError(f"Movie id {movie_id} not found.")

    index = matched[0]

    tfidf_scores = tfidf_similarity[index]
    embedding_scores = embeddings @ embeddings[index]

    tfidf_ranks = np.argsort(np.argsort(-tfidf_scores))
    embedding_ranks = np.argsort(np.argsort(-embedding_scores))

    rrf_scores = 1 / (k + tfidf_ranks + 1) + 1 / (k + embedding_ranks + 1)

    rrf_scores[index] = 0

    top_indices = np.argsort(rrf_scores)[::-1][:n]

    result = df.iloc[top_indices][["movie_id", "title"]].copy()
    result["score"] = rrf_scores[top_indices]

    return result


if __name__ == "__main__":
    df, similarity = build_recommender()

    movie_id = find_movie_ids("Interstellar", df)[0]

    print("TF-IDF recommendations:")
    print(recommend_movies(movie_id, df, similarity))

    embeddings = build_embeddings(df)

    print("\nEmbedding recommendations:")
    print(recommend_embedding(movie_id, df, embeddings))

    print("\nHybrid (RRF) recommendations:")
    print(recommend_hybrid(movie_id, df, similarity, embeddings))
