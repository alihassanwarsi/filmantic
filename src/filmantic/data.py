import ast

import pandas as pd

from .config import RAW_DATA_PATH


def _parse_names(text, limit=None):
    """Extract names from text containing a list of dictionaries."""
    try:
        data = ast.literal_eval(text)
        names = [item["name"] for item in data]

    except (ValueError, TypeError, SyntaxError):
        return []

    return names[:limit] if limit else names


def _director(text):
    """Find and return the director's name from the movie crew list."""

    try:
        for item in ast.literal_eval(text):
            if item.get("job") == "Director":
                return item.get("name", "")
    except (ValueError, TypeError, SyntaxError, AttributeError):
        return ""

    return ""


def load_movies():
    """Load datasets, combine them, and return processed dataset"""

    movies = pd.read_csv(RAW_DATA_PATH / "tmdb_5000_movies.csv")
    credits = pd.read_csv(RAW_DATA_PATH / "tmdb_5000_credits.csv")

    df = movies.merge(credits.drop(columns="title"), left_on="id", right_on="movie_id")
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    df = df.dropna(subset=["overview"]).reset_index(drop=True)

    df["genres"] = df["genres"].apply(_parse_names)
    df["keywords"] = df["keywords"].apply(_parse_names)
    df["cast"] = df["cast"].apply(_parse_names, limit=3)
    df["director"] = df["crew"].apply(_director)

    return df.drop(columns="crew")


if __name__ == "__main__":
    df = load_movies()
    print(df.shape)
    print(df.iloc[0][["title", "genres", "cast", "director"]])
