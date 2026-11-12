from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocess import lemmatize

def squash(names):
    """Remove spaces from names and lowercase them."""

    return [n.replace(" ", "").lower() for n in names if n]

def build_tags(df):
    """Combine movie features into a single text string for each movie."""

    overviews = lemmatize(df["overview"].tolist())
    tags = []

    for overview, genres, keywords, cast, director in zip(
        overviews, df["genres"], df["keywords"], df["cast"], df["director"]
    ):
        parts = [overview]
        parts += squash(genres)
        parts += squash(keywords)
        parts += squash(cast)
        parts += squash([director])

        tags.append(" ".join(parts))

    return tags

def fit_tfidf(tags, max_features=5000):
    """Convert tags into TF-IDF vectors and return vectorizer and vectors"""

    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(tags)

    return vectorizer, vectors

if __name__ == "__main__":
    from .data import load_movies
    df = load_movies()
    tags = build_tags(df)
    print(tags[0])
    print()
    _, matrix = fit_tfidf(tags)
    print(matrix.shape)