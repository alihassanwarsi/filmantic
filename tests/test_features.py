import pandas as pd
from filmantic.features import build_tags, fit_tfidf

def _tiny_df():
    return pd.DataFrame({
        "overview": ["A hero saves the city", "A detective hunts a killer"],
        "genres": [["Action"], ["Crime", "Thriller"]],
        "keywords": [["super hero"], ["serial killer"]],
        "cast": [["Tom Hardy"], ["Brad Pitt"]],
        "director": ["Chris Nolan", "David Fincher"]
    })

def test_build_tags_includes_squashed_metadata():
    tags = build_tags(_tiny_df())
    assert len(tags) == 2
    assert "tomhardy" in tags[0].split()
    assert "davidfincher" in tags[1].split()

def test_fit_tfidf_shape():
    tags = build_tags(_tiny_df())
    _, vectors = fit_tfidf(tags)
    assert vectors.shape[0] == 2