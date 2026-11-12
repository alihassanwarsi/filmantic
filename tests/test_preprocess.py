from filmantic.preprocess import lemmatize


def test_lemmatize_keeps_one_output_per_input():
    assert len(lemmatize(["a", "b", "c"])) == 3


def test_lemmatize_removes_stopwords_and_lowercases():
    out = lemmatize(["The dogs are running"])[0]
    assert "the" not in out.split()
    assert "are" not in out.split()
    assert out == out.lower()
