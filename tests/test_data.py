from filmantic.data import _director, _parse_names

GENRES = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
CREW = '[{"job": "Editor", "name": "A"}, {"job": "Director", "name": "James Cameron"}]'


def test_parse_names_returns_all_names():
    assert _parse_names(GENRES) == ["Action", "Adventure"]


def test_parse_names_respects_limit():
    assert _parse_names(GENRES, limit=1) == ["Action"]


def test_parse_names_handles_invalid_input():
    assert _parse_names("invalid") == []
    assert _parse_names(None) == []


def test_director_found():
    assert _director(CREW) == "James Cameron"


def test_director_not_found():
    assert _director('[{"job": "Editor", "name": "A"}]') == ""


def test_director_handles_invalid_input():
    assert _director("invalid") == ""
    assert _director(None) == ""
