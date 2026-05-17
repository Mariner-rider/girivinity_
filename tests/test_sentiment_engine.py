from unittest.mock import patch
from app.core.sentiment_engine import SentimentEngine


def _engine():
    e = SentimentEngine.__new__(SentimentEngine)
    e.store_history = False
    return e


def test_detects_frustrated_emotion():
    e = _engine()
    result = e._detect_emotion("this is not working again broken")
    assert result == "frustrated"


def test_detects_hindi_language():
    e = _engine()
    result = e._detect_language("यह क्या है")
    assert result == "hindi"


def test_detects_english_language():
    e = _engine()
    result = e._detect_language("what is machine learning")
    assert result == "english"


def test_detects_urgency():
    e = _engine()
    result = e._detect_urgency("I need this urgently asap now")
    assert result > 0.5


def test_expertise_expert():
    e = _engine()
    result = e._detect_expertise(
        "backpropagation gradient optimization algorithm"
    )
    assert result == "expert"


def test_expertise_beginner():
    e = _engine()
    result = e._detect_expertise("how to learn python for beginners")
    assert result == "beginner"


def test_analyse_returns_profile():
    e = _engine()
    profile = e.analyse("help me fix this error", "user123")
    assert profile.user_id == "user123"
    assert profile.emotion in [
        "frustrated","confused","excited","urgent",
        "curious","sad","neutral"
    ]
    assert profile.response_style != ""
