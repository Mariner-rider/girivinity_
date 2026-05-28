from unittest.mock import MagicMock, patch

from app.core.llm_synthesiser import LLMSynthesiser


def test_fallback_when_no_model():
    with patch("app.core.llm_synthesiser.get_engine", return_value=None):
        result = LLMSynthesiser().synthesise(
            "what is gravity", "context text", []
        )
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesise_uses_engine_when_available():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "Test answer"
    with patch(
        "app.core.llm_synthesiser.get_engine", return_value=mock_engine
    ):
        result = LLMSynthesiser().synthesise(
            "query", "context", [], stream=False
        )
    assert result == "Test answer"


def test_stream_returns_iterable():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = iter(["Hello ", "world"])
    with patch(
        "app.core.llm_synthesiser.get_engine", return_value=mock_engine
    ):
        result = LLMSynthesiser().synthesise("q", "ctx", [], stream=True)
    assert "".join(result) == "Hello world"
