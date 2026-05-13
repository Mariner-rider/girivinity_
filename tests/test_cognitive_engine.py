from app.core.cognitive_engine import CognitiveEngine


def test_classify_technical():
    engine = CognitiveEngine.__new__(CognitiveEngine)
    engine.max_sub_problems = 4
    engine.confidence_threshold = 0.6
    engine.verbose_reasoning = False
    result = engine._classify_reasoning("write a CUDA kernel function")
    assert result == "technical"


def test_classify_factual():
    engine = CognitiveEngine.__new__(CognitiveEngine)
    engine.max_sub_problems = 4
    engine.confidence_threshold = 0.6
    engine.verbose_reasoning = False
    result = engine._classify_reasoning("what is machine learning")
    assert result == "factual"


def test_decompose_returns_steps_for_complex():
    engine = CognitiveEngine.__new__(CognitiveEngine)
    engine.max_sub_problems = 4
    engine.confidence_threshold = 0.6
    engine.verbose_reasoning = False
    result = engine._decompose(
        "compare Python and JavaScript for web development",
        "multi_step",
    )
    assert isinstance(result, list)


def test_think_returns_thought_chain():
    engine = CognitiveEngine.__new__(CognitiveEngine)
    engine.max_sub_problems = 4
    engine.confidence_threshold = 0.6
    engine.verbose_reasoning = False
    chain = engine.think("how does attention work", "context text")
    assert chain.query == "how does attention work"
    assert isinstance(chain.steps, list)
    assert 0.0 <= chain.confidence <= 1.0
