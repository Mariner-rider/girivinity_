from app.profiling.user_profiler import UserProfiler


def test_user_profiler_levels():
    profiler = UserProfiler()
    assert profiler.profile("what is an api model").user_level in {"beginner", "intermediate"}
    assert profiler.profile("Explain faiss quantization retrieval orchestration latency").user_level == "expert"


def test_user_profiler_adaptation_plan_changes_by_level():
    profiler = UserProfiler()
    beginner = profiler.profile("What is AI?")
    expert = profiler.profile(
        "Although the retrieval stack is distributed, explain FAISS quantization tradeoffs and latency impacts."
    )

    assert beginner.adaptation.response_depth == "foundational"
    assert beginner.adaptation.tone == "friendly and guided"
    assert "analogies" in beginner.adaptation.examples

    assert expert.adaptation.response_depth == "deep"
    assert expert.adaptation.tone == "technical and concise"
    assert "advanced" in expert.adaptation.examples


def test_user_profiler_scores_include_complexity_and_domain_signals():
    profiler = UserProfiler()
    result = profiler.profile("Because this API uses vector retrieval and memory context, explain orchestration.")

    assert result.sentence_complexity_score > 0
    assert result.domain_signal_score > 0
