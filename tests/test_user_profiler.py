from app.profiling.user_profiler import UserProfiler


def test_user_profiler_levels():
    profiler = UserProfiler()
    assert profiler.profile("what is an api model").user_level in {"beginner", "intermediate"}
    assert profiler.profile("Explain faiss quantization retrieval orchestration latency").user_level == "expert"
