from __future__ import annotations

from app.security.policy import BenchmarkResult, SecurityGuard, secure_operation


@secure_operation("finetune.model_update_gate")
def approve_model_update(
    baseline_score: float,
    candidate_score: float,
    benchmark_name: str = "validation_exact_match",
    min_delta: float = 0.0,
    security_guard: SecurityGuard | None = None,
) -> bool:
    """Approve adapter promotion only when benchmark metrics improve."""
    guard = security_guard or SecurityGuard()
    guard.require_benchmark_improvement(
        baseline=BenchmarkResult(name=f"baseline_{benchmark_name}", score=baseline_score),
        candidate=BenchmarkResult(name=f"candidate_{benchmark_name}", score=candidate_score),
        min_delta=min_delta,
    )
    return True
