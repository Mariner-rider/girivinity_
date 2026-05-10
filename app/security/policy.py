from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar
from urllib.parse import urlparse

P = ParamSpec("P")
T = TypeVar("T")


class SecurityPolicyError(PermissionError):
    """Raised when a subsystem violates a required safety policy."""


@dataclass(frozen=True, slots=True)
class TrustScore:
    url: str
    score: float
    reasons: tuple[str, ...]

    @property
    def trusted(self) -> bool:
        return self.score >= 0.6


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    score: float


class SecurityGuard:
    """Central guardrail layer used by runtime, crawler, RAG, and finetuning modules."""

    def validate_prompt(self, prompt: str) -> None:
        if not prompt or not prompt.strip():
            raise SecurityPolicyError("Prompt must be non-empty.")

    def require_grounding(self, sources: list[dict] | list[str], context: str) -> None:
        if not sources:
            raise SecurityPolicyError("Refusing to produce output without RAG/source grounding.")
        if not context.strip():
            raise SecurityPolicyError("Refusing to produce output without grounded context.")

    def score_url_trust(self, url: str) -> TrustScore:
        parsed = urlparse(url)
        reasons: list[str] = []
        score = 0.0

        if parsed.scheme == "https":
            score += 0.45
            reasons.append("https")
        elif parsed.scheme == "http":
            score += 0.2
            reasons.append("http")

        if parsed.netloc and "." in parsed.netloc:
            score += 0.25
            reasons.append("valid_host")

        lowered = parsed.netloc.lower()
        if any(domain in lowered for domain in (".edu", ".gov", ".org")):
            score += 0.2
            reasons.append("reputable_tld")

        if not parsed.fragment:
            score += 0.1
            reasons.append("no_fragment")

        return TrustScore(url=url, score=round(min(score, 1.0), 3), reasons=tuple(reasons))

    def require_trusted_url(self, url: str) -> TrustScore:
        trust = self.score_url_trust(url)
        if not trust.trusted:
            raise SecurityPolicyError(f"Crawler URL failed trust scoring: {url}")
        return trust

    def require_validation_dataset(self, validation_dataset_path: str | Path | None) -> None:
        if validation_dataset_path is None:
            raise SecurityPolicyError("Training requires a validation dataset path.")
        path = Path(validation_dataset_path)
        if not path.exists() or path.stat().st_size == 0:
            raise SecurityPolicyError("Training validation dataset must exist and be non-empty.")

    def require_benchmark_improvement(
        self,
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
        min_delta: float = 0.0,
    ) -> None:
        if candidate.score <= baseline.score + min_delta:
            raise SecurityPolicyError(
                f"Model update rejected: {candidate.name}={candidate.score} did not improve "
                f"over {baseline.name}={baseline.score} by > {min_delta}."
            )


def secure_operation(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Annotate guarded module entrypoints without changing their public signatures."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__security_operation__ = name  # type: ignore[attr-defined]
        return wrapper

    return decorator
