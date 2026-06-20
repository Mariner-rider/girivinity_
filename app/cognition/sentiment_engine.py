"""
SentimentEngine — hybrid sentiment and emotion detection.

Layer 1 (fast): VADER for polarity (positive/negative/neutral) in < 1ms
Layer 2 (deep): Transformer classifier for fine-grained emotions when Layer 1 detects
                non-neutral signal. Run async to avoid blocking.

Outputs per turn:
  {
    "polarity": "positive|negative|neutral",
    "polarity_score": 0.0–1.0,
    "dominant_emotion": "joy|sadness|anger|fear|surprise|confusion|frustration|curiosity|urgency",
    "emotion_scores": {"joy": 0.1, "frustration": 0.7, ...},
    "intensity": 0.0–1.0,
    "crisis_signal": false,
    "requires_empathy": false,
    "tone_recommendation": "formal|casual|empathetic|technical|instructional"
  }

Crisis detection: looks for distress keywords (suicidal ideation, self-harm) and returns
crisis_signal=True. The AgentController will route to a crisis-aware response mode.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from typing import Any


class _RuleBasedSentimentIntensityAnalyzer:
    POSITIVE = ("great", "awesome", "perfect", "love", "excellent", "happy", "thanks", "good")
    NEGATIVE = ("sad", "bad", "angry", "hate", "awful", "terrible", "wrong", "broken")

    def polarity_scores(self, text: str) -> dict[str, float]:
        lower = text.lower()
        pos = sum(1 for word in self.POSITIVE if word in lower)
        neg = sum(1 for word in self.NEGATIVE if word in lower)
        total = max(pos + neg, 1)
        compound = (pos - neg) / total
        return {"compound": max(-1.0, min(1.0, compound))}


class SentimentEngine:
    CRISIS_KEYWORDS = [
        "kill myself",
        "suicide",
        "want to die",
        "end my life",
        "self harm",
        "hurt myself",
        "no reason to live",
    ]

    DEFAULT_EMOTION_CLASSES = [
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "confusion",
        "frustration",
        "curiosity",
        "urgency",
    ]

    def __init__(self, config: Any) -> None:
        self.config = config
        self.vader = self._build_vader()
        self._transformer_pipeline = None  # Lazy loaded

    def _build_vader(self) -> Any:
        if importlib.util.find_spec("vaderSentiment") is None:
            return _RuleBasedSentimentIntensityAnalyzer()
        vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
        return vader_module.SentimentIntensityAnalyzer()

    def _load_transformer(self) -> None:
        if self._transformer_pipeline is None:
            transformers = importlib.import_module("transformers")
            self._transformer_pipeline = transformers.pipeline(
                "text-classification",
                model=self._cfg("transformer_model", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
                top_k=None,
                device=-1,  # CPU to avoid competing with LLM for GPU
            )

    def analyze(self, text: str) -> dict[str, Any]:
        """Synchronous analysis. Always fast for the VADER/heuristic path."""
        # Crisis check first — always
        text_lower = text.lower()
        crisis = any(kw in text_lower for kw in self.CRISIS_KEYWORDS)

        # VADER polarity
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            polarity = "positive"
        elif compound <= -0.05:
            polarity = "negative"
        else:
            polarity = "neutral"

        polarity_score = abs(compound)

        # Fast heuristic emotion detection
        emotion_scores = self._heuristic_emotions(text_lower, compound)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        intensity = max(emotion_scores.values())

        requires_empathy = (
            dominant_emotion in ("sadness", "fear", "frustration", "anger")
            or crisis
            or intensity > 0.6
        )

        tone = self._recommend_tone(dominant_emotion, polarity, intensity)

        return {
            "polarity": polarity,
            "polarity_score": round(polarity_score, 3),
            "dominant_emotion": dominant_emotion,
            "emotion_scores": {k: round(v, 3) for k, v in emotion_scores.items()},
            "intensity": round(intensity, 3),
            "crisis_signal": crisis,
            "requires_empathy": requires_empathy,
            "tone_recommendation": tone,
        }

    async def analyze_deep(self, text: str) -> dict[str, Any]:
        """Async deep analysis using transformer. Call when VADER shows non-neutral."""
        self._load_transformer()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._transformer_pipeline, text)
        # Merge with VADER results
        base = self.analyze(text)
        emotion_map = {r["label"].lower(): r["score"] for r in results[0]}
        base["emotion_scores"].update(emotion_map)
        base["dominant_emotion"] = max(base["emotion_scores"], key=base["emotion_scores"].get)
        return base

    def _heuristic_emotions(self, text: str, compound: float) -> dict[str, float]:
        """Fast keyword-based emotion scoring. O(n) over emotion keywords."""
        scores = {e: 0.0 for e in self._cfg("emotion_classes", self.DEFAULT_EMOTION_CLASSES)}

        frustration_kw = ["frustrated", "annoying", "doesn't work", "broken", "wrong", "stupid", "useless"]
        curiosity_kw = ["how does", "why does", "what is", "explain", "curious", "wonder", "interesting"]
        urgency_kw = ["asap", "urgent", "immediately", "emergency", "critical", "now", "quick"]
        confusion_kw = ["confused", "don't understand", "unclear", "what do you mean", "help"]
        joy_kw = ["great", "awesome", "perfect", "love it", "excellent", "happy"]
        sadness_kw = ["sad", "depressed", "unhappy", "terrible", "awful"]
        fear_kw = ["scared", "afraid", "worried", "anxious", "nervous"]
        anger_kw = ["angry", "furious", "rage", "hate", "ridiculous"]

        for kw_list, emotion in [
            (frustration_kw, "frustration"),
            (curiosity_kw, "curiosity"),
            (urgency_kw, "urgency"),
            (confusion_kw, "confusion"),
            (joy_kw, "joy"),
            (sadness_kw, "sadness"),
            (fear_kw, "fear"),
            (anger_kw, "anger"),
        ]:
            hit = sum(1 for kw in kw_list if kw in text)
            scores[emotion] = min(1.0, hit * 0.35)

        # VADER compound as baseline for joy/sadness
        if compound > 0.3:
            scores["joy"] = max(scores["joy"], compound)
        elif compound < -0.3:
            scores["sadness"] = max(scores["sadness"], abs(compound))

        # If all zero → neutral
        if max(scores.values()) == 0.0:
            scores["joy"] = 0.05  # ensure a dominant exists

        return scores

    def _recommend_tone(self, emotion: str, polarity: str, intensity: float) -> str:
        if emotion in ("sadness", "fear"):
            return "empathetic"
        if emotion in ("anger", "frustration") and intensity > 0.5:
            return "calm_and_clear"
        if emotion == "curiosity":
            return "instructional"
        if emotion == "urgency":
            return "concise_and_direct"
        if polarity == "positive":
            return "casual"
        return "formal"

    def _cfg(self, name: str, default: Any) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)
