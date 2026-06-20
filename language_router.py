"""LanguageRouter — language and script detection plus routing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageDetectionResult:
    detected_language: str
    confidence: float
    script: str
    is_mixed: bool
    secondary_language: str = ""


class LanguageRouter:
    SCRIPT_PATTERNS = {
        "devanagari": re.compile(r"[\u0900-\u097F]"),
        "arabic": re.compile(r"[\u0600-\u06FF]"),
        "cjk": re.compile(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"),
        "cyrillic": re.compile(r"[\u0400-\u04FF]"),
        "latin": re.compile(r"[A-Za-z]"),
    }
    LANG_HINTS = {"devanagari": "hi", "arabic": "ar", "cjk": "zh", "cyrillic": "ru", "latin": "en"}

    def __init__(self, use_langdetect: bool = True) -> None:
        self._use_langdetect = use_langdetect

    def detect(self, text: str) -> LanguageDetectionResult:
        text = text or ""
        script_counts = self._script_counts(text)
        script = self._dominant_script(script_counts)
        active_scripts = [name for name, count in script_counts.items() if count > 0]
        script_mixed = len(active_scripts) > 1

        if script != "latin":
            return LanguageDetectionResult(
                detected_language=self.LANG_HINTS.get(script, "unknown"),
                confidence=0.9 if not script_mixed else 0.75,
                script=script,
                is_mixed=script_mixed,
                secondary_language=self.LANG_HINTS.get(active_scripts[1], "") if script_mixed and len(active_scripts) > 1 else "",
            )

        if self._use_langdetect and text.strip():
            try:
                from langdetect import detect_langs

                probabilities = detect_langs(text)
                top = probabilities[0]
                mixed = len(probabilities) > 1 and probabilities[1].prob > 0.2
                return LanguageDetectionResult(
                    detected_language=str(top.lang),
                    confidence=float(top.prob),
                    script="latin",
                    is_mixed=mixed,
                    secondary_language=str(probabilities[1].lang) if mixed else "",
                )
            except Exception:
                pass

        return LanguageDetectionResult("en", 0.5 if text.strip() else 0.0, "latin", False)

    def _script_counts(self, text: str) -> dict[str, int]:
        return {script: len(pattern.findall(text)) for script, pattern in self.SCRIPT_PATTERNS.items()}

    def _dominant_script(self, counts: dict[str, int]) -> str:
        dominant = max(counts, key=counts.get)
        return dominant if counts[dominant] > 0 else "latin"

    def _detect_script(self, text: str) -> str:
        return self._dominant_script(self._script_counts(text))

    def get_prompt_template(self, lang: str, style: str = "formal") -> str:
        templates = {
            "hi": {"formal": "आप एक सहायक AI हैं। कृपया हिंदी में उत्तर दें।", "casual": "आप एक दोस्ताना AI हैं।"},
            "ar": {"formal": "أنت مساعد ذكاء اصطناعي. يرجى الرد باللغة العربية.", "casual": "أنت مساعد ودود."},
            "zh": {"formal": "您是一位AI助手。请用中文回答。", "casual": "你是一个友好的AI。"},
            "ru": {"formal": "Вы полезный ИИ-ассистент. Отвечайте по-русски.", "casual": "Вы дружелюбный ИИ."},
            "en": {"formal": "You are a helpful AI assistant.", "casual": "You are a friendly AI."},
        }
        return templates.get(lang, templates["en"]).get(style, templates.get(lang, templates["en"])["formal"])

    def should_translate(self, detected: LanguageDetectionResult) -> bool:
        return detected.detected_language != "en" and detected.confidence > 0.7
