"""Multilingual routing layer with Indian language support.

This module intentionally uses deterministic heuristics and pluggable translation
adapters so it can run in constrained/offline environments while remaining easy
to replace with production translation providers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from app.security.policy import secure_operation


@dataclass(frozen=True, slots=True)
class LanguageMatch:
    code: str
    name: str
    confidence: float
    script: str


@dataclass(frozen=True, slots=True)
class TranslationResult:
    source_language: LanguageMatch
    target_language: str
    original_text: str
    translated_text: str
    provider: str


@dataclass(frozen=True, slots=True)
class RoutedText:
    source_language: LanguageMatch
    target_language: str
    original_text: str
    normalized_text: str
    translated_text: str
    cultural_context: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)


class Translator(Protocol):
    provider: str

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        ...


class IdentityTranslator:
    """Safe fallback translator that preserves text when no provider is configured."""

    provider = "identity"

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        return text


class DictionaryTranslator:
    """Tiny deterministic translator useful for tests and offline demos."""

    provider = "dictionary"

    _TERMS = {
        ("hi", "en"): {
            "नमस्ते": "hello",
            "इतिहास": "history",
            "कानून": "law",
            "रामायण": "Ramayana",
            "महाभारत": "Mahabharata",
        },
        ("sa", "en"): {
            "धर्म": "dharma",
            "वेद": "Veda",
            "पुराण": "Purana",
            "रामायण": "Ramayana",
        },
        ("ta", "en"): {
            "வணக்கம்": "hello",
            "வரலாறு": "history",
            "சட்டம்": "law",
            "கம்பராமாயணம்": "Kamba Ramayanam",
        },
    }

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        translated = text
        for source, target in self._TERMS.get((source_language, target_language), {}).items():
            translated = translated.replace(source, target)
        return translated


class LanguageRouter:
    """Detects language, translates text, and annotates cultural/domain context."""

    _LANGUAGES = {
        "en": ("English", "Latin"),
        "hi": ("Hindi", "Devanagari"),
        "sa": ("Sanskrit", "Devanagari"),
        "ta": ("Tamil", "Tamil"),
        "bn": ("Bengali", "Bengali"),
        "gu": ("Gujarati", "Gujarati"),
        "kn": ("Kannada", "Kannada"),
        "ml": ("Malayalam", "Malayalam"),
        "pa": ("Punjabi", "Gurmukhi"),
        "te": ("Telugu", "Telugu"),
        "ur": ("Urdu", "Arabic"),
    }

    _SCRIPT_RANGES = {
        "hi": ("Devanagari", re.compile(r"[\u0900-\u097F]")),
        "ta": ("Tamil", re.compile(r"[\u0B80-\u0BFF]")),
        "bn": ("Bengali", re.compile(r"[\u0980-\u09FF]")),
        "gu": ("Gujarati", re.compile(r"[\u0A80-\u0AFF]")),
        "kn": ("Kannada", re.compile(r"[\u0C80-\u0CFF]")),
        "ml": ("Malayalam", re.compile(r"[\u0D00-\u0D7F]")),
        "pa": ("Gurmukhi", re.compile(r"[\u0A00-\u0A7F]")),
        "te": ("Telugu", re.compile(r"[\u0C00-\u0C7F]")),
        "ur": ("Arabic", re.compile(r"[\u0600-\u06FF]")),
    }

    _SANSKRIT_HINTS = {
        "धर्म",
        "वेद",
        "उपनिषद",
        "पुराण",
        "संस्कृत",
        "श्लोक",
        "मोक्ष",
        "कर्म",
    }

    _HINDI_HINTS = {"है", "मैं", "क्या", "क्यों", "इतिहास", "कानून", "सरकार"}

    _CULTURAL_TAGS = {
        "vedic": {"वेद", "उपनिषद", "ऋग्वेद", "यजुर्वेद", "सामवेद", "अथर्ववेद"},
        "epic": {"रामायण", "महाभारत", "गीता", "கம்பராமாயணம்", "Ramayana", "Mahabharata"},
        "dharmic": {"धर्म", "कर्म", "मोक्ष", "dharma", "karma", "moksha"},
        "dravidian": {"தமிழ்", "சங்கம்", "திருக்குறள்", "கம்பராமாயணம்"},
        "legal_india": {"संविधान", "कानून", "சட்டம்", "IPC", "constitution"},
    }

    _DOMAIN_TERMS = {
        "history": {"history", "इतिहास", "வரலாறு", "empire", "dynasty", "king", "colonial"},
        "mythology": {"रामायण", "महाभारत", "पुराण", "mythology", "deva", "asura"},
        "law": {"law", "कानून", "சட்டம்", "constitution", "court", "rights", "IPC"},
        "religion": {"वेद", "धर्म", "उपनिषद", "temple", "ritual", "mantra", "श्लोक"},
        "literature": {"poem", "काव्य", "சங்கம்", "திருக்குறள்", "epic", "novel"},
    }

    def __init__(self, translator: Translator | None = None, target_language: str = "en") -> None:
        self.translator = translator or IdentityTranslator()
        self.target_language = target_language

    @secure_operation("language.detect")
    def detect_language(self, text: str) -> LanguageMatch:
        normalized = self._normalize(text)
        if not normalized:
            return LanguageMatch("und", "Unknown", 0.0, "Unknown")

        script_scores: dict[str, int] = {}
        for code, (_, pattern) in self._SCRIPT_RANGES.items():
            script_scores[code] = len(pattern.findall(normalized))

        best_code = max(script_scores, key=script_scores.get)
        best_score = script_scores[best_code]
        if best_score == 0:
            return LanguageMatch("en", "English", 0.65, "Latin")

        code = self._disambiguate_devanagari(best_code, normalized)
        name, script = self._LANGUAGES[code]
        confidence = min(0.99, 0.55 + best_score / max(len(normalized), 1))
        return LanguageMatch(code, name, round(confidence, 3), script)

    @secure_operation("language.translate")
    def translate(
        self,
        text: str,
        target_language: str | None = None,
        source_language: str | None = None,
    ) -> TranslationResult:
        detected = self.detect_language(text)
        source = source_language or detected.code
        target = target_language or self.target_language
        translated = text if source == target else self.translator.translate(text, source, target)
        return TranslationResult(
            source_language=detected,
            target_language=target,
            original_text=text,
            translated_text=translated,
            provider=self.translator.provider,
        )

    @secure_operation("language.route")
    def route(self, text: str, target_language: str | None = None) -> RoutedText:
        translation = self.translate(text, target_language=target_language)
        combined = f"{translation.original_text}\n{translation.translated_text}"
        return RoutedText(
            source_language=translation.source_language,
            target_language=translation.target_language,
            original_text=translation.original_text,
            normalized_text=self._normalize(text),
            translated_text=translation.translated_text,
            cultural_context=self.tag_cultural_context(combined),
            domains=self.classify_domain(combined),
        )

    @secure_operation("language.cultural_context")
    def tag_cultural_context(self, text: str) -> list[str]:
        lowered = text.lower()
        tags = []
        for tag, terms in self._CULTURAL_TAGS.items():
            if any(term.lower() in lowered for term in terms):
                tags.append(tag)
        return tags

    @secure_operation("language.domain_classify")
    def classify_domain(self, text: str) -> list[str]:
        lowered = text.lower()
        domains = []
        for domain, terms in self._DOMAIN_TERMS.items():
            if any(term.lower() in lowered for term in terms):
                domains.append(domain)
        return domains or ["general"]

    def _disambiguate_devanagari(self, detected_code: str, text: str) -> str:
        if detected_code != "hi":
            return detected_code
        tokens = set(re.findall(r"[\u0900-\u097F]+", text))
        if tokens & self._SANSKRIT_HINTS:
            return "sa"
        if tokens & self._HINDI_HINTS:
            return "hi"
        return "hi"

    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().split())
