from language_router import DictionaryTranslator, LanguageRouter


def test_detects_hindi_and_law_domain():
    router = LanguageRouter(translator=DictionaryTranslator())
    routed = router.route("भारत का कानून क्या है?")

    assert routed.source_language.code == "hi"
    assert "law" in routed.domains


def test_detects_sanskrit_and_cultural_context():
    router = LanguageRouter()
    routed = router.route("धर्म और वेद पर श्लोक")

    assert routed.source_language.code == "sa"
    assert "vedic" in routed.cultural_context
    assert "religion" in routed.domains


def test_detects_tamil_and_dictionary_translation():
    router = LanguageRouter(translator=DictionaryTranslator())
    routed = router.route("வணக்கம் தமிழ் வரலாறு")

    assert routed.source_language.code == "ta"
    assert "hello" in routed.translated_text
    assert "history" in routed.domains
