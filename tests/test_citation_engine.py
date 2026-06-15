from app.core.citation_engine import CitationEngine


def test_generates_citation_from_source():
    engine = CitationEngine()
    sources = [{
        "url": "https://isro.gov.in/chandrayaan3",
        "title": "Chandrayaan-3 Mission Details",
        "text": "The mission was launched in 2023.",
    }]
    citations = engine.generate_citations(sources)
    assert len(citations) == 1
    assert "isro.gov.in" in citations[0].url
    assert citations[0].credibility_score >= 0.8


def test_apa_format_contains_year():
    engine = CitationEngine()
    sources = [{
        "url": "https://thehindu.com/article/2024/test",
        "title": "Test Article",
        "text": "Published in 2024",
    }]
    citations = engine.generate_citations(sources)
    assert len(citations) == 1
    assert "2024" in citations[0].apa


def test_bibtex_format_structure():
    engine = CitationEngine()
    sources = [{
        "url": "https://arxiv.org/abs/2401.12345",
        "title": "Deep Learning Research Paper",
        "text": "Neural networks 2024",
    }]
    citations = engine.generate_citations(sources)
    assert "@" in citations[0].bibtex
    assert "url" in citations[0].bibtex


def test_empty_sources_returns_empty():
    engine = CitationEngine()
    assert engine.generate_citations([]) == []
