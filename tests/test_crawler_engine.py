from crawler_engine.engine import CrawlerEngine, PageData


def test_crawler_engine_accepts_quality_content():
    engine = CrawlerEngine(seeds=["https://example.org"])
    page = PageData(
        url="https://example.org/ai",
        html="<html><title>AI News</title><body><p>The model and machine learning system is useful and robust. " + "insightful data " * 200 + "</p></body></html>",
    )
    results = engine.run([page])
    assert len(results) == 1
    assert results[0].language == "en"
    assert "ai" in results[0].topics


def test_crawler_engine_rejects_duplicate_and_spam():
    engine = CrawlerEngine(seeds=["https://example.org"])
    spam_html = "<html><title>Spam</title><body><p>Buy now click here free money</p></body></html>"
    page1 = PageData(url="https://example.org/a", html=spam_html)
    page2 = PageData(url="https://example.org/a", html=spam_html)
    results = engine.run([page1, page2])
    assert results == []


def test_crawler_engine_rejects_low_trust_domain():
    engine = CrawlerEngine(seeds=["abc"], min_trust_score=0.5)
    page = PageData(url="abc", html="<html><body><p>Valid enough content and the data is present.</p></body></html>")
    assert engine.process_page(page) is None
