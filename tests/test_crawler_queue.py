from app.crawler.queue import URLQueue


def test_url_queue_deduplicates():
    queue = URLQueue()
    assert queue.push("https://example.com") is True
    assert queue.push("https://example.com") is False
    assert len(queue) == 1
