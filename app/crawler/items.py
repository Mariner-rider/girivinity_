from dataclasses import dataclass, field


@dataclass(slots=True)
class CrawledDocument:
    url: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)
