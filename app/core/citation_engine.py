from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    title: str
    url: str
    authors: list[str]
    publication: str
    year: str
    citation_type: str
    apa: str
    mla: str
    chicago: str
    bibtex: str
    accessed_date: str
    credibility_score: float


CREDIBILITY_SIGNALS = {
    "high": [
        ".gov.in", ".gov", ".edu", ".ac.in", ".ac.uk",
        "ncbi.nlm.nih.gov", "pubmed", "arxiv.org",
        "nature.com", "science.org", "springer.com",
        "ieee.org", "acm.org", "jstor.org",
        "supremecourt.gov.in", "rbi.org.in",
        "isro.gov.in", "nasa.gov",
    ],
    "medium": [
        "wikipedia.org", "britannica.com",
        "investopedia.com", "healthline.com",
        "livemint.com", "thehindu.com", "ndtv.com",
        "economictimes.com", "reuters.com", "bbc.com",
    ],
    "low": [
        "medium.com", "quora.com", "reddit.com",
        "blog.", "wordpress.com",
    ],
}


class CitationEngine:
    def generate_citations(self, sources: list[dict]) -> list[Citation]:
        citations = []
        for src in sources:
            url = src.get("url", "")
            title = src.get("title", "")
            if not url or not title:
                continue
            citation = self._build_citation(url, title, src)
            if citation:
                citations.append(citation)
        return citations

    def format_citations_block(self, citations: list[Citation], style: str = "apa") -> str:
        if not citations:
            return ""
        lines = ["\n\n**References:**\n"]
        for i, cit in enumerate(citations, 1):
            if style == "apa":
                lines.append(f"[{i}] {cit.apa}")
            elif style == "mla":
                lines.append(f"[{i}] {cit.mla}")
            elif style == "chicago":
                lines.append(f"[{i}] {cit.chicago}")
            elif style == "bibtex":
                lines.append(cit.bibtex)
        return "\n".join(lines)

    def _build_citation(self, url: str, title: str, src: dict) -> Citation | None:
        try:
            authors = self._extract_authors(url)
            year = self._extract_year(src.get("text", ""), url)
            publication = self._extract_publication(url)
            ctype = self._classify_source(url)
            credibility = self._score_credibility(url)
            accessed = datetime.now(timezone.utc).strftime("%B %d, %Y")
            key = self._make_bibtex_key(authors, year, title)

            apa = self._format_apa(authors, year, title, publication, url)
            mla = self._format_mla(authors, title, publication, url, accessed)
            chicago = self._format_chicago(authors, year, title, publication, url)
            bibtex = self._format_bibtex(key, ctype, authors, year, title, publication, url)

            return Citation(
                title=title,
                url=url,
                authors=authors,
                publication=publication,
                year=year,
                citation_type=ctype,
                apa=apa,
                mla=mla,
                chicago=chicago,
                bibtex=bibtex,
                accessed_date=accessed,
                credibility_score=credibility,
            )
        except Exception as exc:
            logger.warning("Citation build failed for %s: %s", url, exc)
            return None

    def _extract_authors(self, url: str) -> list[str]:
        domain = self._extract_domain(url)
        org_map = {
            "isro.gov.in": ["ISRO"],
            "nasa.gov": ["NASA"],
            "rbi.org.in": ["Reserve Bank of India"],
            "supremecourt.gov.in": ["Supreme Court of India"],
            "arxiv.org": ["arXiv"],
            "ncbi.nlm.nih.gov": ["NCBI"],
            "thehindu.com": ["The Hindu"],
            "livemint.com": ["Mint"],
        }
        for key, authors in org_map.items():
            if key in domain:
                return authors
        return ["Unknown Author"]

    def _extract_year(self, text: str, url: str) -> str:
        year_match = re.search(r"/(20\d{2}|19\d{2})/", url)
        if year_match:
            return year_match.group(1)
        year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text[:500])
        if year_match:
            return year_match.group(1)
        return str(datetime.now(timezone.utc).year)

    def _extract_publication(self, url: str) -> str:
        domain = self._extract_domain(url)
        pub_map = {
            "isro.gov.in": "Indian Space Research Organisation",
            "nasa.gov": "NASA",
            "rbi.org.in": "Reserve Bank of India",
            "thehindu.com": "The Hindu",
            "ndtv.com": "NDTV",
            "livemint.com": "Mint",
            "economictimes.com": "The Economic Times",
            "arxiv.org": "arXiv preprint",
            "nature.com": "Nature",
            "wikipedia.org": "Wikipedia",
        }
        for key, pub in pub_map.items():
            if key in domain:
                return pub
        return domain.split(".")[0].title()

    def _classify_source(self, url: str) -> str:
        if any(x in url for x in [".gov", ".gov.in"]):
            return "government"
        if any(x in url for x in ["arxiv", "pubmed", "ncbi", "springer", "nature", "science", "ieee", "acm", "jstor"]):
            return "journal"
        if any(x in url for x in [".edu", ".ac.in", ".ac.uk"]):
            return "academic"
        if any(x in url for x in ["thehindu", "ndtv", "livemint", "bbc", "reuters"]):
            return "news"
        return "web"

    def _score_credibility(self, url: str) -> float:
        for sig in CREDIBILITY_SIGNALS["high"]:
            if sig in url:
                return 0.9
        for sig in CREDIBILITY_SIGNALS["medium"]:
            if sig in url:
                return 0.6
        for sig in CREDIBILITY_SIGNALS["low"]:
            if sig in url:
                return 0.3
        return 0.5

    def _extract_domain(self, url: str) -> str:
        match = re.search(r"https?://([^/]+)", url)
        return match.group(1).lower() if match else url

    def _make_bibtex_key(self, authors: list[str], year: str, title: str) -> str:
        author_part = authors[0].split()[-1].lower() if authors else "unknown"
        title_part = re.sub(r"[^a-zA-Z]", "", title.split()[0].lower()) if title else "untitled"
        return f"{author_part}{year}{title_part}"

    def _format_apa(self, authors: list[str], year: str, title: str, publication: str, url: str) -> str:
        author_str = "; ".join(authors)
        return f"{author_str}. ({year}). {title}. {publication}. Retrieved from {url}"

    def _format_mla(self, authors: list[str], title: str, publication: str, url: str, accessed: str) -> str:
        author_str = ", ".join(authors)
        return f'{author_str}. "{title}." {publication}. Web. Accessed {accessed}. {url}'

    def _format_chicago(self, authors: list[str], year: str, title: str, publication: str, url: str) -> str:
        author_str = ", ".join(authors)
        return f'{author_str}. "{title}." {publication}, {year}. {url}.'

    def _format_bibtex(
        self,
        key: str,
        ctype: str,
        authors: list[str],
        year: str,
        title: str,
        publication: str,
        url: str,
    ) -> str:
        btype = {
            "journal": "article",
            "government": "techreport",
            "academic": "inproceedings",
            "news": "article",
            "web": "misc",
        }.get(ctype, "misc")

        author_str = " and ".join(authors)
        return (
            f"@{btype}{{{key},\n"
            f"  author  = {{{author_str}}},\n"
            f"  title   = {{{title}}},\n"
            f"  journal = {{{publication}}},\n"
            f"  year    = {{{year}}},\n"
            f"  url     = {{{url}}}\n"
            f"}}"
        )
