from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DomainMatch:
    domain: str
    confidence: float
    domain_prompt: str


DOMAIN_KEYWORDS = {
    "cuda_kernels": ["cuda", "gpu", "kernel", "nvcc", "warp", "thread block", "shared memory", "tensor core", "ptx"],
    "space_astronomy": ["space", "isro", "nasa", "planet", "star", "galaxy", "satellite", "rocket", "orbit", "chandrayaan", "gaganyaan", "telescope", "asteroid", "cosmos", "astrophysics"],
    "computer_science": ["algorithm", "data structure", "operating system", "compiler", "database", "network", "tcp", "ip", "computer science", "software", "programming"],
    "three_d_design": ["3d", "blender", "maya", "cad", "render", "animation", "model", "texture", "shader", "opengl", "vulkan", "3d printing", "game engine", "unity", "unreal"],
    "artificial_intelligence": ["machine learning", "deep learning", "neural network", "ai", "transformer", "llm", "gpt", "diffusion", "reinforcement learning", "computer vision", "nlp"],
    "medical_clinical": ["diagnosis", "medicine", "doctor", "patient", "drug", "treatment", "surgery", "hospital", "mbbs", "clinical", "symptom", "disease", "anatomy", "pharmacology"],
    "indian_legal": ["ipc", "bns", "crpc", "bnss", "section", "supreme court", "high court", "law", "legal", "act", "constitution", "fir", "bail", "judgment", "advocate", "case law"],
    "international_law": ["gdpr", "international law", "treaty", "convention", "us law", "uk law", "eu law", "wto", "human rights"],
    "business_strategy": ["startup", "business plan", "pitch deck", "investor", "valuation", "market analysis", "strategy", "vc", "funding", "entrepreneur", "business model"],
    "accounting_finance": ["ca", "chartered accountant", "gst", "income tax", "audit", "balance sheet", "financial", "accounting", "itr", "tds", "cs", "company secretary"],
    "mathematics": ["theorem", "proof", "calculus", "algebra", "statistics", "probability", "matrix", "differential equation", "graph theory", "number theory"],
    "education_pedagogy": ["teach me", "explain", "i want to learn", "how to", "tutorial", "study", "exam", "jee", "neet", "upsc"],
    "research_academia": ["research", "paper", "citation", "journal", "thesis", "literature review", "methodology", "publish", "academic", "peer review"],
    "history_geopolitics": ["history", "mughal", "british india", "independence", "geopolitics", "war", "civilisation", "ancient india"],
    "economics": ["gdp", "rbi", "inflation", "monetary policy", "economics", "market", "fiscal", "budget", "trade"],
}

DOMAIN_EXPERT_PROMPTS = {
    "cuda_kernels": "You are an expert CUDA kernel engineer with deep knowledge of GPU architecture, memory hierarchy, warp-level primitives, and performance optimisation for H100, A100, and V100 GPUs.",
    "space_astronomy": "You are an expert in space science, astrophysics, and astronomy. You have deep knowledge of ISRO, NASA, ESA missions, orbital mechanics, cosmology, and space technology.",
    "indian_legal": "You are an expert in Indian law with comprehensive knowledge of the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), Indian Constitution, Supreme Court judgments, and all major Indian Acts and Codes. Always cite specific section numbers when relevant.",
    "international_law": "You are an expert in international and comparative law, covering UN conventions, US, UK, EU legal systems, international treaties, human rights law, and trade law.",
    "medical_clinical": "You are an expert clinician with knowledge of internal medicine, surgery, pharmacology, diagnostics, and evidence-based treatment guidelines. Always recommend consulting a qualified doctor for medical decisions.",
    "business_strategy": "You are a top-tier strategy consultant and startup advisor. You can build business plans, pitch decks, financial models, market analyses, and growth strategies for Indian and global markets.",
    "accounting_finance": "You are a Chartered Accountant and Company Secretary with deep expertise in Indian taxation (GST, income tax), audit procedures, financial analysis, SEBI regulations, and corporate compliance.",
    "three_d_design": "You are an expert in 3D design, modelling, rendering, and animation using Blender, Maya, CAD tools, game engines like Unity and Unreal, and visual computing pipelines.",
    "research_academia": "You are an expert research assistant and academic writer. You provide properly cited information with real sources, use rigorous methodology, and follow academic writing standards.",
    "education_pedagogy": "You are the world's best teacher. You adapt your explanation to the learner's level — from absolute beginner to expert. You teach step by step, use analogies, check understanding, and guide the learner to the next concept.",
}


class DomainRouter:
    def route(self, query: str) -> DomainMatch:
        q = query.lower()
        scores: dict[str, int] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in q)

        best_domain = max(scores, key=lambda k: scores[k])
        best_score = scores[best_domain]

        if best_score == 0:
            return DomainMatch(domain="general_reasoning", confidence=0.3, domain_prompt="")

        confidence = min(1.0, best_score * 0.25 + 0.25)
        prompt = DOMAIN_EXPERT_PROMPTS.get(best_domain, "")
        logger.info("DomainRouter: query routed to '%s' confidence=%.2f", best_domain, confidence)
        return DomainMatch(domain=best_domain, confidence=confidence, domain_prompt=prompt)

    def get_all_domains(self) -> list[str]:
        return list(DOMAIN_KEYWORDS.keys())
