from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DOMAINS = {

    # ── SCIENCE & TECHNOLOGY ──────────────────────────────

    "cuda_kernels": {
        "description": "CUDA GPU kernel programming and optimisation",
        "search_queries": [
            "CUDA kernel optimization shared memory coalescing",
            "CUDA warp shuffle reduction tensor core",
            "CUDA flash attention implementation H100 A100",
            "PTX assembly GPU optimization tricks",
            "CUDA occupancy register pressure optimization",
            "GPU parallel reduction algorithms CUDA",
            "CUDA cooperative groups warp primitives",
            "CUDA graph execution pipeline optimization",
        ],
        "weight": 2.0,
    },
    "space_astronomy": {
        "description": "Space science, astronomy, astrophysics, NASA, ISRO missions",
        "search_queries": ["ISRO Chandrayaan Gaganyaan mission details", "NASA Mars mission space exploration", "black hole physics relativity cosmology", "exoplanet discovery methods telescope", "satellite orbital mechanics launch vehicles", "dark matter dark energy universe expansion", "space station ISS living working space", "asteroid comet meteor astronomy", "solar system planets formation", "gravitational waves LIGO astronomy", "James Webb telescope discoveries 2024 2025", "India space programme future missions ISRO"],
        "weight": 1.5,
    },
    "computer_science": {"description": "Algorithms, data structures, system design, OS, networks", "search_queries": ["data structures algorithms time complexity", "system design distributed systems scalability", "computer networks TCP IP OSI model", "operating systems process memory management", "database design SQL NoSQL optimization", "compiler design theory automata", "computer architecture CPU cache memory hierarchy", "parallel computing concurrency threading", "software engineering design patterns SOLID", "microservices architecture Docker Kubernetes", "graph algorithms dynamic programming", "cryptography security algorithms RSA AES"], "weight": 2.0},
    "three_d_design": {"description": "3D modelling, CAD, visual computing, rendering, animation", "search_queries": ["3D modelling Blender Maya tutorial techniques", "CAD design AutoCAD SolidWorks engineering", "3D rendering ray tracing path tracing", "computer graphics OpenGL Vulkan shader", "3D animation rigging skinning character", "photogrammetry 3D scanning reconstruction", "procedural generation 3D assets texturing", "3D printing design guidelines materials", "game engine Unity Unreal 3D environment", "architectural visualisation BIM modelling", "VFX visual effects compositing pipeline", "point cloud 3D reconstruction LiDAR"], "weight": 1.5},
    "artificial_intelligence": {"description": "ML, deep learning, NLP, computer vision, AI research", "search_queries": ["transformer architecture attention mechanism", "reinforcement learning reward policy gradient", "computer vision CNN image recognition", "natural language processing NLP BERT GPT", "generative AI diffusion models stable diffusion", "federated learning privacy preserving ML", "AI ethics bias fairness responsible AI", "graph neural networks knowledge graph", "multimodal AI vision language models", "AI research papers 2024 2025 advances", "model quantization pruning compression", "AI in healthcare diagnosis prediction"], "weight": 2.0},
    "medical_clinical": {"description": "Clinical medicine, diagnosis, pharmacology, surgery", "search_queries": ["clinical diagnosis differential MBBS medicine", "pharmacology drug mechanism dose side effects", "surgery operative technique procedures", "internal medicine treatment guidelines India", "emergency medicine critical care ICU", "pathology disease mechanisms histology", "radiology CT MRI X-ray interpretation", "paediatrics child health development", "gynaecology obstetrics pregnancy childbirth", "psychiatry mental health treatment DSM", "cardiology heart disease ECG treatment", "neurology brain nervous system disorders"], "weight": 2.0},
    "ayurveda_traditional": {"description": "Ayurveda, Unani, Siddha, traditional Indian medicine", "search_queries": ["Ayurveda herbs remedies treatment", "Panchakarma detox Ayurvedic treatment", "Unani medicine traditional treatment", "yoga therapy health benefits pranayama", "naturopathy alternative medicine India", "Siddha medicine Tamil traditional healing"], "weight": 1.0},
    "psychology_mental_health": {"description": "Psychology, cognitive science, mental health, therapy", "search_queries": ["cognitive behavioural therapy CBT techniques", "psychology theories Freud Jung Maslow", "mental health anxiety depression treatment", "neuroscience brain cognition memory", "positive psychology well-being happiness", "child psychology development stages", "counselling therapy techniques approaches", "trauma PTSD treatment recovery", "personality disorders assessment", "Indian mental health cultural context"], "weight": 1.5},
    "indian_legal": {"description": "BNS, IPC, CrPC, BNSS, Constitution, Supreme Court", "search_queries": ["Bharatiya Nyaya Sanhita BNS 2023 sections", "Bharatiya Nagarik Suraksha Sanhita BNSS", "Indian Constitution fundamental rights DPSP", "Supreme Court India landmark judgment 2024", "Indian Penal Code IPC sections offences", "CrPC criminal procedure code India", "Indian Evidence Act provisions digital evidence", "GST income tax India provisions", "RTI act India transparency", "Consumer Protection Act India 2019", "POCSO act child protection India", "Cyber law IT act India section 66", "Property law transfer act India", "Family law Hindu Marriage Act India", "Labour law employment India 2024"], "weight": 2.0},
    "international_law": {"description": "International law, comparative law, US UK EU laws", "search_queries": ["international law treaty convention UN", "US law constitution Supreme Court ruling", "UK law common law precedent", "European Union GDPR regulation law", "WTO trade law disputes", "human rights international UDHR ICCPR", "investment law arbitration ICSID", "maritime law shipping convention", "intellectual property WIPO patent trademark", "comparative constitutional law countries"], "weight": 1.5},
    "corporate_governance": {"description": "Companies Act, SEBI, corporate law, governance", "search_queries": ["Companies Act 2013 India provisions", "SEBI regulations securities law India", "corporate governance board directors", "mergers acquisitions India law process", "insolvency IBC NCLT India", "startup legal compliance India", "corporate social responsibility CSR India", "foreign direct investment FDI India regulations"], "weight": 1.5},
    "business_strategy": {"description": "Strategy, startups, business plans, pitch decks, consulting", "search_queries": ["business plan startup India template", "pitch deck investor presentation guide", "business model canvas strategy", "startup funding Series A B valuation India", "market analysis competitive strategy Porter", "go-to-market strategy product launch", "OKR KPI goal setting business", "product management roadmap agile", "growth hacking user acquisition startup", "business problem solving McKinsey framework", "SWOT analysis market research India", "scaling startup operations India unicorn"], "weight": 2.0},
    "accounting_finance": {"description": "CA, CS, accounting, taxation, financial analysis, audit", "search_queries": ["CA chartered accountant exam India syllabus", "income tax India computation filing 2024", "GST return filing India compliance", "financial statement analysis balance sheet", "audit procedure internal external audit India", "cost accounting management accounting", "CS company secretary compliance India", "IFRS accounting standards India IndAS", "corporate taxation India transfer pricing", "financial modelling valuation DCF", "stock market equity analysis India NSE BSE", "mutual fund investment analysis India"], "weight": 2.0},
    "economics": {"description": "Macroeconomics, microeconomics, Indian economy, policy", "search_queries": ["Indian economy GDP growth RBI policy", "macroeconomics monetary fiscal policy", "microeconomics supply demand market", "public policy welfare economics India", "international trade economics", "development economics poverty inequality", "behavioural economics nudge theory", "RBI monetary policy inflation India", "Union Budget India analysis", "economic survey India sectors"], "weight": 1.5},
    "education_pedagogy": {"description": "Teaching methods, curriculum, exam prep, all subjects", "search_queries": ["teaching methods pedagogy Bloom's taxonomy", "CBSE NCERT curriculum India syllabus", "JEE NEET competitive exam preparation India", "UPSC IAS exam preparation strategy", "learning theory constructivism Vygotsky", "online education e-learning platforms India", "special education inclusive learning", "assessment evaluation rubrics education", "STEM education science maths teaching", "language learning English Hindi teaching"], "weight": 1.5},
    "mathematics": {"description": "Pure math, applied math, statistics, probability, ML math", "search_queries": ["linear algebra matrix operations proofs", "calculus differential equations solutions", "probability statistics theorems", "number theory proofs cryptography", "graph theory algorithms combinatorics", "numerical methods computational mathematics", "optimization convex non-convex theory", "topology abstract algebra group theory", "information theory entropy coding", "mathematical proofs logic foundations"], "weight": 1.5},
    "engineering_technical": {"description": "Electronics, mechanical, civil, chemical engineering", "search_queries": ["electronics circuit design PCB", "signal processing DSP filter design", "control systems PID controller design", "mechanical engineering thermodynamics", "civil structural engineering design", "chemical process engineering", "VLSI chip design semiconductor", "embedded systems Arduino Raspberry Pi", "IoT Internet of Things sensor networks", "renewable energy solar wind systems", "manufacturing process automation Industry 4.0", "materials science engineering properties"], "weight": 1.5},
    "hindi_regional": {"description": "Hindi, regional Indian languages, Devanagari, Indian culture", "search_queries": ["हिंदी व्याकरण नियम वाक्य संरचना", "भारतीय इतिहास संस्कृति परंपरा", "हिंदी साहित्य कविता कहानी", "भारत की भूगोल नदी पर्वत", "भारतीय त्योहार धर्म समाज", "हिंदी में विज्ञान तकनीक", "संस्कृत श्लोक अर्थ व्याख्या", "Indian classical music Hindustani Carnatic", "Indian art painting sculpture architecture"], "weight": 1.5},
    "research_academia": {"description": "Academic research, scientific papers, citations, methodology", "search_queries": ["research methodology literature review", "academic writing thesis dissertation", "scientific paper structure abstract methods", "citation styles APA MLA Chicago IEEE", "peer review academic journal publishing", "research ethics plagiarism academic integrity", "systematic review meta-analysis methodology", "qualitative quantitative research methods", "data analysis SPSS R Python research", "grant writing research proposal funding"], "weight": 2.0},
    "creative_writing": {"description": "Creative writing, storytelling, content creation, screenwriting", "search_queries": ["creative writing fiction short story techniques", "screenwriting script format Hollywood", "content writing SEO blog article", "copywriting persuasion marketing", "poetry writing techniques forms", "novel writing plot character development", "journalism reporting news writing", "technical writing documentation"], "weight": 1.0},
    "environment_climate": {"description": "Climate change, environment, sustainability, green tech", "search_queries": ["climate change global warming solutions India", "renewable energy solar wind India policy", "environmental law NGT India", "biodiversity conservation wildlife India", "water management irrigation India", "pollution control air water soil India", "carbon footprint net zero sustainability", "electric vehicles EV India policy market"], "weight": 1.0},
    "agriculture_rural": {"description": "Indian agriculture, farming, rural development, food security", "search_queries": ["Indian agriculture crops farming techniques", "precision agriculture technology India", "farmer income policy MSP India", "irrigation water management agriculture India", "organic farming sustainable agriculture", "agricultural loans credit India", "food processing industry India", "rural development government schemes India"], "weight": 1.0},
    "history_geopolitics": {"description": "World history, Indian history, geopolitics, international relations", "search_queries": ["Indian history Maurya Mughal British", "world history modern contemporary", "India China relations geopolitics", "India Pakistan relations Kashmir", "BRICS G20 India multilateral", "ancient Indian history Vedic period", "freedom struggle independence India", "geopolitics South Asia SAARC", "Indian foreign policy non-alignment", "World War history causes effects"], "weight": 1.0},
    "general_reasoning": {"description": "Logic, critical thinking, problem solving, philosophy", "search_queries": ["logical reasoning puzzles solutions", "critical thinking frameworks argumentation", "philosophy ethics moral reasoning", "decision making cognitive bias", "Socratic method reasoning inquiry"], "weight": 1.0},
    "sports_fitness": {"description": "Sports science, fitness, nutrition, cricket, Indian sports", "search_queries": ["cricket coaching tactics India IPL", "fitness training exercise science", "sports nutrition diet athlete", "yoga asana benefits practice", "sports psychology performance mental", "Indian sports kabaddi wrestling"], "weight": 1.0},
}



@dataclass
class DomainDataset:
    domain: str
    samples: list[dict] = field(default_factory=list)
    token_count: int = 0

    def add(self, instruction: str, response: str, source: str = "") -> None:
        self.samples.append(
            {
                "instruction": instruction,
                "response": response,
                "source": source,
                "domain": self.domain,
            }
        )
        self.token_count += len(instruction.split()) + len(response.split())


class DomainCrawler:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        self.output_dir = Path(
            cfg.get("domain_training", {}).get("data_dir", "data/domain_training")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_per_domain = int(
            cfg.get("domain_training", {}).get("target_tokens_per_domain", 5_000_000)
        )

    def crawl_domain(self, domain_key: str) -> DomainDataset:
        domain = DOMAINS[domain_key]
        dataset = DomainDataset(domain=domain_key)
        try:
            from app.core.web_intelligence import WebIntelligence

            wi = WebIntelligence()
            for query in domain["search_queries"]:
                result = wi.search(query)
                for chunk in result.get("answer_chunks", []):
                    text = chunk.get("text", "").strip()
                    if len(text) < 100:
                        continue
                    dataset.add(
                        instruction=f"{query}",
                        response=text,
                        source=chunk.get("url", ""),
                    )
                if dataset.token_count >= self.target_per_domain:
                    break
        except Exception as exc:
            logger.error("Domain crawl failed for %s: %s", domain_key, exc)
        return dataset

    def crawl_all(self) -> dict[str, DomainDataset]:
        datasets = {}
        for key in DOMAINS:
            datasets[key] = self.crawl_domain(key)
            self._save_domain(datasets[key])
        return datasets

    def _save_domain(self, dataset: DomainDataset) -> None:
        out = self.output_dir / f"{dataset.domain}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for sample in dataset.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def build_mixed_dataset(self, output_path: str = "data/domain_training/mixed.jsonl") -> str:
        import random

        all_samples: list[dict] = []
        for key, domain in DOMAINS.items():
            domain_file = self.output_dir / f"{key}.jsonl"
            if not domain_file.exists():
                continue
            with open(domain_file, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
            weight = domain.get("weight", 1.0)
            repeated = samples * int(weight)
            if weight % 1 > 0 and samples:
                repeated += random.sample(samples, int(len(samples) * (weight % 1)))
            all_samples.extend(repeated)
        random.shuffle(all_samples)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return str(out)


class DomainFineTuner:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        dt = cfg.get("domain_training", {})
        self.epochs = int(dt.get("finetune_epochs", 3))
        self.lr = float(dt.get("finetune_lr", 5e-5))
        self.batch_size = int(dt.get("finetune_batch_size", 2))
        self.grad_accum = int(dt.get("grad_accum", 16))
        self.max_len = int(dt.get("max_seq_len", 2048))
        self.output_dir = Path(dt.get("output_dir", "models/domain_finetuned"))

    def finetune(self, dataset_path: str) -> None:
        from model.train import train

        train(
            data_path=dataset_path,
            tokeniser_path="models/tokeniser/tokeniser.json",
            output_dir=str(self.output_dir),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            grad_accum=self.grad_accum,
            max_len=self.max_len,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", action="store_true")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--dataset", default="data/domain_training/mixed.jsonl")
    args = parser.parse_args()

    if args.crawl:
        crawler = DomainCrawler()
        if args.domain == "all":
            crawler.crawl_all()
        else:
            ds = crawler.crawl_domain(args.domain)
            crawler._save_domain(ds)

    if args.mix:
        DomainCrawler().build_mixed_dataset()

    if args.finetune:
        DomainFineTuner().finetune(args.dataset)
