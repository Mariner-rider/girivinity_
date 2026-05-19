from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)
_ = yaml


@dataclass
class LearnerProfile:
    user_id: str
    subject: str
    level: str
    learning_goal: str
    preferred_style: str
    pace: str


@dataclass
class TeachingPlan:
    subject: str
    topic: str
    learner_level: str
    steps: list[str]
    examples: list[str]
    exercises: list[str]
    next_topic: str
    system_prompt_injection: str


TEACHING_TRIGGERS = [
    "teach me", "explain to me", "how do i learn",
    "i want to learn", "help me understand",
    "what is", "how does", "why does",
    "i am a beginner", "i am new to",
    "can you tutor", "be my teacher",
    "step by step", "from scratch",
    "मुझे सिखाओ", "समझाओ", "सीखना है",
]

LEVEL_SIGNALS = {
    "absolute_beginner": [
        "never heard", "no idea", "what is", "basics",
        "just started", "beginner", "newbie", "new to",
        "don't know anything", "zero knowledge",
    ],
    "beginner": [
        "little bit", "some idea", "heard about",
        "basic understanding", "just learning",
    ],
    "intermediate": [
        "know the basics", "been doing for a while",
        "understand fundamentals", "working on",
    ],
    "advanced": [
        "experienced", "years of", "deep understanding",
        "working professionally",
    ],
    "expert": [
        "expert", "mastery", "deep dive", "advanced topics",
        "nuanced", "edge cases",
    ],
}

STYLE_SIGNALS = {
    "example": ["example", "show me", "demonstrate", "code"],
    "theory": ["theory", "concept", "why", "how works"],
    "practice": ["practice", "exercise", "try", "hands-on"],
    "visual": ["diagram", "picture", "visualise", "chart"],
}


class TeachingEngine:
    def is_teaching_request(self, query: str) -> bool:
        q = query.lower()
        return any(trigger in q for trigger in TEACHING_TRIGGERS)

    def build_learner_profile(self, query: str, user_id: str) -> LearnerProfile:
        return LearnerProfile(
            user_id=user_id,
            subject=self._detect_subject(query),
            level=self._detect_level(query),
            learning_goal=self._extract_goal(query),
            preferred_style=self._detect_style(query),
            pace=self._detect_pace(query),
        )

    def build_teaching_plan(self, profile: LearnerProfile, context: str) -> TeachingPlan:
        return TeachingPlan(
            subject=profile.subject,
            topic=profile.learning_goal,
            learner_level=profile.level,
            steps=self._build_steps(profile, context),
            examples=self._build_examples(profile),
            exercises=self._build_exercises(profile),
            next_topic=self._suggest_next(profile),
            system_prompt_injection=self._build_system_injection(profile),
        )

    def get_prompt_injection(self, query: str, user_id: str) -> str:
        if not self.is_teaching_request(query):
            return ""
        profile = self.build_learner_profile(query, user_id)
        plan = self.build_teaching_plan(profile, "")
        return plan.system_prompt_injection

    def _detect_subject(self, query: str) -> str:
        subject_map = {
            "python": "Python Programming",
            "coding": "Programming",
            "javascript": "JavaScript",
            "law": "Law",
            "ipc": "Indian Penal Code",
            "bns": "Bharatiya Nyaya Sanhita",
            "math": "Mathematics",
            "physics": "Physics",
            "chemistry": "Chemistry",
            "biology": "Biology",
            "history": "History",
            "economics": "Economics",
            "accounting": "Accounting",
            "ca ": "Chartered Accountancy",
            "machine learning": "Machine Learning",
            "ai": "Artificial Intelligence",
            "cuda": "CUDA Programming",
            "surgery": "Surgery",
            "medicine": "Medicine",
            "3d": "3D Design",
            "blender": "Blender 3D",
        }
        q = query.lower()
        for keyword, subject in subject_map.items():
            if keyword in q:
                return subject
        return "the requested topic"

    def _detect_level(self, query: str) -> str:
        q = query.lower()
        for level, signals in LEVEL_SIGNALS.items():
            if any(sig in q for sig in signals):
                return level
        return "beginner"

    def _detect_style(self, query: str) -> str:
        q = query.lower()
        for style, signals in STYLE_SIGNALS.items():
            if any(sig in q for sig in signals):
                return style
        return "example"

    def _extract_goal(self, query: str) -> str:
        q = query.strip()
        for trigger in TEACHING_TRIGGERS:
            if trigger in q.lower():
                idx = q.lower().find(trigger) + len(trigger)
                remainder = q[idx:].strip()
                if len(remainder) > 3:
                    return remainder[:100]
        return q[:100]

    def _detect_pace(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["quickly", "fast", "brief", "summary"]):
            return "fast"
        if any(w in q for w in ["slowly", "detail", "thorough", "complete"]):
            return "slow"
        return "normal"

    def _build_steps(self, profile: LearnerProfile, context: str) -> list[str]:
        if profile.level == "absolute_beginner":
            return [
                f"Start with: What is {profile.subject}?",
                "Core concepts one at a time",
                "Simple real-world analogy",
                "First hands-on example",
                "Common beginner mistakes to avoid",
            ]
        if profile.level == "beginner":
            return [
                "Quick recap of what you already know",
                "Fill in the foundational gaps",
                "Build on basics with intermediate concepts",
                "Practical exercise",
            ]
        if profile.level == "intermediate":
            return [
                "Identify your specific knowledge gaps",
                "Deep dive into target concept",
                "Advanced examples and edge cases",
                "Best practices and real-world application",
            ]
        return [
            "Direct to advanced concepts",
            "Nuanced discussion of trade-offs",
            "Expert-level examples",
            "Further reading and research directions",
        ]

    def _build_examples(self, profile: LearnerProfile) -> list[str]:
        if profile.preferred_style == "example":
            return [
                f"Concrete example in {profile.subject}",
                "Step-by-step walkthrough",
                "Real-world application",
            ]
        return [f"Key example for {profile.subject}"]

    def _build_exercises(self, profile: LearnerProfile) -> list[str]:
        if profile.level in ("absolute_beginner", "beginner"):
            return ["Simple practice problem", "Fill-in-the-blank exercise"]
        return ["Applied problem", "Design/build challenge"]

    def _suggest_next(self, profile: LearnerProfile) -> str:
        next_map = {
            "Python Programming": "Object-Oriented Programming",
            "Mathematics": "Calculus",
            "Machine Learning": "Deep Learning",
            "Law": "Constitutional Law",
            "Accounting": "Financial Analysis",
            "CUDA Programming": "GPU Memory Optimization",
        }
        return next_map.get(profile.subject, "Advanced topics")

    def _build_system_injection(self, profile: LearnerProfile) -> str:
        level_instructions = {
            "absolute_beginner": (
                "This person is a complete beginner. Use simple words. "
                "No jargon without explanation. Use analogies. Be encouraging. "
                "Teach one concept at a time. Check understanding with a simple question at the end."
            ),
            "beginner": (
                "This person knows a little. Build on their foundation. "
                "Explain why, not just how. Use examples. Keep sentences short."
            ),
            "intermediate": (
                "This person has working knowledge. Skip basics, focus on depth. "
                "Discuss trade-offs and best practices. Use precise technical language."
            ),
            "advanced": (
                "Expert learner. Go deep immediately. Discuss nuances, edge cases, "
                "recent developments. Treat them as a peer."
            ),
            "expert": (
                "Peer-level discussion. Focus on cutting-edge, unsolved problems, "
                "research directions. Be precise and thorough."
            ),
        }
        style_additions = {
            "example": " Prioritise code examples and demonstrations.",
            "theory": " Explain the underlying theory and why it works.",
            "practice": " Give exercises after each concept.",
            "visual": " Describe diagrams and visual structures clearly.",
        }
        return (
            f"You are teaching {profile.subject} to this user. "
            f"{level_instructions.get(profile.level, '')} "
            f"{style_additions.get(profile.preferred_style, '')} "
            "After your response, ask one follow-up question to assess understanding "
            "and guide the next step."
        )
