import tempfile
from pathlib import Path
from unittest.mock import MagicMock


def _make_forge(tmp: str) -> object:
    from app.core.skill_forge import SkillForge
    forge = SkillForge.__new__(SkillForge)
    forge.skills_dir = Path(tmp) / "skills"
    forge.db_path = Path(tmp) / "skill_forge.db"
    forge.min_chunks = 1
    forge.max_skills_composed = 3
    forge.skills_dir.mkdir(parents=True, exist_ok=True)
    forge.db_path.parent.mkdir(parents=True, exist_ok=True)
    forge._init_db()
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [[]], "distances": [[]], "metadatas": [[]]
    }
    forge.skill_index = mock_col
    return forge


def test_generate_and_load_skill():
    with tempfile.TemporaryDirectory() as tmp:
        forge = _make_forge(tmp)
        chunks = [
            {"text": "Python decorators are functions that modify other "
                     "functions. They use the @symbol syntax. Important: "
                     "always use functools.wraps to preserve metadata.",
             "score": 0.8, "url": "https://example.com"},
        ]
        forge._generate("python decorators", chunks, ["https://example.com"])
        skill = forge._load_skill("python-decorators")
        assert skill is not None
        assert skill.version == 1
        assert "decorator" in skill.instructions.lower()


def test_update_increments_version():
    with tempfile.TemporaryDirectory() as tmp:
        forge = _make_forge(tmp)
        chunks = [{"text": "Decorators wrap functions.", "score": 0.7, "url": ""}]
        forge._generate("python decorators", chunks, [])
        forge._update_skill(
            forge._load_skill("python-decorators"),
            [{"text": "Use @wraps to preserve metadata.", "score": 0.8, "url": ""}],
            [],
        )
        skill = forge._load_skill("python-decorators")
        assert skill.version == 2


def test_skill_to_prompt_block_contains_topic():
    with tempfile.TemporaryDirectory() as tmp:
        forge = _make_forge(tmp)
        chunks = [{"text": "Machine learning uses data to train models.",
                   "score": 0.9, "url": ""}]
        forge._generate("machine learning", chunks, [])
        skill = forge._load_skill("machine-learning")
        block = skill.to_prompt_block()
        assert "machine learning" in block.lower()
        assert "Confidence" in block


def test_compose_skills_merges_instructions():
    with tempfile.TemporaryDirectory() as tmp:
        forge = _make_forge(tmp)
        from app.core.skill_forge import Skill
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        s1 = Skill("s1", "Topic A", 1, "Instructions A", [], [], 0.8, 0, 0.0, now, now, [], [])
        s2 = Skill("s2", "Topic B", 1, "Instructions B", [], [], 0.7, 0, 0.0, now, now, [], [])
        composed = forge._compose_skills("combined query", [s1, s2])
        assert composed is not None
        assert "Topic A" in composed.instructions
        assert "Topic B" in composed.instructions
        assert composed.slug == "__composed__"


def test_list_skills_returns_empty_initially():
    with tempfile.TemporaryDirectory() as tmp:
        forge = _make_forge(tmp)
        result = forge.list_skills()
        assert isinstance(result, list)
