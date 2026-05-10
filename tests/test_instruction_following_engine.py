from instruction_following_engine import InstructionFollowingEngine, ResponseGenerator


class FlakyGenerator(ResponseGenerator):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, requirements: list[str]) -> str:
        self.calls += 1
        if self.calls == 1:
            return "Incomplete answer"
        return super().generate(prompt, requirements)


def test_instruction_engine_parses_extracts_and_validates_requirements():
    engine = InstructionFollowingEngine(generator=ResponseGenerator())
    prompt = """Build response\n- include api testing\n- provide summary\n"""

    requirements = engine.extract_requirements(prompt)
    assert "include api testing" in [r.lower() for r in requirements]

    result = engine.run(prompt)
    assert result.compliant is True
    assert result.attempts >= 1


def test_instruction_engine_regenerates_on_mismatch():
    engine = InstructionFollowingEngine(generator=FlakyGenerator(), max_attempts=3)
    prompt = """Task:\n- include security policy\n- output json format\n"""

    result = engine.run(prompt)
    assert result.compliant is True
    assert result.attempts == 2
    assert result.missing_requirements == []
