from pathlib import Path


REQUIRED_MODULE_DIRS = [
    "core",
    "llm",
    "memory",
    "agents",
    "crawler",
    "rag",
    "security",
    "analytics",
    "multimodal",
    "training",
]


def test_required_module_directories_exist():
    for module in REQUIRED_MODULE_DIRS:
        assert (Path("app") / module).is_dir()


def test_docker_and_requirements_files_exist():
    assert Path("requirements.txt").is_file()
    assert Path("Dockerfile").is_file()
    assert Path("config.yaml").is_file()
    assert Path("app/main.py").is_file()
