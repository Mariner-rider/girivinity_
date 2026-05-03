from app.core.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.app_name == "Girivinity"
    assert settings.model_load_in_4bit is True
