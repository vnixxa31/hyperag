from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# agent-generated function
def _find_project_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return current_file.parent.parent.parent


class Settings(BaseSettings):
    app_env: str = "development"
    chroma_persist_path: str = "./data/chroma_db"
    collection_name: str = "default_collection"
    default_openai_model: str = "gpt-4o"
    title_extractor_nodes: int = 5
    qa_extractor_questions: int = 3

    model_config = SettingsConfigDict(
        env_file=_find_project_root() / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
