from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    app_env: str = "development"
    default_llm_model_name: str = "gpt-4o"
    default_embedding_model_name: str = "text-embedding-3-small"
    default_chunk_size: int = 1024
    default_chunk_overlap: int = 256
    chroma_persist_path: str = "./data/chroma_db"

    class Config:
        # Find the project root by looking for pyproject.toml
        _current_file = Path(__file__).resolve()
        _project_root = None
        for parent in _current_file.parents:
            if (parent / "pyproject.toml").exists():
                _project_root = parent
                break

        if _project_root is None:
            # Fallback: assume we're in src/agent/core and go up 3 levels
            _project_root = _current_file.parent.parent.parent

        env_file = _project_root / ".env"
        env_file_encoding = "utf-8"
        extra = (
            "ignore"  # Ignore extra fields from .env that aren't defined in the model
        )


settings = Settings()
