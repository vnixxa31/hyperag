from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the HyperAG application."""

    openai_api_key: str = ""
    openrouter_api_key: str = ""
    jina_api_key: str = ""

    default_openai_model: str = "gpt-4.1-nano"
    default_openrouter_model: str = "qwen/qwen3-14b:free"
    jina_embedding_model: str = "jina-embeddings-v4"

    chroma_persist_path: str = "./data/chroma_db"
    chroma_collection_name: str = "default"

    default_chunk_size: int = 512
    default_chunk_overlap: int = 128
    default_embed_batch_size: int = 16

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
