from pydantic import BaseModel


class Settings(BaseModel):
    """Configuration settings for the HyperAG application."""

    openai_api_key: str = ""
    openrouter_api_key: str = ""
    jina_api_key: str = ""

    openai_model: str = "gpt-4.1-nano"
    openrouter_model: str = "qwen/qwen3-14b:free"
    jina_embedding_model: str = "jina-embeddings-v4"

    chroma_persist_path: str = "./data/chroma_db"
    chroma_collection_name: str = "default"

    default_chunk_size: int = 512
    default_chunk_overlap: int = 128
    default_embed_batch_size: int = 16


settings = Settings()
