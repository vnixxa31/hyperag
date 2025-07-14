from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from agent.core.config import settings


def configure_global_dependencies():
    """
    Reads from the application config and sets up the global
    LlamaIndex settings context. This should be run once on app startup.
    """
    print(f"--- Configuring LlamaIndex Settings for ENV: {settings.app_env} ---")

    llm_instance = OpenAI(
        model=settings.default_llm_model_name,  # e.g., "gpt-4o"
        api_key=settings.openai_api_key,
    )

    embed_model_instance = OpenAIEmbedding(
        model_name=settings.default_embedding_model_name,
        api_key=settings.openai_api_key,
    )

    LlamaIndexSettings.llm = llm_instance
    LlamaIndexSettings.embed_model = embed_model_instance
    LlamaIndexSettings.chunk_size = settings.default_chunk_size
    LlamaIndexSettings.chunk_overlap = settings.default_chunk_overlap

    print("--- LlamaIndex Settings configured successfully. ---")
