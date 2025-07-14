import os

import pytest
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

os.environ["APP_ENV"] = "testing"
from agent.core.config import settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    print("\n--- (Setting up Test Environment: Fixture running) ---")

    # TODO: Don't hardcode values
    LlamaIndexSettings.llm = OpenAI(
        model="gpt-4.1-mini", api_key=settings.openai_api_key
    )
    LlamaIndexSettings.embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small", api_key=settings.openai_api_key
    )
    LlamaIndexSettings.chunk_size = 1024
    LlamaIndexSettings.chunk_overlap = 256

    yield

    # TODO: Add actual teardown logic
    print("\n--- (Tearing down Test Environment: Fixture finished) ---")
