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
    print(f"[DEBUG] Test working directory: {os.getcwd()}")
    print(f"[DEBUG] Test .env file exists: {os.path.exists('.env')}")
    print(
        f"[DEBUG] Test OPENAI_API_KEY in os.environ: {'OPENAI_API_KEY' in os.environ}"
    )
    print(f"[DEBUG] Test settings.openai_api_key: {bool(settings.openai_api_key)}")
    print(f"[DEBUG] Test APP_ENV: {os.environ.get('APP_ENV', 'NOT_SET')}")

    LlamaIndexSettings.llm = OpenAI(
        model="gpt-4.1-mini", api_key=settings.openai_api_key
    )
    LlamaIndexSettings.embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small", api_key=settings.openai_api_key
    )
    LlamaIndexSettings.chunk_size = 1024
    LlamaIndexSettings.chunk_overlap = 256

    yield

    # Teardown code could go here if needed (e.g., cleaning up a test database)
    print("\n--- (Tearing down Test Environment: Fixture finished) ---")
