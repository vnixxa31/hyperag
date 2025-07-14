import os

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# from llama_index.llms.openrouter import OpenRouter

# openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure global settings
Settings.llm = OpenAI(model="gpt-4o", api_key=openai_api_key)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", api_key=openai_api_key
)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.chunk_size = 512
