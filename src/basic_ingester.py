import os

import chromadb
import dotenv
import openai
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

dotenv.load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

chroma_client = chromadb.PersistentClient("./data/chroma_db")
chroma_collection = chroma_client.create_collection("test_v1")

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print(response)
