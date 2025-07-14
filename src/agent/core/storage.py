from datetime import timedelta

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from prefect import task
from prefect.tasks import task_input_hash

from agent.core.config import settings


@task(
    retries=3,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def get_vector_store(collection_name: str = "default_collection") -> ChromaVectorStore:
    """
    Sets up and returns a ChromaDB vector store
    """
    db = chromadb.PersistentClient(path=settings.chroma_persist_path)

    chroma_collection = db.get_or_create_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store
