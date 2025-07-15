import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from prefect import task

from agent.core.config import settings


# TODO: Add caching to vector store
@task(
    retries=3,
    retry_delay_seconds=10,
)
def get_vector_store(
    collection_name: str = "default_collection",
) -> ChromaVectorStore:
    """
    Sets up ChromaDB vector store using a persistent client and gets or
    creates a collection.
    """
    db = chromadb.PersistentClient(path=settings.chroma_persist_path)

    chroma_collection = db.get_or_create_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store
