import os

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from agent.core.config import settings


def get_vector_store(collection_name: str = "default_collection") -> ChromaVectorStore:
    """
    Sets up a ChromaDB collection
    """
    db = chromadb.PersistentClient(path=settings.chroma_persist_path)

    chroma_collection = db.get_or_create_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store
