import os

import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_storage_context(collection_name: str = "default_collection") -> StorageContext:
    """
    Initializes and returns the storage context for the application.
    This function encapsulates the logic for setting up the vector store.
    """
    # Use temporary persistent storage solution for ChromaDB
    persist_directory = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db")
    db = chromadb.PersistentClient(path=persist_directory)

    chroma_collection = db.get_or_create_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context
