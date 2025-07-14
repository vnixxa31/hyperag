import os

import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_storage_context() -> StorageContext:
    """
    Initializes and returns the storage context for the application.
    This function encapsulates the logic for setting up the vector store.
    """
    # Use temporary persistent storage for ChromaDB
    persist_dir = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db")
    db = chromadb.PersistentClient(path=persist_dir)

    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
    chroma_collection = db.get_or_create_collection(chroma_collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context
