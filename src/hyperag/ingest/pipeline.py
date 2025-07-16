from llama_index.core.ingestion import IngestionPipeline

from hyperag.core.storage import get_vector_store
from hyperag.ingest.transformations.chunking import simple_chunker
from hyperag.ingest.transformations.embedding import jina_embed


def create_metadata_pipeline() -> IngestionPipeline:
    ingest_pipeline = IngestionPipeline(
        transformations=[
            simple_chunker(),
            jina_embed(),
        ],
        vector_store=get_vector_store(),
    )

    return ingest_pipeline
