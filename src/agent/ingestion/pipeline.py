from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI

from agent.core.config import settings
from agent.core.storage import get_vector_store


def create_ingestion_pipeline() -> IngestionPipeline:
    """
    Returns simple ingestion pipeline with token text splitter,
    title extractor, and questions answered extractor.
    """

    # NOTE: Should configuration be moved to settings or args?

    # TODO: Model should be configurable
    extraction_llm = OpenAI(model="gpt-4.1-nano", api_key=settings.openai_api_key)

    # TODO: Chunk size and overlap should be configurable
    text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=256)

    # TODO: Number of nodes and questions should be configurable
    title_extractor = TitleExtractor(nodes=5, llm=extraction_llm)
    qa_extractor = QuestionsAnsweredExtractor(questions=3, llm=extraction_llm)

    # TODO: Collection name should be configurable
    vector_store = get_vector_store("ingestion_collection_v1")

    return IngestionPipeline(
        transformations=[text_splitter, title_extractor, qa_extractor],
        vector_store=vector_store,
    )
