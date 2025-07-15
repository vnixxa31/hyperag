from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI

from core.storage import get_vector_store


def create_metadata_pipeline() -> IngestionPipeline:
    """
    Creates a simple ingestion pipeline with token text splitter,
    title extractor, and questions answered extractor.
    """
    llm = OpenAI(mode="gpt-4.1-nano")

    text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=256)

    title_extractor = TitleExtractor(nodes=5, llm=llm)

    qa_extractor = QuestionsAnsweredExtractor(questions=3, llm=llm)

    return IngestionPipeline(
        transformations=[text_splitter, title_extractor, qa_extractor],
        vector_store=get_vector_store(),
    )
