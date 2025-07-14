from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI

from agent.core.config import settings


def create_ingestion_pipeline() -> IngestionPipeline:
    """
    Creates and returns an simple ingestion pipeline using Title and QA extrators
    This pipeline is used to process and transform text data before storage.
    """
    extraction_llm = OpenAI(model="gpt-4.1-nano", api_key=settings.openai_api_key)

    text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=256)
    title_extractor = TitleExtractor(nodes=5, llm=extraction_llm)
    qa_extractor = QuestionsAnsweredExtractor(questions=3, llm=extraction_llm)

    return IngestionPipeline(
        transformations=[text_splitter, title_extractor, qa_extractor]
    )
