from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.llms.openai import OpenAI
from prefect import task


@task
def create_metadata_pipeline(
    llm: OpenAI,
    vector_store: BasePydanticVectorStore,
    chunk_size: int,
    chunk_overlap: int,
    title_extractor_nodes: int,
    qa_extractor_questions: int,
) -> IngestionPipeline:
    """
    Creates a simple ingestion pipeline with token text splitter,
    title extractor, and questions answered extractor.
    """
    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    title_extractor = TitleExtractor(nodes=title_extractor_nodes, llm=llm)

    qa_extractor = QuestionsAnsweredExtractor(questions=qa_extractor_questions, llm=llm)

    return IngestionPipeline(
        transformations=[text_splitter, title_extractor, qa_extractor],
        vector_store=vector_store,
    )
