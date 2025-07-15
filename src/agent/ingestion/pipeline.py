from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from prefect import flow, task

from agent.core.config import settings
from agent.core.storage import get_vector_store


@task
def create_llm(model: str, api_key: str) -> OpenAI:
    """
    Creates an OpenAI LLM instance with the specified model and API key.
    """
    return OpenAI(model=model, api_key=api_key)


@task
def create_text_splitter(chunk_size: int, chunk_overlap: int) -> TokenTextSplitter:
    """
    Creates a TokenTextSplitter with the specified chunk size and overlap.
    """
    return TokenTextSplitter(
        separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


@task
def create_title_extractor(llm: OpenAI, nodes: int) -> TitleExtractor:
    """
    Creates a TitleExtractor with the specified LLM and number of nodes.
    """
    return TitleExtractor(nodes=nodes, llm=llm)


@task
def create_qa_extractor(llm: OpenAI, questions: int) -> QuestionsAnsweredExtractor:
    """
    Creates a QuestionsAnsweredExtractor with the specified LLM and number of questions.
    """
    return QuestionsAnsweredExtractor(questions=questions, llm=llm)


@flow(name="LlamaIndex Ingestion Pipeline", log_prints=True)
def create_ingestion_pipeline(
    model: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    title_extractor_nodes: int = 5,
    qa_extractor_questions: int = 3,
    collection_name: str = "default_collection",
) -> IngestionPipeline:
    """
    Creates a simple ingestion pipeline with token text splitter,
    title extractor, and questions answered extractor.
    """

    extraction_llm = create_llm(model=model, api_key=settings.openai_api_key)

    text_splitter = create_text_splitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    title_extractor = create_title_extractor(
        llm=extraction_llm, nodes=title_extractor_nodes
    )

    qa_extractor = create_qa_extractor(
        llm=extraction_llm, questions=qa_extractor_questions
    )

    vector_store = get_vector_store(collection_name=collection_name)

    return IngestionPipeline(
        transformations=[text_splitter, title_extractor, qa_extractor],
        vector_store=vector_store,
    )
