import uuid

from llama_index.core import SimpleDirectoryReader

from agent.core.config import settings
from agent.ingestion.pipeline import create_ingestion_pipeline


def test_ingest_basic_documents():
    documents = SimpleDirectoryReader("./data/paul_graham").load_data()

    pipeline = create_ingestion_pipeline(
        model=settings.test_model_name,
        chunk_size=1024,
        chunk_overlap=256,
        title_extractor_nodes=5,
        qa_extractor_questions=3,
        collection_name=uuid.uuid4().hex,
    )
    nodes = pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True,
    )

    assert len(nodes) > 0, "Pipeline did not produce any nodes."
    assert all("document_title" in node.metadata for node in nodes), (
        "Not all nodes have titles in metadata."
    )
    assert all(
        "questions_this_excerpt_can_answer" in node.metadata for node in nodes
    ), "Not all nodes have answered questions in metadata."
