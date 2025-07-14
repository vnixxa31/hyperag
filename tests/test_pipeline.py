import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llama_index.core import SimpleDirectoryReader

from agent.ingestion.pipeline import create_ingestion_pipeline


def test_ingestion_pipeline():
    """
    Test the ingestion pipeline to ensure it processes documents correctly.
    This test checks if the pipeline can run without errors and produces expected nodes.
    """
    documents = SimpleDirectoryReader("./data/paul_graham").load_data()

    pipeline = create_ingestion_pipeline()

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
