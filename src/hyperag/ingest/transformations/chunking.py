from llama_index.core.node_parser import SentenceSplitter

from hyperag.core.config import settings


def simple_chunker():
    text_splitter = SentenceSplitter(
        separator=" ",
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
    )

    return text_splitter
