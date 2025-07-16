from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

from hyperag.core.config import settings
from hyperag.core.storage import get_vector_store
from hyperag.ingest.transformations.embedding import JinaEmbedding


def test_query():
    vector_store = get_vector_store()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=JinaEmbedding(
            model=settings.jina_embedding_model, api_key=settings.jina_api_key
        ),
    )

    query_engine = index.as_query_engine(
        llm=OpenAI(api_key=settings.openai_api_key, model=settings.default_openai_model)
    )

    print("What did the author do growing up?")
    print(query_engine.query("What did the author do growing up?"))


if __name__ == "__main__":
    test_query()
