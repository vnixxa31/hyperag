from llama_index.embeddings.jinaai import JinaEmbedding

from hyperag.core.config import settings


def jina_embed():
    embedding = JinaEmbedding(
        api_key=settings.jina_api_key,
        model=settings.jina_embedding_model,
        embed_batch_size=settings.default_embed_batch_size,
        task="retrieval.passage",
    )

    return embedding
