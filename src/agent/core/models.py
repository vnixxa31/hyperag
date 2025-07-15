from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from prefect import task

from agent.core.storage import settings


@task
def get_openai_model(model: str = "") -> OpenAI:
    if model:
        return OpenAI(model=model, api_key=settings.openai_api_key)
    else:
        return OpenAI(api_key=settings.openai_api_key)


@task
def get_openrouter_model(model: str = "") -> OpenRouter:
    if model:
        return OpenRouter(model=model, api_key=settings.openrouter_api_key)
    else:
        return OpenRouter(api_key=settings.openrouter_api_key)
