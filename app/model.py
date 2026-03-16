from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from .core.env_config import settings

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key= settings.api_key,
    base_url="https://openrouter.ai/api/v1"
)

llm = ChatOpenAI(
    model="openrouter/free",
    api_key= settings.api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=settings.temperature,
    max_tokens=1000,
    streaming=True #turn on streaming
)