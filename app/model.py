from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from .core.env_config import settings

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key= settings.api_key,
    base_url="https://openrouter.ai/api/v1"
)

llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1",  # paid model
    api_key= settings.api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=settings.temperature,
    max_tokens=32000,
    request_timeout=60,   # 120s -- RAGAs calls co the cham
    streaming=False         # turn on streaming
)
