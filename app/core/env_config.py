from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_key:str
    database_url:str
    temperature: float
    top_k: int
    max_file_size:int
    strategy:str
    redis:str 
    bm25_w: float
    faiss_w: float 
    min_relevant_docs: int
    relevance_threshold: float
    confidence: float
    max_retry: int
    confidence_threshold: float
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    ) #Pydantic v2

settings = Settings()
