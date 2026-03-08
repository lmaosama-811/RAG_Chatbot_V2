from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_key:str
    database_url:str
    temperature: float
    top_k: int
    max_file_size:int 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    ) #Pydantic v2

settings = Settings()
