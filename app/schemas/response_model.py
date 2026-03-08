from pydantic import BaseModel, Field

class Message(BaseModel):
    message:str

class Error(BaseModel):
    code: int
    error: str

class ChatBotResponse(BaseModel):
    model_name: str
    session_id: str
    session_name: str
    answer: str

class Session(BaseModel):
    session_id: str 
    session_name: str 