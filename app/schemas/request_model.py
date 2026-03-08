from pydantic import BaseModel
from sqlmodel import SQLModel, Field

class ChatbotRequest(BaseModel):
    session_id: str|None=None
    file_id: str
    question: str 

class ConversationHistory(SQLModel, table=True):
    __tablename__="conversation_history"
    id:int|None = Field(default=None, primary_key=True)
    session_id: str|None = None
    session_name: str|None = None
    role: str
    content: str 

class Summary(SQLModel,table=True):
    __tablename__="summary"
    id: int|None = Field(default=None,primary_key=True)
    session_id: str
    covered_until_message_id: int
    content: str 

class FileStatus(SQLModel,table=True):
    __tablename__="file_status"
    id: int=Field(default=None, primary_key=True)
    file_id:str
    type:str
    status:str 

class ParentStore(SQLModel,table=True):
    __tablename__="parent_store"
    parent_id: str = Field(default=None,primary_key=True)
    file_id: str
    context: str