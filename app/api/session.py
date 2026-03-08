from fastapi import APIRouter,Depends,Path
from sqlmodel import Session
import logging 

from ..schemas.response_model import Message,Session
from ..service.DB_service import db_service
from ..db import get_session

router = APIRouter(prefix="/session",tags=["Session"])
logger = logging.getLogger(__name__)

@router.get("/list",response_model=list[Session])
def get_list_session(db:Session = Depends(get_session)):
    logger.info("Get list session")
    return db_service.get_list_conversation(db)

@router.post("/update/{session_id}",response_model=Message)
def update_session_name(new_name:str, session_id:str=Path(), db:Session= Depends(get_session)):
    logger.info(f"Update session name for {session_id} ")
    db_service.update_session_name(session_id,new_name,db)
    logger.info("Update successfully")
    return Message("Update Successfully")

@router.delete("/delete/{session_id}",response_model=Message)
def delete_session(session_id:str=Path(), db:Session = Depends(get_session)):
    logger.info(f"Delete session {session_id}")
    db_service.delete_conversation(session_id,db)
    logger.info("Delete successfully")
    return Message("Delete successfully")

