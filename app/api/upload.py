from fastapi import APIRouter, UploadFile, File, Depends
from typing import Union, Literal 
from sqlmodel import Session
import logging 
import os 

import app.service.File_service
from ..schemas.response_model import Message, Error
from ..core.celery_app import process_upload_file_background, process_crawled_web_background
from ..service.File_service.FileFactory import FileProcessorRegistry
from ..deps import validate_file_size
from ..service.DB_service import db_service 
from ..core.env_config import settings 
from ..db import get_session

router = APIRouter(prefix ="/upload", tags=["Upload File"])
logger = logging.getLogger(__name__)

@router.post("", response_model=Union[Message,Error])
async def upload_file(file: UploadFile = File(), db: Session = Depends(get_session),strategy:str = settings.strategy):
    """Upload file. This chatbot only support PDF,DOCX,TXT file!\n
    Remark: Except hybrid,there's no contextual chunking support!"""
    logger.info("Upload request received")
    if validate_file_size(file):
        logger.warning("File is too large")
        return Error(code=413,error="File is too large")
    content = await file.read()
    processor = FileProcessorRegistry.get_registry(file.filename)
    file_id = processor.save_file(content, file.filename)
    db_service.create_file(db,file_id,file.filename,os.path.splitext(file.filename)[1].lower())
    background_task = process_upload_file_background.delay(file_id,file.filename,strategy)
    return Message(message =f"You have upload file {file.filename} successfully! File ID: {file_id}, status = {background_task.state}") 

@router.get("/list")
def get_files(type_of_file: Literal[".docx",".pdf",".txt","rest"],db:Session = Depends(get_session)):
    """Check list of file that you have uploaded"""
    return db_service.get_list_file(type_of_file,db)

@router.post("/crawl-urls",response_model = Message)
async def trigger_crawl_web(urls: list[str],strategy:str = settings.strategy):
    for url in urls:
        background_task = process_crawled_web_background.delay(url,strategy)
        logger.info(f"Triggered crawl for url {url}, status = {background_task.state}")
    return Message(message =f"You have triggered crawl for {len(urls)} urls successfully!")
