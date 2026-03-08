from fastapi import APIRouter, UploadFile, File, Depends
from typing import Union
from sqlmodel import Session
import logging 
import os 

import app.service.File_service
from ..schemas.response_model import Message, Error
from ..core.celery_app import process_pdf_file_background
from ..service.File_service.Factory import FileProcessorRegistry
from ..deps import check_file_type, validate_file_size
from ..service.DB_service import db_service 
from ..db import get_session

router = APIRouter(prefix ="/upload", tags=["Upload File"])
logger = logging.getLogger(__name__)

@router.post("", response_model=Union[Message,Error])
async def upload_file(file: UploadFile = File(), db: Session = Depends(get_session)):
    """Upload file. This chatbot only support PDF,DOCX,TXT file!"""
    logger.info("Upload request received")
    if not check_file_type(file):
        logger.warning("Invalid file upload attempt")
        logger.info(f"Extension: {os.path.splitext(file.filename)[1].lower()}")
        logger.info(f"Registry: {FileProcessorRegistry.registry}")
        return Error(code=400,error=f"Invalid file. This chatbot only support PDF,DOCX,TXT file!")
    if validate_file_size(file):
        logger.warning("File is too large")
        return Error(code=413,error="File is too large")
    content = await file.read()
    processor = FileProcessorRegistry.get_registry(file.filename)
    file_id = processor.save_file(content, file.filename)
    db_service.create_file(db,file_id,os.path.splitext(file.filename)[1].lower())
    background_task = process_pdf_file_background.delay(file_id,file.filename)
    return Message(message =f"You have upload file {file.filename} successfully! File ID: {file_id}, status = {background_task.state}") 

@router.get("/list")
def get_files(): #sửa để thấy list file 
    """Check list of file that you have uploaded"""
    return 
