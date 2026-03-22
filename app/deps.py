import os
from langdetect import detect
import logging

from .core.env_config import settings 
from .service.File_service.FileFactory import FileProcessorRegistry
from .service.DB_service import db_service
logger =logging.getLogger(__name__)
def get_processor_from_file_type(file_id,db):
    file_type= db_service.get_file_type(file_id,db)
    if file_type is None:
        raise ValueError(f"File ID {file_id} not found in database")
    return FileProcessorRegistry.get_registry(extension=file_type)

def check_file_available(file_id,db):
    processor = get_processor_from_file_type(file_id,db)
    file_path = processor.get_file_path("upload",file_id)
    return os.path.isfile(file_path) 

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"
def check_session_id_available(session_id,db):
    return db_service.get_conversation(session_id, db) is not None 

def validate_file_size(file, max_file_size=settings.max_file_size):
# Advanced version 
    file.file.seek(0,2)
    actual_size=file.file.tell() # trả về int(số bytes)là vị trí con trỏ so với đầu file
    file.file.seek(0)
    return actual_size > max_file_size 
""" Cú pháp đầy đủ của phương thức này là: 
file.seek(offset, whence)
offset: Số lượng byte bạn muốn di chuyển con trỏ.
whence: Điểm mốc để bắt đầu di chuyển (mặc định là 0).
0: Tính từ đầu file.
1: Tính từ vị trí hiện tại của con trỏ.
2: Tính từ cuối file.
Initial version:
    file_size = 0
    for chunk in file.file:
        file_size += len(chunk)
        if file_size > max_file_size:
            file.file.seek(0)
            return True 
    file.file.seek(0)
    return False """
    
def validate_file_status(file_id,db):
    status = db_service.get_file_status(file_id,db)
    logger.info(f"Status state:{status}")
    return status == 'SUCCESS'