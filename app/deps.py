import os

from .model import lang_model
from .core.env_config import settings 
from .service.File_service.Factory import FileProcessorRegistry
from .service.DB_service import db_service

def get_processor_from_file_type(file_id,db):
    file_type= db_service.get_file_type(file_id,db)
    return FileProcessorRegistry.get_registry(extension=file_type)

def check_file_available(file_id,db):
    processor = get_processor_from_file_type(file_id,db)
    file_path = processor.get_file_path("upload",file_id)
    return os.path.isfile(file_path) 

def check_file_type(file):
    ext = os.path.splitext(file.filename)[1].lower()
    return ext in FileProcessorRegistry.registry

def detect_language(text: str):
    if not text.strip():
        return "unknown"
    label, prob = lang_model.predict(text)
    lang = label[0].replace("__label__", "")
    return lang

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
    return status == "SUCCESS"