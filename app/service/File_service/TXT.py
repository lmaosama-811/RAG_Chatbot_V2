import os 
import uuid
from langchain_community.document_loaders import TextLoader
from fastapi import HTTPException

from .base import FileProcessor
from .Factory import FileProcessorRegistry

@FileProcessorRegistry.register(".txt")
class TXTProcessor(FileProcessor):
    def __init__(self):
        self.upload_dir = "data/upload/text"
        self.indexes_dir = "data/indexes/text"
        os.makedirs(self.upload_dir,exist_ok=True)
        os.makedirs(self.indexes_dir,exist_ok = True)

    def save_file(self,file_bytes, file_name):
        file_id = uuid.uuid4().hex #.uuid4() create random UUID, .hex() convert it into 32-character string 
        safe_name = os.path.basename(file_name)
        new_filename = f"{file_id}_{safe_name}"
        file_path = os.path.join(self.upload_dir,new_filename)
        with open(file_path,"wb") as f:
            f.write(file_bytes)
        return file_id 
    
    def get_file_path(self,folder,file_id):
        base_dir = self.upload_dir if folder == "upload" else self.indexes_dir
        files = [f for f in os.listdir(base_dir) if f.startswith(f"{file_id}_")]
        if not files:
            return os.path.join(base_dir, file_id)
        return os.path.join(base_dir, files[0]) 
    
    def get_file(self,file_id):
        loader = TextLoader(self.get_file_path("upload",file_id))
        return loader.load()
    
    def process_file(self,file_id):
        try:
            return self.get_file(file_id)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500,detail="Failed to process TXT file")
    
    def get_list_file(self):
        return [tuple(f.split("_", 1)) for f in os.listdir(self.upload_dir)]