import os 
import uuid
from langchain_community.document_loaders import PyPDFLoader
from fastapi import HTTPException
from pypdf import PdfReader

from .base import FileProcessor
from .Factory import FileProcessorRegistry

@FileProcessorRegistry.register(".pdf")
class PDFProcessor(FileProcessor):
    def __init__(self):
        self.upload_dir = "data/upload/pdf"
        self.indexes_dir = "data/indexes/pdf"
        os.makedirs(self.upload_dir,exist_ok=True)
        os.makedirs(self.indexes_dir,exist_ok = True)

    def is_scanned_pdf(self, file_path):
        reader = PdfReader(file_path)

        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                return False
        return True

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
        file_path = self.get_file_path("upload",file_id)
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def process_file(self,file_id):
        try:
            return self.get_file(file_id)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500,detail="Failed to process PDF file")
    
    def get_list_file(self):
        return [tuple(f.split("_", 1)) for f in os.listdir(self.upload_dir)]