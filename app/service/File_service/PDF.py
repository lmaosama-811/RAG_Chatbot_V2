import os 
import uuid
from fastapi import HTTPException
import fitz 
from langchain_core.documents import Document as LCDocument

from .base import FileProcessor
from .FileFactory import FileProcessorRegistry

@FileProcessorRegistry.register(".pdf")
class PDFProcessor(FileProcessor):
    def __init__(self):
        self.upload_dir = "data/upload/pdf"
        self.indexes_dir = "data/indexes/pdf"
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
    
    def get_file(self,file_id,upload_file_path=None):
        upload_file_path = (self.get_file_path("upload",file_id) if upload_file_path is None else upload_file_path)
        return fitz.open(upload_file_path) #fitz.Document
        
    def process_file(self,file_id,upload_file_path=None):
        try:
            pdf = self.get_file(file_id,upload_file_path)
            raw_name = (os.path.basename(self.get_file_path("upload",file_id)) if upload_file_path is None else os.path.basename(upload_file_path))
            file_name = raw_name.replace(f"{file_id}_", "", 1)
            total_page = pdf.page_count
            list_LCDocuments = []
            for page_num in range(total_page):
                page = pdf.load_page(page_num)
                list_LCDocuments.append(LCDocument(page_content=page.get_text(),
                                                   metadata={"title":pdf.metadata.get("title"),
                                                             "file_name":file_name,
                                                             "file_id":file_id,
                                                             "page":page_num+1,
                                                             "total_page":total_page}))
            return list_LCDocuments # return List[Langchain Document]
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500,detail="Failed to process PDF file")
    def is_complicated_file(self,doc): #fitz.Document
        total_pages = doc.page_count 
        sample_pages = [0, total_pages // 2, total_pages - 1] if total_pages > 2 else range(total_pages) #page index in first,middle and last page 
        for p_idx in sample_pages:
            page = doc[p_idx]
            
            # 1. Identify "Text Density": If huge page but too little words -> Scanned file or image 
            text = page.get_text().strip()
            if len(text) < 100: 
                return True # (OCR/Docling Required)
            
            # 2. Examine "Vector Complexity": Count lines and rectangles. If > 50 lines/pages => May be complicated tables
            drawings = page.get_drawings()
            if len(drawings) > 50:
                return True # (High Table Density)
                
            # 3. Examine "Image Overlap": PDF PDF có nhiều ảnh đè lên nhau
            if len(page.get_images()) > 5:
                return True # (Image-heavy Layout)
        return False 

    def get_list_file(self):
        return [tuple(f.split("_", 1)) for f in os.listdir(self.upload_dir)]