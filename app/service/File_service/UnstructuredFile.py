import os
import uuid
import logging

from langchain_community.document_loaders import UnstructuredFileLoader
from .base import FileProcessor
from .FileFactory import FileProcessorRegistry

logger = logging.getLogger(__name__)

@FileProcessorRegistry.register("rest")
class UnstructuredFile(FileProcessor):
    def __init__(self):
        self.upload_dir = "data/upload/unstructured"
        self.indexes_dir = "data/indexes/unstructured"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.indexes_dir, exist_ok=True)

    def get_file_path(self, folder, file_id):
        base_dir = self.upload_dir if folder == "upload" else self.indexes_dir
        if not os.path.exists(base_dir):
            return os.path.join(base_dir, file_id)
            
        files = [f for f in os.listdir(base_dir) if f.startswith(f"{file_id}_")]
        if not files:
            return os.path.join(base_dir, file_id)
        return os.path.join(base_dir, files[0]) 

    def get_file(self, file_id, upload_file_path=None):
        upload_file_path = (self.get_file_path("upload",file_id) if upload_file_path is None else upload_file_path)
        
        # initialize Langchain Loader to wrap unstructured
        loader = UnstructuredFileLoader(
            file_path=upload_file_path,
            mode="elements", # Split into separate text, table, title chunks
            strategy="fast"  # recommend "fast" to avoid installing Poppler/Tesseract
        )
        return loader

    def process_file(self, file_id, upload_file_path=None):
        try:
            loader = self.get_file(file_id, upload_file_path)
            
            # load() will automatically detect file extension (.csv, .pptx, .xlsx, .eml...) and use the corresponding parser
            # Return List[Document] directly
            list_LCDocument = loader.load()
            
            # Add file_id and source to Metadata for synchronization with other Processors
            for doc in list_LCDocument:
                doc.metadata["file_id"] = file_id
                
            return list_LCDocument
            
        except Exception as e:
            logger.error(f"False in parse Unstructured File ID {file_id}: {str(e)}")
            raise e

    def is_complicated_file(self, doc):
        # KHÔNG NÊN giao mảng "Tạp nham" cho Docling vì Docling chuyên trị PDF, Word. Bắt bọn chúng tự xử bằng Unstructured.
        return False

    def save_file(self, file_bytes, file_name):
        file_id = str(uuid.uuid4())[:8]
        new_filename = f"{file_id}_{file_name}"    
        file_path = os.path.join(self.upload_dir, new_filename)

        with open(file_path, "wb") as f:
            f.write(file_bytes)
        return file_id

    def get_list_file(self):
        if not os.path.exists(self.upload_dir):
            return []
        files = os.listdir(self.upload_dir)
        return [{"file_id": f.split("_", 1)[0], "file_name": f.split("_", 1)[1]} for f in files if "_" in f]
