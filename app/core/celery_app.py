import eventlet
eventlet.monkey_patch()

from celery import Celery 
from sqlmodel import Session
from ..service.File_service.FileFactory import FileProcessorRegistry
from ..service.RAG_services.RAG_service import rag_service
from ..service.DB_service import db_service
from ..db import engine 
from .env_config import settings 
from ..service.File_service.Docling_service import docling_service

import logging

celery_app = Celery("myapp", broker=settings.redis,
                    backend=settings.redis)
celery_app.conf.update(task_track_started=True)



logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_pdf_file_background(self,file_id,file_name,strategy):
    with Session(engine) as db:
        try:
            self.update_state(state="STARTED")
            processor = FileProcessorRegistry.get_registry(file_name)
            index_file_path = processor.get_file_path("indexes",file_id)
            upload_file_path = processor.get_file_path("upload",file_id)

            logger.info(f"Start analyzing file {file_id}")
            is_complicated_file = FileProcessorRegistry.inspect_file_and_routing(file_id,processor) #return True/False 
            logger.info(f"Analyzing successfully: Class:{is_complicated_file}")
                
            file = (docling_service.convert_docling_to_list_document(upload_file_path) if (is_complicated_file and strategy=="semantic")
                    else (processor.process_file(file_id) if not is_complicated_file else None)) #List[Document]
            logger.info('Load file successfully')
            
            rag_service.parse_file_and_save_FAISS(file,is_complicated_file,file_id,upload_file_path,index_file_path,strategy,db)
            logger.info(f"FAISS index created for file {file_id}")

            db_service.update_file_status(file_id,"SUCCESS",db)
            logger.info(f"Update file status successfully! State: SUCCESS")
            
            return {"file_id": file_id, "status": "SUCCESS"}
        except Exception as e:
            db_service.update_file_status(file_id,"FAILURE",db)
            logger.error(f"Error processing file {file_id}: {str(e)}")
            raise self.retry(exc=e,countdown=5,max_retries=1) #raise retry -> state = 'RETRY', sau 10s worker chạy lại task này 

#Tắt các log không cần thiết 
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
