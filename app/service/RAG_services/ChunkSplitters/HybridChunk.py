from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import logging  
import uuid 

from ....model import embeddings 
from ...DB_service import db_service
from ...LLM_service import llm_service
from ....model import llm 
from .ChunkBase import ChunkBase
from .ChunkFactory import ChunkerRegistry 
from ...File_service.Docling_service import docling_service

logger =logging.getLogger(__name__)

@ChunkerRegistry.register('hybrid')
class HybridChunk(ChunkBase):
    def __init__(self):
        super().__init__()
        self.parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),("####", "Header 4")])
        self.sub_parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n","\n"," ",""])
        self.child_splitter = SemanticChunker(embeddings)
    def _build_parent_child_chunks(self,file_id,markdown_content,doc_title,llm,db):
        parent_chunks = self.parent_splitter.split_text(markdown_content) #List[Document]
        logger.info("Chunking file into parent chunks with markdown header splitter successfully")
        child_chunks = []

        logger.info("Starting chunking parent chunks")
        for j, parent_chunk in enumerate(parent_chunks):
            parent_id = str(uuid.uuid4()) 
            parent_chunk.metadata["title"] = doc_title
            parent_chunk.page_content = super().format_page_content_to_synthesize(parent_chunk)
            db_service.create_parent_chunk(parent_id,file_id,parent_chunk.page_content,db) #Save to database
            logger.info(f"Save no.{j+1} parent chunk to database successfully!")
            children = self.child_splitter.split_documents([parent_chunk]) #Convert into List[Document] as split_documents apply only for list
            logger.info(f"Split child chunks for no.{j+1} parent chunk successfully!")
            for i,child_chunk in enumerate(children): 
                child_chunk.metadata["parent_id"] = parent_id
                child_page_content = super().format_page_content_to_synthesize(child_chunk)
                synthetic_context = llm_service.ask_model(llm,"synthesize_context",llm_service.format_user_content("synthesize_context",chunk=child_page_content))
                child_chunk.page_content = f"""Context: {synthetic_context}
                                               Chunk: {child_chunk.page_content}"""
                child_chunks.append(child_chunk)
                logger.info(f"Contextualize {i+1} child chunk for no.{j+1} parent chunk successfully!")
        logger.info("Chunking all children into children chunks successfully")
        return child_chunks
    def _build_sub_parent_child_chunks(self,file_id,file,llm,db):
        parent_chunks = self.sub_parent_splitter.split_documents(file) #List[Document]
        logger.info("Chunking file into parent chunks with recursive splitter successfully")
        child_chunks = []

        logger.info("Starting chunking parent chunks")
        for j, parent_chunk in enumerate(parent_chunks):
            parent_id = str(uuid.uuid4())
            parent_chunk.page_content = super().format_page_content_to_synthesize(parent_chunk)
            db_service.create_parent_chunk(parent_id,file_id,parent_chunk.page_content,db) #Save to database
            logger.info(f"Save no.{j+1} parent chunk to database successfully!")
            children = self.child_splitter.split_documents([parent_chunk]) #Convert into List[Document] as split_documents apply only for list
            logger.info(f"Split child chunks for {j+1} parent chunk successfully!")
            for i,child_chunk in enumerate(children): 
                child_chunk.metadata["parent_id"] = parent_id
                child_page_content = super().format_page_content_to_synthesize(child_chunk)
                synthetic_context = llm_service.ask_model(llm,"synthesize_context",llm_service.format_user_content("synthesize_context",chunk=child_page_content))
                child_chunk.page_content = f"""Context: {synthetic_context}
                                               Chunk: {child_chunk.page_content}"""
                child_chunks.append(child_chunk)
                logger.info(f"Contextualize {i+1} child chunk for {j+1} parent chunk successfully!")
        logger.info("Chunking all children into children chunks successfully")
        return child_chunks
    def get_parent_chunks_from_database(self,reranked_docs,db):
        final_result=[]
        for child_chunk in reranked_docs:
            parent_chunk = db_service.get_parent_chunk(child_chunk.metadata["parent_id"],db) #ParentStore
            final_result.append(parent_chunk.context)
        return final_result #List[str]
    def do_split(self,**kwargs): 
        file_id =kwargs.get("file_id","")
        db =kwargs.get("db")
        is_complicated_file = kwargs.get("is_complicated_file")
        if not is_complicated_file:
            file = kwargs.get("file")
            if not file:
                raise ValueError("File parameter is required for non-complicated files")
            return self._build_sub_parent_child_chunks(file_id,file,llm,db)
        upload_file_path = kwargs.get("upload_file_path")
        markdown_content,doc_title = docling_service.convert_docling_to_markdown_file(upload_file_path)
        return self._build_parent_child_chunks(file_id,markdown_content,doc_title,llm,db) #List[Document]
