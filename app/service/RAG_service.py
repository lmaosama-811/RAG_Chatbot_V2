from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder 
from docling.document_converter import DocumentConverter
import os
import logging
import uuid

from ..model import embeddings
from ..core.env_config import settings 
from ..deps import get_processor_from_file_type
from ..deps import detect_language
from .DB_service import db_service

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self,embeddings):
        self.child_splitter = SemanticChunker(embeddings=embeddings)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200,separators=["\n\n", "\n", ".", " "])# seperators: if text is too large after split, recursive split according to seperators 
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.embeddings = embeddings 
        self.converter = DocumentConverter()
    def analyze_document(self,file_path):
        file = self.converter.convert(file_path) #Convert to docling.datamodel.base_models.ConversionResult
        doc =file.document #DoclingDocument
        markdown_content = doc.export_to_markdown() #Get markdown content => Use for MarkdownHeaderSplitter
        all_elements = list(doc.texts) #doc.texts là một Iterator (trình lặp) chứa các phân đoạn văn bản. Bằng cách ép kiểu sang list, ta lấy được danh sách các đối tượng văn bản.
        headers = [el for el in all_elements if "heading" in el.label.lower()] #Find element which label contains 'heading'
        return len(headers) >= 5 # If num of headers greater or equal to 5

    def rerank(self,query,documents,top_k=5):
        pairs = [(query,doc.page_content) for doc in documents] #create pair (query,doc)
        scores = self.reranker.predict(pairs) #predict scores 
        scored_docs = list(zip(documents,scores)) 
        scored_docs.sort(key=lambda x:x[1], reverse=True) #sort descending
        return [doc for doc,_ in scored_docs[:top_k]] #List[Document]
    def load_FAISS_and_retrieve(self,file_id,question,db):
        logger.info("Start processing File", extra={"file_id": file_id})

        processor = get_processor_from_file_type(file_id,db)
        file_path = processor.get_file_path("indexes",file_id)
        if not os.path.exists(file_path):
            return None 
        vectorstore = FAISS.load_local(file_path, self.embeddings,allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully")
        

        retriever = vectorstore.as_retriever(search_kwargs={"k":settings.top_k}) #Create wrapper object surrounding FAISS => save k, save search type and prepare for search (NOT search)
        docs = retriever.invoke(question) #get k chunks with highest similarity -> Measure vector distance -> List[Document]
        logger.info("Documents retrieved")
        
        reranked_docs = self.rerank(question,docs) #List[Document]
        logger.info("Rerank child chunks successfully")

        final_result=[]
        for child_chunk in reranked_docs:
            parent_chunk = db_service.get_parent_chunk(child_chunk.metadata["parent_id"],db) #ParentStore
            final_result.append(parent_chunk.context)
        logger.info("Get context from parent chunks successfully") 
        return "\n".join(final_result)
    def parse_file_and_save_FAISS(self,file,file_id,file_path,db): #file_path for indexes
        child_chunks = self.build_parent_child_chunks(file,file_id,db)
        for i,chunk in enumerate(child_chunks):
            chunk.metadata.update({"chunk_index": i,
                                  "total_chunk":len(child_chunks),
                                  "language":detect_language(db_service.clean_text(chunk.page_content))})
        logger.info("Text split completed",extra={"chunks": len(child_chunks)})

        vectorstore = FAISS.from_documents(child_chunks,self.embeddings)
        vectorstore.save_local(file_path)
        logger.info("FAISS index saved")
    def build_parent_child_chunks(self,file,file_id,db):
        parent_chunks = self.parent_splitter.split_documents(file)
        logger.info("Chunking file into parent chunks successfully")
        child_chunks = []

        logger.info("Starting chunking parent chunks")
        for parent_chunk in parent_chunks:
            parent_id = str(uuid.uuid4()) 
            db_service.create_parent_chunk(parent_id,file_id,parent_chunk.page_content,db) #Save to database
            children = self.child_splitter.split_documents([parent_chunk]) #Convert into List[Document] as split_documents apply only for list
            for child_chunk in children:
                child_chunk.metadata["parent_id"] = parent_id
                child_chunks.append(child_chunk)
        logger.info("Chunking children into children chunks successfully")
        return child_chunks

rag_service = RAGService(embeddings)