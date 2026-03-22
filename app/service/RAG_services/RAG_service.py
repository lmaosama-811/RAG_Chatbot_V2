from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import os, pickle
import logging

import app.service.RAG_services.ChunkSplitters
from .ChunkSplitters.ChunkFactory import ChunkerRegistry 
from ...model import embeddings,llm
from ...core.env_config import settings 
from ...deps import get_processor_from_file_type, detect_language
from ..DB_service import db_service
from ..LLM_service import llm_service

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embeddings = embeddings
        self.global_index_path = "government_data/global_faiss_index"
        self.global_vectorstore = None
        self.global_bm25 = None
        self.last_loaded_mtime = 0.0
        self.load_global_index()

    def load_global_index(self):
        # Load Global FAISS and BM25 index into RAM when Uvicorn starts or RAGService initializes
        faiss_file = os.path.join(self.global_index_path, "index.faiss")
        if os.path.exists(faiss_file):
            self.last_loaded_mtime = os.path.getmtime(faiss_file)
            self.global_vectorstore = FAISS.load_local(self.global_index_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # Read file index.pkl which FAISS has saved directly 
            with open(f"{self.global_index_path}/index.pkl", "rb") as f:
                raw = pickle.load(f)
                
            # raw is (docstore, index_to_docstore_id). docstore._dict is {id: Document} => Get Document (.values()) and convert to List[Document]
            docs = list(raw[0]._dict.values())
            
            if docs:
                self.global_bm25 = BM25Retriever.from_documents(docs, k=settings.top_k)
                logger.info(f"Global FAISS and BM25 index loaded into RAM successfully! ({len(docs)} chunks)")

    def build_block_context(self, docs): #List[Document]
        blocks = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            file_name = doc.metadata.get("file_name", "Tài liệu Chính phủ")
            
            # Chỉ hiển thị số trang nếu nó tải từ PDF/Word, bằng cách kiểm tra if page
            page_info = f" | Trang {page}" if page is not None else ""

            blocks.append(
                f"[{i}] Nguồn: {file_name}{page_info}\n"
                f"Trích xuất: {doc.page_content[:400]}..."
            )
        return "\n\n".join(blocks)
    
    def _rrf_score(self,bm25_results: list,faiss_results: list,bm25_w: float,faiss_w: float,k: int = 60):
        """
        Compute RRF score from 2 ranked list.
        Return (scores_dict, docs_dict)
        """
        n          = max(len(bm25_results), len(faiss_results), 1)

        #Create ranked list for BM25 and FAISS 
        bm25_rank  = {doc.page_content: rank for rank, doc in enumerate(bm25_results)}
        faiss_rank = {doc.page_content: rank for rank, doc in enumerate(faiss_results)}
        all_docs   = {doc.page_content: doc for doc in bm25_results + faiss_results}

        scores = {
            c: bm25_w  / (k + bm25_rank.get(c, n))
             + faiss_w / (k + faiss_rank.get(c, n))
            for c in all_docs
        } #Compute score
        return scores, all_docs
    
    def relevance_and_sufficiency_check(self,docs, min_docs=settings.min_relevant_docs, score_threshold=settings.relevance_threshold):
        relevant_docs = [d for d in docs if d.metadata.get("_rrf_score", 1) >= score_threshold] # Only get documents that rrf score greater than score_threshold => Get relevant documents
        return len(relevant_docs) > 0, len(relevant_docs) >= min_docs #is_relevant, is_suficiency 
    
    def multi_query_hybrid_search(self,question,bm25,faiss,generated_queries, filter_dict=None, top_k=settings.top_k,bm25_w=settings.bm25_w,faiss_w=settings.faiss_w):
        #Generate queries if None
        if not generated_queries:
            logger.warning("Generated queries is empty")
            raise ValueError("Generated queries is empty") 

        generated_queries = list(set([question] + generated_queries))
        accumulated_scores= {}
        accumulated_docs = {}
        for q in generated_queries:
            bm25_results  = bm25.invoke(q)
            faiss_results = faiss.invoke(q)
            
            # Manual metadata soft-filtering for BOTH BM25 and FAISS
            if filter_dict:
                def is_valid(doc):
                    for k, v in filter_dict.items():
                        doc_val = doc.metadata.get(k)
                        # Soft filter: Only drop documents that explicitly have a conflicting year.
                        # If a document has no 'year' metadata (None), KEEP it!
                        if doc_val is not None and doc_val != v:
                            return False
                    return True
                
                bm25_results = [d for d in bm25_results if is_valid(d)]
                faiss_results = [d for d in faiss_results if is_valid(d)]

            scores,all_docs = self._rrf_score(bm25_results,faiss_results,bm25_w,faiss_w)
            for content, score in scores.items():
                accumulated_scores[content] = accumulated_scores.get(content, 0) + score

            accumulated_docs.update(all_docs)
        ranked = sorted(accumulated_scores.items(), key=lambda x: x[1], reverse=True)

        #Asign score to each document => use later 
        result = []
        for content, score in ranked[:top_k]:
            doc = accumulated_docs[content]
            doc.metadata["_rrf_score"] = round(score, 6)
            result.append(doc)

        return result
        
    def _retrieve_and_format(self, question, analysis, strategy, db, bm25_retriever, faiss_retriever):
        """
        Helper method to perform retrieval, retry logic, reranking, and formatting.
        Abstracts the common core functionality shared between specific-file retrieval and global retrieval.
        """
        chunker = ChunkerRegistry.get_registry(strategy)
        filter_dict = analysis.get("filter", {})
        
        # ── Retrieve and retry ───────────────────────────────
        attempts = 0
        queries = analysis["round1"]
        while attempts < settings.max_retry: # max 2
            if not queries:
                break 
            docs = self.multi_query_hybrid_search(question, bm25_retriever, faiss_retriever, queries, filter_dict) # List[Document]

            is_relevant, is_sufficient = self.relevance_and_sufficiency_check(docs)
            logger.info("Relevance check",
                        extra={"relevant": is_relevant,
                            "sufficient": is_sufficient,
                            "attempt": attempts + 1})

            if is_relevant and is_sufficient:
                break

            # Reformulate — use another version of queries in analysis
            attempts += 1
            queries = analysis["round2"]
            logger.warning("Retrying with other queries", extra={"attempt": attempts})

        # ── Sufficiency check: lack of context ─────────────
        if not docs or not is_sufficient:
            return None
        
        logger.info("Documents retrieved")
        
        reranked_docs = chunker.rerank(question,docs) #List[Document]
        logger.info("Rerank child chunks successfully")
        
        if strategy == 'hybrid':
            final_result = chunker.get_parent_chunks_from_database(reranked_docs, db) #List[str]
            logger.info("Get context from parent chunks successfully") 
            return "\n\n".join(f"[{i}] {content[:400]}" for i, content in enumerate(final_result, start=1)) # List[str] with citation
        else:
            logger.info("Format final result successfully")
            return self.build_block_context(reranked_docs)  # List[str] with citation

    def load_FAISS_and_retrieve(self,file_id,question,analysis,strategy,db):
        logger.info("Start processing File", extra={"file_id": file_id})

        processor = get_processor_from_file_type(file_id,db)
        index_file_path = processor.get_file_path("indexes",file_id)
        if not os.path.exists(index_file_path):
            return None 
            
        vectorstore = FAISS.load_local(index_file_path, self.embeddings,allow_dangerous_deserialization=True)
        # Read file index.pkl which FAISS has saved directly 
        with open(f"{index_file_path}/index.pkl", "rb") as f:
            raw = pickle.load(f)

        docs = list(raw[0]._dict.values()) # raw is (docstore, index_to_docstore_id). docstore._dict is {id: Document} => Get Document (.values()) and convert to List[Document]
        logger.info("FAISS and BM25 index loaded successfully")

        bm25_retriever = BM25Retriever.from_documents(docs, k=settings.top_k)
        
        # Create wrapper object surrounding FAISS => save k, save search type and prepare for search (NOT search)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})

        return self._retrieve_and_format(question, analysis, strategy, db, bm25_retriever, faiss_retriever)

    def retrieve_from_global(self, question, analysis, strategy, db):
        # Check if Celery just updated the Global FAISS index on disk 
        # By comparing the latest modification time. If updated -> Hot-Reload into Uvicorn RAM instantly.
        faiss_file = os.path.join(self.global_index_path, "index.faiss")
        if os.path.exists(faiss_file) and os.path.getmtime(faiss_file) > self.last_loaded_mtime:
            logger.info("Hot-reloading Global FAISS index into RAM - New updates detected from Celery Worker!")
            self.load_global_index()

        # Query directly from Global FAISS Index stored in RAM 
        if not self.global_vectorstore or not self.global_bm25:
            logger.warning("Global FAISS index not found in RAM!")
            return None
            
        logger.info("Querying from Global FAISS Database...")
        
        # Create wrapper object surrounding FAISS => save k, save search type and prepare for search (NOT search)
        faiss_retriever = self.global_vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
        
        return self._retrieve_and_format(question, analysis, strategy, db, self.global_bm25, faiss_retriever)

    def parse_file_and_save_FAISS(self,file,is_complicated_file,file_id,upload_file_path,index_file_path,strategy,db,file_name=None): 
        chunker = ChunkerRegistry.get_registry(strategy)
        list_chunks = chunker.do_split(file=file,is_complicated_file=is_complicated_file,upload_file_path=upload_file_path, file_id=file_id, db=db)
        
        import re
        extracted_year = None
        if file_name:
            year_match = re.search(r'\b(20\d{2})\b', file_name)
            if year_match:
                extracted_year = int(year_match.group(1))

        for i,chunk in enumerate(list_chunks):
            chunk.metadata.update({"chunk_index": i,
                                  "total_chunk":len(list_chunks),
                                  "language":detect_language(db_service.clean_text(chunk.page_content)),
                                  "page":chunk.metadata.get("page",None)})
            if file_name:
                chunk.metadata["file_name"] = file_name
            if extracted_year:
                chunk.metadata["year"] = extracted_year
        logger.info("Text split completed",extra={"chunks": len(list_chunks)})

        # Safeguard: Prevent FAISS crashing when crawling an empty page or a blank attachment file
        if not list_chunks:
            logger.warning(f"File {file_id} generated 0 chunks! Skipping FAISS and BM25 index updating.")
            return

        if os.path.exists(os.path.join(index_file_path, "index.faiss")):
            # If FAISS database already exists, load and use add_documents to append new chunks safely to prevent overwriting
            vectorstore = FAISS.load_local(index_file_path, self.embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(list_chunks)
            vectorstore.save_local(index_file_path)
            logger.info(f"Appended new chunks into existing FAISS db at {index_file_path}")
        else:
            # Create new FAISS database if none exists
            vectorstore = FAISS.from_documents(list_chunks, self.embeddings)
            vectorstore.save_local(index_file_path)
            logger.info(f"Created new FAISS db at {index_file_path}")
            
        # Reload into RAM immediately to keep the server updated with the latest crawled data
        if "global_faiss_index" in index_file_path:
            self.load_global_index()

rag_service = RAGService()
