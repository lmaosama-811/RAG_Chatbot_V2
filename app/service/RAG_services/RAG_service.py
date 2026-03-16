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
    def build_block_context(self, docs): #List[Document]
        blocks = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_info = f"trang {page}" if page is not None else "trang N/A"
            file_name = doc.metadata.get("file_name", "unknown")

            blocks.append(
                f"[{i}] {file_name} | {page_info}\n"
                f"{doc.page_content[:400]}"
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
    
    def multi_query_hybrid_search(self,question,bm25,faiss,generated_queries, top_k=settings.top_k,bm25_w=settings.bm25_w,faiss_w=settings.faiss_w):
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
        

    def load_FAISS_and_retrieve(self,file_id,question,analysis,strategy,db):
        logger.info("Start processing File", extra={"file_id": file_id})

        processor = get_processor_from_file_type(file_id,db)
        chunker = ChunkerRegistry.get_registry(strategy)
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
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k":settings.top_k}) #Create wrapper object surrounding FAISS => save k, save search type and prepare for search (NOT search)

        # ── Retrieve and retry ───────────────────────────────
        attempts = 0
        queries = analysis["round1"]
        while attempts < settings.max_retry: # max 2
            if not queries:
                break 
            docs = self.multi_query_hybrid_search(question, bm25_retriever, faiss_retriever,queries) # List[Document]

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
        
    def parse_file_and_save_FAISS(self,file,is_complicated_file,file_id,upload_file_path,index_file_path,strategy,db): 
        chunker = ChunkerRegistry.get_registry(strategy)
        list_chunks = chunker.do_split(file=file,is_complicated_file=is_complicated_file,upload_file_path=upload_file_path, file_id=file_id, db=db)
        for i,chunk in enumerate(list_chunks):
            chunk.metadata.update({"chunk_index": i,
                                  "total_chunk":len(list_chunks),
                                  "language":detect_language(db_service.clean_text(chunk.page_content)),
                                  "page":chunk.metadata.get("page",None)})
        logger.info("Text split completed",extra={"chunks": len(list_chunks)})

        vectorstore = FAISS.from_documents(list_chunks,self.embeddings)
        vectorstore.save_local(index_file_path)
        logger.info("FAISS and BM25 index saved")

rag_service = RAGService()
