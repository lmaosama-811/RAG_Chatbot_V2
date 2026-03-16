from abc import ABC, abstractmethod
from sentence_transformers import CrossEncoder 

class ChunkBase(ABC):
    def __init__(self):
        self._reranker = None 

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker
    def rerank(self,query,documents,top_k=5):
        pairs = [(query,doc.page_content) for doc in documents] #create pair (query,doc)
        scores = self.reranker.predict(pairs) #predict scores 
        scored_docs = list(zip(documents,scores)) 
        scored_docs.sort(key=lambda x:x[1], reverse=True) #sort descending
        return [doc for doc,_ in scored_docs[:top_k]] #List[Document]
    def build_section_path(self,metadata):
        return ">".join(v for v in [metadata.get("Header 1",""), metadata.get("Header 2",""), metadata.get("Header 3",""), metadata.get("Header 4","")])
    def format_page_content_to_synthesize(self,chunk): #parent chunk or child chunk
        return f"""Title: {chunk.metadata.get("title","")}
                   Section path: {self.build_section_path(chunk.metadata)}
                   Content: {chunk.page_content} """
    @abstractmethod
    def do_split(self,**kwargs):
        pass