from langchain_experimental.text_splitter import SemanticChunker

from .ChunkBase import ChunkBase
from .ChunkFactory import ChunkerRegistry 
from ....model import embeddings

@ChunkerRegistry.register("semantic")
class SemanticChunk(ChunkBase):
    def __init__(self):
        super().__init__()
        self.splitter = SemanticChunker(embeddings)
    def do_split(self,**kwargs):
        file = kwargs.get('file',"")
        return self.splitter.split_documents(file) #List[Document]