from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .ChunkBase import ChunkBase
from .ChunkFactory import ChunkerRegistry 
from ...File_service.Docling_service import docling_service

@ChunkerRegistry.register("hierarchical")
class HierarchicalChunk(ChunkBase):
    def __init__(self):
        super().__init__()
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),("####", "Header 4")])
        self.sub_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n","\n"," ",""])
    def do_split(self, **kwargs):
        is_complicated_file = kwargs.get("is_complicated_file","")
        if not is_complicated_file:
             file = kwargs.get("file","")
             if not file:
                raise ValueError("File parameter is required for non-complicated files")
             return self.sub_splitter.split_documents(file) #List[Document]
        upload_file_path= kwargs.get("upload_file_path","")
        markdown_content,doc_title = docling_service.convert_docling_to_markdown_file(upload_file_path)
        sections = self.splitter.split_text(markdown_content)
        for section in sections:
                section.metadata["title"] = doc_title
        return sections #List[Document]
