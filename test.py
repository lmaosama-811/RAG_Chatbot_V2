from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter 
from langchain_experimental.text_splitter import SemanticChunker
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings


converter = DocumentConverter()
converted_file = converter.convert(r"C:\Users\Thanh Nam\Desktop\Sách\Các_ không_gian_chính_của_AI.pdf") #Convert to docling.datamodel.base_models.ConversionResult
doc =converted_file.document #DoclingDocument
markdown_content = doc.export_to_markdown()
parent_splitter_2 = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),("####", "Header 4")])
docs_1 = parent_splitter_2.split_text(markdown_content)
print(docs_1[0])