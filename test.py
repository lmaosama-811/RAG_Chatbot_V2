from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter 
from langchain_experimental.text_splitter import SemanticChunker
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings

"""
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key= "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",
    base_url="https://openrouter.ai/api/v1"
)
pdf = PyPDFLoader(r"C:\Users\Thanh Nam\Desktop\Sách\RAG.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
child_splitter = SemanticChunker(embeddings)
docs = text_splitter.split_documents(pdf.load())
children = child_splitter.split_documents(docs)"""

converter = DocumentConverter()
converted_file = converter.convert(r"C:\Users\Thanh Nam\Desktop\Sách\Các_ không_gian_chính_của_AI.pdf") #Convert to docling.datamodel.base_models.ConversionResult
print("After converting")
doc =converted_file.document #DoclingDocument
print("Before exporting")
markdown_content = doc.export_to_markdown()
print("After exporting. Prepare for split")
parent_splitter_2 = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),("####", "Header 4")])
print("Split successfully")
docs_1 = parent_splitter_2.split_text(markdown_content)
print(docs_1[2])