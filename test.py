from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key= "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",
    base_url="https://openrouter.ai/api/v1"
)
pdf = PyPDFLoader(r"C:\Users\Thanh Nam\Desktop\Sách\RAG.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
child_splitter = SemanticChunker(embeddings)
docs = text_splitter.split_documents(pdf.load())
children = child_splitter.split_documents(docs)
print(children[0])