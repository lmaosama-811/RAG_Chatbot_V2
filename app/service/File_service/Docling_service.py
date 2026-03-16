from docling.document_converter import DocumentConverter
from langchain_docling import DoclingLoader 
import os 
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import DoclingDocument
#Use this for low settings
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = False  # Tắt TableFormer (Rất tốn RAM)
pipeline_options.do_ocr = True               # Nếu PDF đã có text (Vector PDF), hãy tắt OCR

class DoclingService:
    def __init__(self):
        self.converter = DocumentConverter(allowed_formats=[InputFormat.PDF,InputFormat.DOCX,InputFormat.MD],
                                           format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)})
    def loading_Docling_Document(self,upload_file_path):
        if upload_file_path.endswith(".txt"):
            with open(upload_file_path, "r", encoding="utf-8") as f:
                return f.read()
        converted_file = self.converter.convert(upload_file_path) #Convert to docling.datamodel.base_models.ConversionResult
        return converted_file.document #DoclingDocument
    def convert_docling_to_markdown_file(self,upload_file_path):
        doc = self.loading_Docling_Document(upload_file_path)
        doc_title = (doc.name if isinstance(doc,DoclingDocument) else os.path.basename(upload_file_path))
        markdown_content = (doc.export_to_markdown() if isinstance(doc,DoclingDocument) else doc) #Get markdown content => Use for MarkdownHeaderSplitter
        return markdown_content,doc_title 
    def convert_docling_to_list_document(self,upload_file_path): #Convert pdf, docx, pptx, html, markdown to List[Document] (but slower than pymupdf and python-docx)
        loader = DoclingLoader(upload_file_path)
        list_LCDocument = loader.load()
        raw_name = os.path.basename(upload_file_path)
        file_id, file_name = raw_name.split("_",1)
        for document in list_LCDocument:
            document.metadata.update({"file_id":file_id,
                                      "file_name":file_name})
        return list_LCDocument

docling_service = DoclingService()
