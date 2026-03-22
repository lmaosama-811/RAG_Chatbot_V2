import logging,os,requests
import uuid
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler

from .FileFactory import FileProcessorRegistry
from ..RAG_services.RAG_service import rag_service
from ..DB_service import db_service
from .Docling_service import docling_service
from ...core.env_config import settings

logger = logging.getLogger(__name__)

class FileIngestion:
    @staticmethod
    def process_single_upload_file(file_id:str, file_name:str,strategy:str,db):
        processor = FileProcessorRegistry.get_registry(file_name)
        index_file_path = processor.get_file_path("indexes",file_id)
        upload_file_path = processor.get_file_path("upload",file_id)

        logger.info(f"Start analyzing file {file_id}")
        is_complicated_file = FileProcessorRegistry.inspect_file_and_routing(file_id,processor) #return True/False 
        logger.info(f"Analyzing successfully: Class: {is_complicated_file}")
                
        file = (docling_service.convert_docling_to_list_document(upload_file_path) if (is_complicated_file and strategy=="semantic")
                    else (processor.process_file(file_id) if not is_complicated_file else None)) #List[Document]
        logger.info('Load file successfully')
            
        rag_service.parse_file_and_save_FAISS(file,is_complicated_file,file_id,upload_file_path,index_file_path,strategy,db,file_name=file_name)
        logger.info(f"FAISS index created for file {file_id}")

        db_service.update_file_status(file_id,"SUCCESS",db)
        logger.info(f"Update file status successfully! State: SUCCESS")
        return 

    @staticmethod
    def process_crawl_file(file_id:str, file_name:str,strategy:str,upload_file_path:str,db,parent_dir_name:str=None):
        processor = FileProcessorRegistry.get_registry(file_name)
        index_file_path = "government_data/global_faiss_index"
        if not os.path.exists(index_file_path):
            os.makedirs(index_file_path,exist_ok=True)

        logger.info(f"Start analyzing file {file_id}")
        is_complicated_file = FileProcessorRegistry.inspect_file_and_routing(file_id,processor,upload_file_path=upload_file_path) #return True/False 
        logger.info(f"Analyzing successfully: Class: {is_complicated_file}")
                
        file = (docling_service.convert_docling_to_list_document(upload_file_path) if (is_complicated_file and strategy=="semantic")
                    else (processor.process_file(file_id,upload_file_path) if not is_complicated_file else None)) #List[Document]
        logger.info('Load file successfully')
            
        rag_service.parse_file_and_save_FAISS(file,is_complicated_file,file_id,upload_file_path,index_file_path,strategy,db,file_name=file_name)
        logger.info(f"FAISS index created for file {file_id}")

        db_service.create_file(db,file_id,file_name,os.path.splitext(file_name)[1].lower(),"SUCCESS")
        logger.info(f"Update file status successfully! State: SUCCESS")
        return 

    @staticmethod
    def download_crawled_attachment(file_url: str, download_dir: str = "downloads"):
        """Use requests to download attachment"""
        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)
            
        try:
            file_name = file_url.split('/')[-1]
            file_name = file_name.split('?')[0]
            file_id = uuid.uuid4().hex
            new_filename = f"{file_id}_{file_name}"

            file_path = os.path.join(download_dir, new_filename)
            
            response = requests.get(file_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Download successfully: {file_name}")
            return file_path, file_name, file_id
        except Exception as e:
            logger.error(f"Error in dowloading process {file_url}: {e}")
            return None,None, None

    @staticmethod
    async def scrape_article_and_attachments(url: str, save_dir: str = "government_data"): # Handle 1 url => Get Markdown content on that web and attachments (if exists)
        target_extensions = settings.target_extensions
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Crawling data from: {url}")
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, bypass_cache=True) #Call crawler to start crawl in web at url path. await: program "pause" until crawler has loaded web.
            # bypass_cache = True: crawler will ignore cache and crawl the web again => guarantee up-to-date.
            #result includes raw HTML, Markdown content and state whether success or not.
            if not result.success:
                logger.error(f"Error when crawling {url}: {result.error_message}")
                return None
            
            file_id = uuid.uuid4().hex
            file_name = url.split('/')[-1]
            file_name = file_name.split('?')[0]
            folder_name = f"{file_id}_{file_name}"
            url_specific_dir = os.path.join(save_dir, folder_name)
            if not os.path.exists(url_specific_dir):
                os.makedirs(url_specific_dir, exist_ok=True) #Create a specific folder for each url (post)
            # 1. Use crawl4ai to take Markdown for RAG
            markdown_content = result.markdown #Markdown content is used for RAG.
            md_file_path = os.path.join(url_specific_dir, f"article_content.md")
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(f"# Source: {url}\n\n")
                f.write(markdown_content)
            logger.info(f"Saved Markdown content to: {md_file_path}")


            # 2. Use BeautifulSoup on original HTML to find attachments
            html_content = result.html #raw HTML content is used to find attachments.
            soup = BeautifulSoup(html_content, 'html.parser') #initialize parser => convert raw HTML to tree structure (DOM tree) => find tags inside 
            links = soup.find_all('a', href=True) #Find all <a> tags link in web (if that tag has link path href=True)=> return List[<a>]
            
            downloaded_files = []
            attachments_dir = os.path.join(url_specific_dir, "attachments")
            if not os.path.exists(attachments_dir):
                os.makedirs(attachments_dir, exist_ok=True) #Create a specific folder for attachments
            
            for link in links:
                href = link['href'] #Get link path from <a> tag
                if any(href.lower().endswith(ext) for ext in target_extensions): #Check if link path ends with any of the target extensions
                    full_file_url = urljoin(url, href) #Join the base url with the link path to get the full file url 
                    """If link is: https://domain.com/file.pdf (absolute path) -> remain.
                    If link is: /tailieu/file.pdf (relative path) and original URL is https://gov.vn -> it will automatically combine them into https://gov.vn/tailieu/file.pdf."""
                    logger.info(f"Found attachment: {full_file_url}")
                    
                    saved_path, saved_file_name, saved_file_id = FileIngestion.download_crawled_attachment(full_file_url, download_dir=attachments_dir)
                    if saved_path:
                        downloaded_files.append((saved_path,saved_file_name,saved_file_id))

            return {
                "markdown_file": (md_file_path, f"{file_name}.md", file_id, folder_name), #tuple: (path of markdown file, file_name, file_id, folder_name)
                "attachments": downloaded_files #List[tuple]: (path of downloaded file, file_name, file_id)
            }

file_ingestion = FileIngestion()