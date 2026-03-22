import eventlet
eventlet.monkey_patch()

from celery import Celery 
from sqlmodel import Session
import asyncio
import re

from ..service.File_service.FileIngestion import file_ingestion
from ..service.DB_service import db_service
from ..db import engine 
from .env_config import settings 

import logging

celery_app = Celery("myapp", broker=settings.redis,
                    backend=settings.redis)
celery_app.conf.update(task_track_started=True)
celery_app.conf.beat_schedule ={}



logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_upload_file_background(self,file_id,file_name,strategy):
    with Session(engine) as db:
        try:
            self.update_state(state="STARTED")

            logger.info(f"Start ingesting file {file_id}")
            file_ingestion.process_single_upload_file(file_id,file_name,strategy,db)
            logger.info(f"Ingesting file {file_id} successfully")
            
            return {"file_id": file_id, "status": "SUCCESS"}
        except Exception as e:
            db_service.update_file_status(file_id,"FAILURE",db)
            logger.error(f"Error processing file {file_id}: {str(e)}")
            raise self.retry(exc=e,countdown=5,max_retries=1) #raise retry -> state = 'RETRY', after 5s worker run this task again

@celery_app.task(bind=True)
def process_crawled_web_background(self,url,strategy:str=settings.strategy):
    with Session(engine) as db:
        try:
            result = asyncio.run(file_ingestion.scrape_article_and_attachments(url)) #wrapper async function with asyncio to create event loop for async tasks
            # As celery is synchronize and file_ingestion.scrape_article_and_attachments is async => async function return courotine object (not run right aways)
            # and celery thinks that it has done ít work -> return coroutine object. -> Need an event loop to run coroutine object
            if result is None:
                logger.warning(f"Worker ignore this url as it failed to crawl: {url}")
                return
            markdown_file, attachments = result["markdown_file"], result["attachments"]
            logger.info(f"Worker successfully crawled: {url}")
            
            logger.info(f"Start ingesting markdown content in {url}")
            file_ingestion.process_crawl_file(markdown_file[2],markdown_file[1],strategy,markdown_file[0],db,markdown_file[3])
            logger.info(f"Ingesting markdown content in {url} successfully")

            logger.info(f"Start ingesting attachments in {url}. Length: {len(attachments)}")
            for i, attachment in enumerate(attachments):
                file_ingestion.process_crawl_file(attachment[2],attachment[1],strategy,attachment[0],db,markdown_file[3])
                logger.info(f"Ingesting attachment {i+1}/{len(attachments)}: {attachment[1]} successfully")
            logger.info(f"Ingesting attachments in {url} successfully")
            return 
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing crawled web {url}: {str(e)}")
            raise self.retry(exc=e,countdown=5,max_retries=1) #raise retry -> state = 'RETRY', after 5s worker run this task again
            
#Tắt các log không cần thiết 
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ----------------- CELERY BEAT CRON JOBS -----------------
from celery.schedules import crontab
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

celery_app.conf.beat_schedule = {
    'auto-crawl-chinhphu-news-every-6-hours': {
        'task': 'app.core.celery_app.task_trigger_auto_crawl',
        'schedule': crontab(minute=0,hour="*/6"), # Run/6 hours
    },
}

@celery_app.task(bind=True)
def task_trigger_auto_crawl(self):
    homepage_urls = ["https://xaydungchinhsach.chinhphu.vn/", "https://baochinhphu.vn/","https://chinhphu.vn/"]
    strategy = settings.strategy
    
    total_queued = 0
    for homepage_url in homepage_urls:
        logger.info(f" Cron job started: Checking for new articles at {homepage_url}")
        with Session(engine) as db:
            try:
                response = requests.get(homepage_url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                links = soup.find_all('a', href=True)
                valid_urls = set()
                for link in links:
                    href = link['href']
                    # Applying Regex to identify links containing article IDs (usually with numbers before .htm) to block 100% of category links or homepage
                    if (href.endswith('.htm') or href.endswith('.html')) and re.search(r'\d+', href):
                        full_url = urljoin(homepage_url, href)
                        if full_url != homepage_url:
                            valid_urls.add(full_url)
                
                logger.info(f"Found {len(valid_urls)} valid article links at {homepage_url}.")
                new_count = 0
                for url in list(valid_urls)[:20]: # Only check 20 latest articles on homepage
                    file_name = url.split('/')[-1].split('?')[0] + ".md"
                    
                    # Anti-duplicate: Check in Database
                    if not db_service.check_file_exists_by_name(file_name, db):
                        logger.info(f"New article found! Queuing to Worker: {url}")
                        # Queue to Worker
                        process_crawled_web_background.delay(url, strategy)
                        new_count += 1
                        total_queued += 1
                
                logger.info(f"Finished checking {homepage_url}. Queued: {new_count} new articles")
            except Exception as e:
                logger.error(f"Auto-crawl cron failed at {homepage_url}: {str(e)}")
                # Continue process next homepage even if one fails
                continue
                
    logger.info(f"Auto-update completely finished. Successfully queued {total_queued} new articles across all sites.")
    return f"Queued a total of {total_queued} new articles"

