import logging 
import os
from logging.handlers import RotatingFileHandler
#handler = where log is saved 
#RotatingFileHandler = Write to a file and rotate automatically(if file is too large, create new file)

LOG_DIR="logs"
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR,"app.log")

def setup_logging():
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s" #time|INFO/ERROR/DEBUG|logger name|log content
    )
    console_handler = logging.StreamHandler() #StreamHandler: write log to terminal 
    console_handler.setFormatter(formatter)
    file_handler=RotatingFileHandler(
        LOG_FILE,
        maxBytes=5*1024*1024,
        backupCount=3
    ) # if file>5MB->rotate; only keep 3 file (oldest one will be delete)

    file_handler.setFormatter(formatter)

    #get root logger
    logger = logging.getLogger() #Root logger is parent logger of entire system, each descendant logger logging.getLogger(__name__) inheritate from root logger
    """logger = logging.getLogger("app")
    logger = logging.getLogger("app.service")
    logger = logging.getLogger("app.api.user") => Descendant logger => Each file should have seperate logger"""
    logger.setLevel(logging.INFO) #setLevel: Minimum level of severity allowed to be recorded. 
    """DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL: 50
    In this code, we only write log that >= INFO (dev can use DEBUG)"""
    """logger.propagate = True as default. Flow: Descendant logger create message, propagate to root logger to write outside => Easy management
    If logger.propagate = False-> Descendant logger doesn't propagate to root logger"""
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

