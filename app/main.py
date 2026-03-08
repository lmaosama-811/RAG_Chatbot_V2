from fastapi import FastAPI, Request
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

from .api import upload,chat,session
from .core.logging_config import setup_logging
from .core.limiter import limiter 

setup_logging()

app = FastAPI()
#create limiter

app.state.limiter =limiter

#add middleware
app.add_middleware(SlowAPIMiddleware)
#SlowAPIMiddleware: get request, find limiter, examine limiter, if exceed ->raise RateLimitExceeded else allow request to proceed
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request:Request,exc:RateLimitExceeded):
    return JSONResponse(status_code=429,content={"detail":"Too many request"})

app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(session.router)
