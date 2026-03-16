import redis 
from .env_config import settings

r =redis.from_url(settings.redis)