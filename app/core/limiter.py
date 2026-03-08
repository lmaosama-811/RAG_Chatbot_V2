from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address) #Limiter object: save request history, examine whether this request exceed threshold and raise if exceeding
# key_func: the function that use to determine who send request 
# get_remote_address: get request.client.host. Ex: IP 123.45.67.89 → 10 request
#assign to app state 