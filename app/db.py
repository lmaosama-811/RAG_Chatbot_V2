from sqlmodel import create_engine, Session # an intermediate object between Python and database, manage: query,transaction,commit,rollback,connection.

from .core.env_config import settings 

engine = create_engine(settings.database_url)

def get_session():
    with Session(engine) as session:
        yield session