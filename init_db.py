#!/usr/bin/env python3
"""init_db.py
Tao tat ca database tables khi chay lan dau.
Goi boi docker-compose api service khi startup,
hoac co the chay thu cong:
    python3 init_db.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlmodel import SQLModel
from app.db import engine
# Import cac model de SQLModel biet can tao table nao
from app.schemas.request_model import (
    ConversationHistory,
    Summary,
    FileStatus,
    ParentStore,
)

def main():
    print('[init_db] Creating all tables...')
    SQLModel.metadata.create_all(engine)
    print('[init_db] Done. Tables created:')
    for table in SQLModel.metadata.tables:
        print(f'  - {table}')

if __name__ == '__main__':
    main()
