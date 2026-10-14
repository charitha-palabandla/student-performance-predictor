import os
from sqlalchemy import create_engine

DB_URL = os.getenv("DB_URL", "sqlite:///student_db.sqlite")

def get_engine():
    return create_engine(DB_URL)