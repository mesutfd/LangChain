import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URI = os.getenv("POSTGRES_URI")
engine = create_engine(POSTGRES_URI)
