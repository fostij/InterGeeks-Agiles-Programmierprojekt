import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Read the hidden configuration file you just created
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from your local .env configuration file!")

engine = create_engine(DATABASE_URL)
