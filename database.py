import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load secret database configuration from local .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is missing from your local .env file!")

# Create the master SQL engine for database communication
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create temporary transaction sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# The base class that maps Python models to database tables
Base = declarative_base()


# The session generator for backend routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
