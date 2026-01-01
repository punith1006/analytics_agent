"""
Database connection and session management
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.debug
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def execute_raw_query(query: str, params: dict = None):
    """
    Execute a raw SQL query and return results as list of dicts
    
    Args:
        query: SQL query string
        params: Optional query parameters
        
    Returns:
        List of dictionaries with query results
    """
    with engine.connect() as connection:
        result = connection.execute(text(query), params or {})
        columns = result.keys()
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]
