"""
LMS Analytics Agent - Configuration
"""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "3306"))
    db_name: str = os.getenv("DB_NAME", "mctlms")
    db_user: str = os.getenv("DB_USER", "root")
    db_password: str = os.getenv("DB_PASSWORD", "")
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Server
    port: int = int(os.getenv("PORT", "8001"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Query Limits
    max_query_rows: int = int(os.getenv("MAX_QUERY_ROWS", "1000"))
    query_timeout_seconds: int = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))
    
    @property
    def database_url(self) -> str:
        """Generate MySQL connection URL"""
        from urllib.parse import quote_plus
        encoded_password = quote_plus(self.db_password)
        return f"mysql+mysqlconnector://{self.db_user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    class Config:
        env_file = ".env"


settings = Settings()
