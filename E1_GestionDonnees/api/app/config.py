from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from pathlib import Path


env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    """
    Application configuration
    Uses environment variables from .env file
    """
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # API
    API_VERSION: str = os.getenv("API_VERSION", "1.0")
    API_TITLE: str = os.getenv("API_TITLE", "API France Optique")
    API_DESCRIPTION: str = os.getenv("API_DESCRIPTION", "API for optical lens management")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    
    # Admin
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    class Config:
        env_file = ".env"

settings = Settings() 