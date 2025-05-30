from pydantic_settings import BaseSettings
from pydantic import field_validator
import os
from pathlib import Path

# WORKSPACE_ROOT est la racine de votre projet (par exemple, certif_simplon)
# config.py est dans E1_GestionDonnees/api/app/config.py
# Path(__file__).resolve() -> .../E1_GestionDonnees/api/app/config.py
# .parent -> .../E1_GestionDonnees/api/app
# .parent.parent -> .../E1_GestionDonnees/api
# .parent.parent.parent -> .../E1_GestionDonnees
# .parent.parent.parent.parent -> .../certif_simplon (WORKSPACE_ROOT)
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    """
    Application configuration.
    Pydantic charge les variables depuis le fichier .env spécifié dans Config.
    """
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "votre_cle_secrete_ici")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Database PostgreSQL
    db_api_host: str = os.getenv("DB_API_HOST", "localhost")
    db_api_port: str = os.getenv("DB_API_PORT", "5432")
    db_api_name: str = os.getenv("DB_API_NAME", "glass_db")
    db_api_user: str = os.getenv("DB_API_USER", "postgres")
    db_api_password: str = os.getenv("DB_API_PASSWORD", "azerton3359")
    
    # Database SQLite
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "E1_GestionDonnees/Base_de_donnees/france_optique.db")
    DATABASE_URL: str = ""  # Sera configuré par le validateur
    
    # API
    API_VERSION: str = os.getenv("API_VERSION", "1.0")
    API_TITLE: str = os.getenv("API_TITLE", "API France Optique")
    API_DESCRIPTION: str = os.getenv("API_DESCRIPTION", "API for optical lens management")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")    # URLs des services
    api_url_model: str = os.getenv("API_URL_MODEL", "http://localhost:8000")  # Renamed to avoid conflict
    api_url_db: str = os.getenv("API_URL_DB", "http://localhost:8001")  # Renamed to avoid conflict
    
    # Admin
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@example.com")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    api_port: str = os.getenv("API_PORT", "8001")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str, values) -> str:
        """Configure l'URL de la base de données PostgreSQL."""
        return f"postgresql://{values.data['db_api_user']}:{values.data['db_api_password']}@{values.data['db_api_host']}:{values.data['db_api_port']}/{values.data['db_api_name']}"
    
    model_config = {
        "env_file": ".env",
        "protected_namespaces": ('settings_',),  # Change the protected namespace
        "json_schema_extra": {
            "example": {
                "SECRET_KEY": "your-secret-key",
                "ALGORITHM": "HS256",
                "ACCESS_TOKEN_EXPIRE_MINUTES": 30
            }
        }
    }

settings = Settings() 