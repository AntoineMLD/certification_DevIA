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
    
    # Database
    DATABASE_URL: str # Sera chargée depuis .env et validée/transformée ci-dessous

    # API
    API_VERSION: str = os.getenv("API_VERSION", "1.0")
    API_TITLE: str = os.getenv("API_TITLE", "API France Optique")
    API_DESCRIPTION: str = os.getenv("API_DESCRIPTION", "API for optical lens management")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    
    # Admin
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@example.com")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str) -> str:
        """
        Prend le chemin brut DATABASE_URL depuis .env,
        le résout par rapport à la racine du projet (WORKSPACE_ROOT),
        et préfixe avec le schéma sqlite:///.
        """
        if not (isinstance(v, str) and v.strip()):
            # Cela se produira si DATABASE_URL est manquant ou vide dans .env
            raise ValueError("DATABASE_URL dans .env doit être un chemin non vide sous forme de chaîne de caractères.")
        
        # v est le chemin depuis .env, ex: "E1_GestionDonnees\Base_de_donnees\france_optique.db"
        # Ce chemin est relatif à WORKSPACE_ROOT.
        db_path = (WORKSPACE_ROOT / v).resolve()
        return f"sqlite:///{db_path}"

    class Config:
        env_file = ".env"
        # Pydantic va chercher .env par rapport au répertoire de travail actuel (CWD).
        # Si l'application est lancée depuis E1_GestionDonnees/api/, il chargera E1_GestionDonnees/api/.env.

settings = Settings() 