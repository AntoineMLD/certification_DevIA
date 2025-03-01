import secrets
import base64
import os
from pathlib import Path

def generate_secret_key():
    """
    Genere une cle secrete securisee
    """
    # Generer 32 bytes aleatoires et les encoder en base64
    return base64.b64encode(secrets.token_bytes(32)).decode()

if __name__ == "__main__":
    # Obtenir le chemin du dossier courant
    current_dir = Path(__file__).parent.absolute()
    env_path = current_dir / ".env"

    # Generer et afficher la cle
    secret_key = generate_secret_key()
    print("\nVotre nouvelle cle secrete :")
    print("-----------------------------")
    print(secret_key)
    print("-----------------------------")
    
    # Creer ou mettre a jour le fichier .env
    with open(env_path, "w") as f:
        f.write("""# JWT Security Configuration
SECRET_KEY="{}"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database Configuration
DATABASE_URL="sqlite:///./Base_de_donnees/france_optique.db"

# API Configuration
API_VERSION="1.0"
API_TITLE="API France Optique"
API_DESCRIPTION="API for optical lens management"
API_PREFIX="/api/v1"

# Admin Configuration
ADMIN_EMAIL="admin@france-optique.com"
ADMIN_PASSWORD="admin123!@#"

# Server Configuration
HOST="0.0.0.0"
PORT=8000
DEBUG=True
""".format(secret_key)) 