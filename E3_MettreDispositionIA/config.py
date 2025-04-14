"""
Configuration centrale du projet E3_MettreDispositionIA.
Ce fichier contient toutes les constantes et paramètres de configuration utilisés dans le projet.
"""
import os
from pathlib import Path

# Chemins principaux
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'model'
EMBEDDINGS_DIR = PROJECT_ROOT / 'embeddings'

# Noms de fichiers
DEFAULT_MODEL_FILE = "siamese_model.pt"
BEST_MODEL_FILE = "best_siamese_model.pt"
EMBEDDINGS_FILE = "gravures_embeddings.pkl"

# Chemins complets
DEFAULT_MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_FILE
BEST_MODEL_PATH = MODEL_DIR / BEST_MODEL_FILE
EMBEDDINGS_PATH = EMBEDDINGS_DIR / EMBEDDINGS_FILE

# Paramètres d'API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Paramètres de sécurité
# En production, modifier ces valeurs via des variables d'environnement
SECRET_KEY = os.environ.get("SECRET_KEY", "clef_secrete_a_changer_en_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Utilisateurs par défaut (pour la démonstration uniquement)
DEFAULT_USERS = {
    "utilisateur": {
        "username": "utilisateur",
        "hashed_password": "$2b$12$SIr.uOBJfaxNlWCLjYJJ9.xo2oWYLLT5LevIJz15gQ1Rz/FxLN4jS",  # password123
        "disabled": False
    }
}

# Paramètres de l'interface de dessin
CANVAS_WIDTH = 300
CANVAS_HEIGHT = 300
DEFAULT_BRUSH_SIZE = 5

# Paramètres du modèle
IMAGE_SIZE = 64  # Taille des images attendue par le modèle (64x64)
EMBEDDING_DIM = 128  # Dimension de l'embedding
BATCH_SIZE = 16
NUM_EPOCHS = 10

# Vérifier et créer les répertoires si nécessaires
def ensure_directories():
    """Vérifie et crée les répertoires nécessaires au projet."""
    for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, EMBEDDINGS_DIR]:
        directory.mkdir(exist_ok=True, parents=True)

# Exécuter au moment de l'importation du module
ensure_directories() 