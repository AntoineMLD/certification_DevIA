"""
Configuration centralisée pour l'API

Centralise les paramètres de configuration de l'API pour faciliter
la maintenance et les modifications.
"""

import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration de l'API
API_VERSION = "1.0.0"
API_TITLE = "API de Classification d'IA"
API_DESCRIPTION = "API REST pour accéder aux fonctionnalités du modèle d'IA de classification d'images"

# Configuration de sécurité
SECRET_KEY = os.getenv("SECRET_KEY", "")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
ROTATION_THRESHOLD_MINUTES = int(os.getenv("ROTATION_THRESHOLD_MINUTES", "25"))

# Configuration d'authentification
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_DIR = os.path.join(BASE_DIR, "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "oversampled_gravures")
DB_PATH = os.getenv("DB_PATH", os.path.abspath(os.path.join(BASE_DIR, "../../../E1_GestionDonnees/Base_de_donnees/france_optique.db")))

# Configuration du modèle
MODEL_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "efficientnet_triplet.pth")
IMAGE_SIZE = 224

# Configuration du monitoring
LOG_DIR = os.path.join(BASE_DIR, "../logs")
REPORTS_DIR = os.path.join(LOG_DIR, "reports")

# Création des répertoires nécessaires
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
