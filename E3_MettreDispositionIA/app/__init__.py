import os
import sys
import torch
from .model import load_model

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer la configuration
from config import (
    DEFAULT_MODEL_PATH, 
    EMBEDDINGS_PATH, 
    PROCESSED_DATA_DIR
)

# Vérifier si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialiser les variables globales (seront définies lors du démarrage de l'application)
model = None
embeddings_manager = None

def init_app():
    """
    Initialise l'application en chargeant le modèle et les embeddings
    Appelé au démarrage de l'application FastAPI
    """
    
    global model, embeddings_manager
    
    from .embeddings_manager import EmbeddingsManager
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {DEFAULT_MODEL_PATH}")
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = load_model(DEFAULT_MODEL_PATH, device=device)
        print("Modèle chargé avec succès")
    else:
        # Créer un modèle non entraîné pour la démonstration
        from .model import SiameseNetwork
        model = SiameseNetwork()
        print("Modèle d'exemple créé (non entraîné)")
    
    # Charger les embeddings
    print(f"Chargement des embeddings depuis {EMBEDDINGS_PATH}")
    embeddings_manager = EmbeddingsManager(embeddings_path=EMBEDDINGS_PATH)
    
    # Si aucun embedding n'a été chargé, créer des embeddings factices pour la démonstration
    if not embeddings_manager.embeddings_dict:
        print("Aucun embedding trouvé. Création d'embeddings factices pour la démonstration.")
        # Créer quelques embeddings factices
        import numpy as np
        embeddings_manager.add_embedding(1, np.random.rand(128), {'code': 'Varilux', 'indice': 1.67, 'filename': 'varilux_1.67.jpg'})
        embeddings_manager.add_embedding(2, np.random.rand(128), {'code': 'Essilor', 'indice': 1.6, 'filename': 'essilor_1.6.jpg'})
        embeddings_manager.add_embedding(3, np.random.rand(128), {'code': 'Zeiss', 'indice': 1.5, 'filename': 'zeiss_1.5.jpg'})
        
        # Sauvegarder les embeddings factices
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        embeddings_manager.save_embeddings(EMBEDDINGS_PATH)
    
    print(f"Application initialisée avec {len(embeddings_manager.embeddings_dict)} embeddings de gravures") 