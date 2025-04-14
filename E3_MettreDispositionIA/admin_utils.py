#!/usr/bin/env python
"""
Utilitaires d'administration pour les embeddings et les modèles.
Ce script combine les fonctionnalités de:
- check_embeddings.py: vérification des embeddings
- regenerate_embeddings_with_metadata.py: régénération des embeddings avec métadonnées
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import pickle
import argparse
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model import load_model
from app.embeddings_manager import EmbeddingsManager
from config import (
    PROCESSED_DATA_DIR,
    DEFAULT_MODEL_PATH,
    BEST_MODEL_PATH,
    EMBEDDINGS_PATH,
    IMAGE_SIZE
)

def preprocess_image(image_path):
    """
    Prétraite une image pour le modèle
    
    Args:
        image_path: Chemin vers l'image à prétraiter
        
    Returns:
        numpy.ndarray: Image prétraitée
    """
    image = Image.open(image_path).convert('L')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image) / 255.0
    return img_array

def get_class_from_filename(filename):
    """
    Extrait la classe à partir du nom de fichier
    Format attendu: classe_numero.extension
    
    Args:
        filename: Nom du fichier
        
    Returns:
        str: Classe extraite du nom de fichier
    """
    base_name = os.path.basename(filename)
    if '_' in base_name:
        return base_name.split('_')[0]
    return 'inconnu'

def check_embeddings():
    """
    Vérifie les embeddings existants
    """
    print(f"Vérification des embeddings dans {EMBEDDINGS_PATH}")
    
    # Vérifier si le fichier existe
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Erreur: Le fichier {EMBEDDINGS_PATH} n'existe pas.")
        return
    
    # Charger le gestionnaire d'embeddings
    try:
        embeddings_manager = EmbeddingsManager()
        embeddings_manager.load_embeddings(EMBEDDINGS_PATH)
        print(f"Nombre total d'embeddings: {len(embeddings_manager.embeddings_dict)}")
        
        # Afficher des statistiques sur les classes
        classes = {}
        for id_gravure, info in embeddings_manager.gravures_info.items():
            classe = info.get('code', 'inconnu')
            if classe not in classes:
                classes[classe] = 0
            classes[classe] += 1
        
        print("\nDistribution des classes:")
        for classe, count in classes.items():
            print(f"  - {classe}: {count} images")
        
        # Afficher des informations sur quelques embeddings
        print("\nExemples d'embeddings:")
        for i, (id_gravure, embedding) in enumerate(list(embeddings_manager.embeddings_dict.items())[:3]):
            info = embeddings_manager.gravures_info.get(id_gravure, {})
            print(f"  - ID {id_gravure}: {info.get('code', 'inconnu')}, "
                  f"Indice: {info.get('indice', 'N/A')}, "
                  f"Shape: {embedding.shape}")
            
    except Exception as e:
        print(f"Erreur lors du chargement des embeddings: {e}")

def regenerate_embeddings(model_path=None):
    """
    Régénère les embeddings pour toutes les images dans le répertoire de données
    
    Args:
        model_path: Chemin vers le modèle à utiliser (par défaut: DEFAULT_MODEL_PATH)
    """
    # Utiliser le modèle par défaut si aucun n'est spécifié
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    print(f"Régénération des embeddings avec le modèle {model_path}")
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle {model_path} n'existe pas.")
        return
    
    # Vérifier si le répertoire de données existe
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Erreur: Le répertoire de données {PROCESSED_DATA_DIR} n'existe pas.")
        return
    
    # Créer le répertoire d'embeddings s'il n'existe pas
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    
    # Charger le modèle
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(model_path, device=device)
        model.eval()
        print(f"Modèle chargé avec succès sur {device}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Initialiser le gestionnaire d'embeddings
    embeddings_manager = EmbeddingsManager()
    
    # Parcourir les fichiers d'images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    counter = 1
    
    print(f"Recherche d'images dans {PROCESSED_DATA_DIR}")
    for filename in os.listdir(PROCESSED_DATA_DIR):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(PROCESSED_DATA_DIR, filename)
            
            try:
                # Extraire les métadonnées du nom de fichier
                classe = get_class_from_filename(filename)
                
                # Prétraiter l'image
                image_array = preprocess_image(file_path)
                
                # Convertir en tensor
                image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
                if device.type == 'cuda':
                    image_tensor = image_tensor.cuda()
                
                # Calculer l'embedding
                with torch.no_grad():
                    embedding = model.forward_one(image_tensor).cpu().numpy()[0]
                
                # Ajouter l'embedding au gestionnaire
                embeddings_manager.add_embedding(
                    counter, 
                    embedding, 
                    {
                        'code': classe,
                        'indice': 0.0,  # Valeur par défaut
                        'filename': filename
                    }
                )
                
                if counter % 10 == 0:
                    print(f"Traitement de l'image {counter}: {filename}")
                
                counter += 1
                
            except Exception as e:
                print(f"Erreur lors du traitement de {filename}: {e}")
    
    # Sauvegarder les embeddings
    try:
        embeddings_manager.save_embeddings(EMBEDDINGS_PATH)
        print(f"Embeddings sauvegardés dans {EMBEDDINGS_PATH}")
        print(f"Total: {counter-1} images traitées")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des embeddings: {e}")

def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(description="Utilitaires d'administration pour les embeddings")
    parser.add_argument('action', choices=['check', 'regenerate'], 
                        help="Action à effectuer: check (vérifier les embeddings) ou "
                             "regenerate (régénérer les embeddings)")
    parser.add_argument('--model', choices=['default', 'best'], default='default',
                        help="Modèle à utiliser pour la régénération: default ou best")
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_embeddings()
    elif args.action == 'regenerate':
        model_path = BEST_MODEL_PATH if args.model == 'best' else DEFAULT_MODEL_PATH
        regenerate_embeddings(model_path)

if __name__ == "__main__":
    main() 