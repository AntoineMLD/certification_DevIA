import os
import sys
import argparse
import glob
from PIL import Image
import numpy as np
import shutil

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer la configuration
from config import DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE

def preprocess_engraving_image(image_path, output_path, size=IMAGE_SIZE):
    """
    Prétraite une image de gravure pour l'entraînement
    
    Args:
        image_path: Chemin vers l'image d'entrée
        output_path: Chemin où sauvegarder l'image prétraitée
        size: Taille de l'image de sortie (par défaut: 64x64)
    """
    try:
        # Ouvrir l'image
        image = Image.open(image_path)
        
        # Convertir en niveaux de gris
        image = image.convert('L')
        
        # Redimensionner
        image = image.resize((size, size))
        
        # Sauvegarder l'image prétraitée
        image.save(output_path)
        
        return True
    except Exception as e:
        print(f"Erreur lors du prétraitement de {image_path}: {e}")
        return False

def process_directory(input_dir, output_dir, size=IMAGE_SIZE):
    """
    Prétraite toutes les images d'un répertoire
    
    Args:
        input_dir: Répertoire contenant les images d'entrée
        output_dir: Répertoire où sauvegarder les images prétraitées
        size: Taille des images de sortie (par défaut: 64x64)
    
    Returns:
        int: Nombre d'images traitées avec succès
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Trouver toutes les images
    success_count = 0
    
    # Parcourir tous les sous-répertoires
    for root, dirs, files in os.walk(input_dir):
        # Obtenir le chemin relatif par rapport au répertoire d'entrée
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == '.':
            continue
            
        # Créer le répertoire de sortie correspondant
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Traiter chaque image du répertoire courant
        for img_file in files:
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Chemins d'entrée et de sortie
            input_path = os.path.join(root, img_file)
            output_path = os.path.join(output_subdir, img_file)
            
            # Prétraiter l'image
            if preprocess_engraving_image(input_path, output_path, size):
                success_count += 1
                print(f"Prétraitement réussi pour {os.path.join(rel_path, img_file)}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Prétraite des images de gravures pour l'entraînement")
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR), 
                        help=f"Répertoire contenant les sous-dossiers d'images (défaut: {DATA_DIR})")
    parser.add_argument('--output_dir', type=str, default=str(PROCESSED_DATA_DIR), 
                        help=f"Répertoire où sauvegarder les images prétraitées (défaut: {PROCESSED_DATA_DIR})")
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, 
                        help=f"Taille des images prétraitées (défaut: {IMAGE_SIZE})")
    
    args = parser.parse_args()
    
    # Vérifier si le répertoire de données existe
    if not os.path.exists(args.data_dir):
        print(f"Erreur: Le répertoire {args.data_dir} n'existe pas.")
        return
    
    # Nettoyer le répertoire de sortie s'il existe
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    # Traiter les images
    total_processed = process_directory(args.data_dir, args.output_dir, args.image_size)
    print(f"\nPrétraitement terminé. {total_processed} images traitées au total.")

if __name__ == "__main__":
    main() 