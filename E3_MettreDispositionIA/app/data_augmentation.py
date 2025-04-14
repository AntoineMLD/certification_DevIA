import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
from pathlib import Path

def create_augmentation_pipeline():
    """Crée le pipeline d'augmentation de données adapté aux dessins manuels"""
    return A.Compose([
        # Légères rotations pour simuler les variations de dessin
        A.Rotate(limit=5, p=0.3),
        
        # Très légères translations pour simuler les variations de position
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.3),
        
        # Légères variations d'épaisseur du trait
        A.GaussianBlur(blur_limit=(1, 2), p=0.2),
        
        # Légères variations de contraste pour simuler différentes pressions du stylo
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        
        # Légères déformations élastiques pour simuler les variations naturelles du dessin
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.2),
    ])

def augment_image(image, transform, num_augmentations=3):
    """Applique les augmentations à une image"""
    augmented_images = []
    
    # Ajoute l'image originale
    augmented_images.append(image)
    
    # Applique les augmentations
    for _ in range(num_augmentations):
        augmented = transform(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images

def process_directory(input_dir, output_dir, num_augmentations=3):
    """Traite tous les fichiers d'un répertoire"""
    # Crée le pipeline d'augmentation
    transform = create_augmentation_pipeline()
    
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Parcourt tous les sous-répertoires
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        # Crée le répertoire de sortie pour cette classe
        output_class_dir = os.path.join(output_dir, class_dir)
        os.makedirs(output_class_dir, exist_ok=True)
        
        print(f"Traitement du dossier {class_dir}...")
        
        # Traite chaque image
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                # Charge l'image avec gestion des chemins
                img_path = os.path.join(class_path, img_file)
                image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if image is None:
                    print(f"  - Erreur: Impossible de charger l'image {img_file}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Applique les augmentations
                augmented_images = augment_image(image, transform, num_augmentations)
                
                # Sauvegarde les images
                base_name = os.path.splitext(img_file)[0]
                for i, aug_img in enumerate(augmented_images):
                    output_path = os.path.join(output_class_dir, f"{base_name}_aug_{i}.png")
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    cv2.imencode('.png', aug_img)[1].tofile(output_path)
                
                print(f"  - {img_file}: {len(augmented_images)} images générées")
                
            except Exception as e:
                print(f"  - Erreur lors du traitement de {img_file}: {str(e)}")
                continue

def main():
    # Chemins des répertoires
    input_dir = "data/raw_gravures"
    output_dir = "data/augmented_gravures"
    
    # Vérifie que le répertoire d'entrée existe
    if not os.path.exists(input_dir):
        print(f"Erreur: Le répertoire {input_dir} n'existe pas!")
        return
    
    print("Début de l'augmentation des données...")
    process_directory(input_dir, output_dir)
    print("Augmentation des données terminée!")

if __name__ == "__main__":
    main() 