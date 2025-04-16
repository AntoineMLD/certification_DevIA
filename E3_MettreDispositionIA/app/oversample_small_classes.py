#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour suréchantillonner les classes avec peu d'images
en générant plus d'images augmentées pour ces classes.
"""

import os
import sys
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_strong_augmentation_pipeline():
    """Crée un pipeline d'augmentation plus intensif pour les classes avec peu d'images"""
    return A.Compose([
        # Rotations plus importantes
        A.Rotate(limit=30, p=0.7),
        
        # Translations et mises à l'échelle
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.6),
        
        # Variations d'épaisseur du trait
        A.GaussianBlur(blur_limit=(1, 3), p=0.4),
        
        # Variations de contraste
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Déformations élastiques
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=8, p=0.5),
        
        # Distorsion perspective
        A.Perspective(scale=(0.05, 0.15), p=0.4),
        
        # Variations d'échelle
        A.RandomScale(scale_limit=0.15, p=0.5),
        
        # Distorsion optique
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
        
        # Grille de distorsion
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    ])

def augment_image(image, transform, num_augmentations=15):
    """Applique les augmentations à une image"""
    augmented_images = []
    
    # Ajoute l'image originale
    augmented_images.append(image)
    
    # Applique les augmentations
    for _ in range(num_augmentations):
        augmented = transform(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images

def process_small_class(input_dir, output_dir, class_name, min_images=20, num_augmentations=15):
    """Traite une classe avec peu d'images pour générer plus d'augmentations"""
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    
    # Crée le répertoire de sortie
    os.makedirs(class_output_path, exist_ok=True)
    
    # Transformations fortes
    transform = create_strong_augmentation_pipeline()
    
    # Liste les images de la classe
    image_files = [f for f in os.listdir(class_input_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Suréchantillonnage de la classe '{class_name}' avec {len(image_files)} images originales")
    
    # Détermine combien d'augmentations faire par image pour atteindre min_images
    images_needed = max(0, min_images - len(image_files))
    if len(image_files) == 0:
        print(f"  Erreur: Aucune image trouvée dans {class_input_path}")
        return
    
    # Traite chaque image
    for img_file in image_files:
        try:
            # Charge l'image
            img_path = os.path.join(class_input_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"  Erreur: Impossible de charger l'image {img_file}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Applique les augmentations
            augmented_images = augment_image(image, transform, num_augmentations)
            
            # Sauvegarde les images
            base_name = os.path.splitext(img_file)[0]
            for i, aug_img in enumerate(augmented_images):
                output_path = os.path.join(class_output_path, f"{base_name}_aug_{i}.png")
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, aug_img)
            
            print(f"  - {img_file}: {len(augmented_images)} images générées")
                
        except Exception as e:
            print(f"  - Erreur lors du traitement de {img_file}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Suréchantillonne les classes avec peu d'images")
    parser.add_argument("--raw_dir", type=str, default="data/raw_gravures",
                       help="Répertoire contenant les images brutes de gravures")
    parser.add_argument("--augmented_dir", type=str, default="data/augmented_gravures",
                       help="Répertoire où se trouvent les images augmentées")
    parser.add_argument("--min_threshold", type=int, default=10,
                       help="Nombre minimum d'images par classe à suréchantillonner")
    parser.add_argument("--target_count", type=int, default=20,
                       help="Nombre d'images cible par classe après suréchantillonnage")
    parser.add_argument("--augmentations_per_image", type=int, default=15,
                       help="Nombre d'augmentations à générer par image pour les petites classes")
    
    args = parser.parse_args()
    
    # Vérifier les répertoires
    if not os.path.exists(args.raw_dir):
        print(f"Erreur: Le répertoire {args.raw_dir} n'existe pas!")
        return
    
    if not os.path.exists(args.augmented_dir):
        print(f"Erreur: Le répertoire {args.augmented_dir} n'existe pas!")
        os.makedirs(args.augmented_dir)
    
    # Recenser toutes les classes et compter les images
    classes = {}
    small_classes = []
    
    for class_dir in os.listdir(args.raw_dir):
        class_path = os.path.join(args.raw_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        # Compte les images
        image_count = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        classes[class_dir] = image_count
        
        # Identifie les classes avec peu d'images
        if image_count < args.min_threshold:
            small_classes.append((class_dir, image_count))
    
    # Affiche les statistiques
    print(f"Analyse du répertoire: {args.raw_dir}")
    print(f"Nombre total de classes: {len(classes)}")
    print(f"Classes avec moins de {args.min_threshold} images:")
    
    if not small_classes:
        print("  Aucune classe n'a moins de {args.min_threshold} images.")
        return
    
    for class_name, count in small_classes:
        print(f"  - {class_name}: {count} images")
    
    # Suréchantillonne les classes avec peu d'images
    print("\nDébut du suréchantillonnage...")
    
    for class_name, count in small_classes:
        process_small_class(
            args.raw_dir, 
            args.augmented_dir, 
            class_name, 
            min_images=args.target_count, 
            num_augmentations=args.augmentations_per_image
        )
    
    print("\nSuréchantillonnage terminé!")
    
    # Compte les images dans le répertoire augmenté
    augmented_counts = {}
    for class_dir in os.listdir(args.augmented_dir):
        class_path = os.path.join(args.augmented_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        image_count = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        augmented_counts[class_dir] = image_count
    
    print("\nRésultats après suréchantillonnage:")
    for class_name, count in small_classes:
        original_count = classes[class_name]
        new_count = augmented_counts.get(class_name, 0)
        print(f"  - {class_name}: {original_count} → {new_count} images (+{new_count - original_count})")

if __name__ == "__main__":
    main() 