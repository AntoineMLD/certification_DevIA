import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

def create_augmentation_pipeline(): 
    """
    Définit les transformations réalistes à appliquer aux gravures dessinées
    """
    return A.Compose([
        A.Rotate(limit=15, p=0.5),  # rotation aléatoire jusqu’à ±15°
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.4),
        A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.ElasticTransform(alpha=100, sigma=5, alpha_affine=5, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.2),
        A.RandomScale(scale_limit=0.1, p=0.3), 
    ])

def augment_image(image: np.ndarray, transform: A.BasicTransform, num_augmentations: int = 5) -> list:
    """
    Applique des augmentations aléatoires à une image.

    Args:
        image: L'image originale au format numpy (RGB ou grayscale)
        transform: Le pipeline d'augmentation albumentations 
        num_augmentations: Nombre d'exemplaire à générer

    Returns:
        Liste des images augmentées (incluant l'originale en position 0)
    """

    augmented_images = [image]

    for _ in range(num_augmentations):
        augmented = transform(image=image)['image']
        augmented_images.append(augmented)

    return augmented_images


def process_directory(input_dir: str, output_dir: str, num_augmentations: int = 5):
    """
    Applique les augmentations à toutes les images du dossier source.

    Args:
        input_dir: Chemin du dossier contenant les images originales
        output_dir: Chemin du dossier où enregistrer les images augmentées
        num_augmentations: Nombre d'exemplaires à générer pour chaque image
    """
    transform = create_augmentation_pipeline()
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_path):
            print(f"Le dossier {class_input_path} n'existe pas. Passé.")
            continue

        os.makedirs(class_output_path, exist_ok=True)
        print(f"Traitement de la classe: {class_name}")

        for img_name in os.listdir(class_input_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_input_path, img_name)

            try:
                #lecture image en RGB avec cv2
                image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Erreur lors de la lecture de l'image: {img_path}")
                    continue
                
                #appliquer les augmentations
                augmented_images = augment_image(image, transform, num_augmentations)

                base_name = os.path.splitext(img_name)[0]

                for i, aug_img in enumerate(augmented_images):
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    output_filename = f"{base_name}_aug_{i}.jpg"
                    output_path = os.path.join(class_output_path, output_filename)

                    cv2.imencode('.jpg', aug_img_bgr)[1].tofile(output_path)
                    print(f"Image augmentée enregistrée: {output_path}")

            except Exception as e:
                print(f"Erreur lors de l'augmentation de l'image {img_name}: {e}")

        print(f"Traitement de la classe {class_name} terminé.")

    print("Toutes les classes ont été traitées avec succès.")
                
    
if __name__ == "__main__":
    input_dir = "data/raw_gravures"
    output_dir = "data/augmented_gravures"
    process_directory(input_dir, output_dir, num_augmentations=5)