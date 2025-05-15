"""
Fixtures partagées pour les tests de l'API E3_MettreDispositionIA.
Ce module contient des configurations et des fixtures utilisées par tous les tests.
"""

import os
import sys
import pytest
import shutil
import tempfile
from PIL import Image

# Ajouter le chemin parent aux chemins de recherche Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def test_image_dir():
    """Crée un répertoire temporaire pour les images de test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Nettoyer après les tests
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_image_path(test_image_dir):
    """Crée une image de test et retourne son chemin."""
    # Création d'une image test simple en blanc
    test_img_path = os.path.join(test_image_dir, "test_image.png")
    
    # Créer une image simple (carré blanc 100x100)
    img = Image.new('L', (100, 100), 255)
    img.save(test_img_path)
    
    return test_img_path

@pytest.fixture(scope="session")
def test_reference_images(test_image_dir):
    """Crée un ensemble d'images de référence pour les tests."""
    ref_dir = os.path.join(test_image_dir, "references")
    os.makedirs(ref_dir, exist_ok=True)
    
    # Créer quelques images de référence avec différentes classes
    classes = ["classe_test_1", "classe_test_2", "classe_test_3"]
    image_paths = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(ref_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Créer 3 images par classe
        for j in range(3):
            # Créer des images avec des valeurs différentes pour simuler des classes
            value = 100 + i * 50
            img = Image.new('L', (100, 100), value)
            img_path = os.path.join(class_dir, f"{class_name}_{j}.png")
            img.save(img_path)
            image_paths.append(img_path)
    
    return image_paths 