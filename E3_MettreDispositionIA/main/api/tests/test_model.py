"""
Tests pour les fonctionnalités du modèle d'IA de l'API E3_MettreDispositionIA.
Ce module contient des tests pour le chargement du modèle, la génération d'embeddings
et la recherche de similarités.
"""

import pytest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
import io
from PIL import Image

# Ajouter le chemin parent aux chemins de recherche Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import des modules du modèle
from app.model_loader import load_model, preprocess_image, get_embedding
from app.similarity_search import get_top_matches, load_references

# Utilisation des fixtures du conftest.py
@pytest.fixture
def mock_model():
    """Crée un mock du modèle d'IA"""
    model = MagicMock()
    # Configurer le comportement du modèle pour qu'il retourne un embedding
    model.return_value = np.random.random((1, 256))
    return model

@pytest.fixture
def mock_references():
    """Crée des embeddings de référence pour les tests"""
    # Simuler 5 classes avec 3 images chacune
    classes = ["classe_1", "classe_2", "classe_3", "classe_4", "classe_5"]
    refs = []
    
    for cls in classes:
        for i in range(3):
            # Créer des embeddings aléatoires mais cohérents par classe
            base = np.random.random(256)
            # Ajouter un peu de bruit pour simuler des variations dans la même classe
            noise = np.random.normal(0, 0.1, 256)
            embedding = base + noise
            # Normaliser l'embedding
            embedding = embedding / np.linalg.norm(embedding)
            refs.append({
                "class": cls,
                "embedding": embedding
            })
    
    return refs

# Tests du chargement du modèle
def test_load_model():
    """Test du chargement du modèle"""
    with patch("app.model_loader.torch.load") as mock_load, \
         patch("app.model_loader.EfficientNetTriplet") as mock_model_class:
        
        # Configurer les mocks
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance
        mock_load.return_value = {"state_dict": {}}
        
        # Appeler la fonction à tester
        model = load_model()
        
        # Vérifier que le modèle a été correctement chargé
        assert model is mock_model_instance
        mock_model_instance.load_state_dict.assert_called_once()
        mock_model_instance.eval.assert_called_once()

# Tests du prétraitement des images
def test_preprocess_image():
    """Test du prétraitement des images"""
    # Créer une image test
    img = Image.new('L', (100, 100), 255)
    
    # Appeler la fonction de prétraitement
    with patch("app.model_loader.transforms.Compose") as mock_compose:
        # Configurer le mock pour qu'il retourne un mock de la fonction de transformation
        transform_mock = MagicMock()
        mock_compose.return_value = transform_mock
        transform_mock.return_value = np.zeros((3, 224, 224))
        
        # Appeler la fonction à tester
        preprocessed = preprocess_image(img)
        
        # Vérifier que la sortie a la forme attendue
        assert transform_mock.called
        assert preprocessed.shape == (3, 224, 224)

# Tests de la génération d'embeddings
def test_get_embedding(mock_model):
    """Test de la génération d'embeddings"""
    # Créer une image test
    img = Image.new('L', (100, 100), 255)
    
    with patch("app.model_loader.preprocess_image") as mock_preprocess, \
         patch("app.model_loader.torch.no_grad"):
        
        # Configurer les mocks
        mock_preprocess.return_value = np.zeros((3, 224, 224))
        
        # Appeler la fonction à tester
        embedding = get_embedding(mock_model, img)
        
        # Vérifier que l'embedding a la forme attendue
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (256,)

# Tests de la recherche de similarités
def test_get_top_matches(mock_references):
    """Test de la recherche de correspondances similaires"""
    with patch("app.similarity_search.reference_embeddings", mock_references):
        # Créer un embedding de test similaire à la classe_1
        test_embedding = mock_references[0]["embedding"].copy()
        
        # Ajouter un peu de bruit
        noise = np.random.normal(0, 0.05, 256)
        test_embedding += noise
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Rechercher les correspondances
        matches = get_top_matches(test_embedding, n=5)
        
        # Vérifier les résultats
        assert len(matches) == 5
        assert "class" in matches[0]
        assert "similarity" in matches[0]
        assert matches[0]["class"] == "classe_1"  # La première correspondance devrait être de la classe_1
        assert matches[0]["similarity"] > 0.9  # La similarité devrait être élevée

def test_load_references():
    """Test du chargement des références"""
    mock_model = MagicMock()
    
    with patch("app.similarity_search.os.walk") as mock_walk, \
         patch("app.similarity_search.Image.open") as mock_open, \
         patch("app.similarity_search.get_embedding") as mock_get_embedding:
        
        # Configurer les mocks
        mock_walk.return_value = [
            ("root/references", ["classe_1", "classe_2"], []),
            ("root/references/classe_1", [], ["img1.png", "img2.png"]),
            ("root/references/classe_2", [], ["img1.png", "img2.png"]),
        ]
        
        mock_image = MagicMock()
        mock_open.return_value = mock_image
        mock_image.convert.return_value = mock_image
        
        # Mock pour get_embedding
        mock_get_embedding.return_value = np.random.random(256)
        
        # Appeler la fonction à tester
        load_references(mock_model)
        
        # Vérifier que les références ont été chargées
        from app.similarity_search import reference_embeddings
        
        assert len(reference_embeddings) == 4  # 2 classes x 2 images
        assert "class" in reference_embeddings[0]
        assert "embedding" in reference_embeddings[0]
        assert isinstance(reference_embeddings[0]["embedding"], np.ndarray)

def test_similarity_calculation():
    """Test du calcul de similarité"""
    # Créer deux embeddings avec une similarité connue
    embedding1 = np.array([1, 0, 0, 0])  # Vecteur unitaire dans la direction x
    embedding2 = np.array([0.707, 0.707, 0, 0])  # Vecteur unitaire à 45°
    
    # La similarité cosinus devrait être le cosinus de l'angle (45°) = 0.707
    with patch("app.similarity_search.reference_embeddings", [
        {"class": "test", "embedding": embedding2}
    ]):
        matches = get_top_matches(embedding1, n=1)
        
        # Vérifier la similarité calculée (avec une tolérance pour les erreurs d'arrondi)
        assert len(matches) == 1
        assert abs(matches[0]["similarity"] - 0.707) < 0.01 