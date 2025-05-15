"""
Tests des modules internes de l'API E3_MettreDispositionIA.
Ce fichier teste les fonctions internes sans passer par les endpoints HTTP.
"""

import pytest
import sys
import os
import io
import numpy as np
from unittest.mock import patch, MagicMock
import jwt
from datetime import datetime, timedelta
from PIL import Image
from fastapi import HTTPException

# Ajouter le chemin parent aux chemins de recherche Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Variables de test
TEST_USERNAME = "admin@example.com"
TEST_PASSWORD = "admin_password"
TEST_SECRET_KEY = "test_secret_key"

# Import des modules à tester
try:
    from app.security import (
        create_access_token,
        verify_token,
        validate_image_file,
        log_security_event,
        TOKEN_SETTINGS
    )
    
    from app.model_loader import (
        load_model,
        preprocess_image,
        get_embedding
    )
    
    from app.similarity_search import (
        get_top_matches,
        load_references
    )
    
    # Si on arrive ici, tous les imports fonctionnent
    IMPORTS_OK = True
except ImportError as e:
    print(f"Erreur d'import: {e}")
    IMPORTS_OK = False

# Test simple pour vérifier que les imports fonctionnent
def test_imports():
    """Vérifie que tous les imports fonctionnent"""
    assert IMPORTS_OK, "Les imports ont échoué"

# Fixtures pour les tests
@pytest.fixture
def test_image():
    """Crée une image de test en mémoire"""
    img = Image.new('L', (100, 100), color=255)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture(autouse=True)
def mock_security_config():
    """Mock les configurations de sécurité"""
    if IMPORTS_OK:
        with patch.dict("app.security.TOKEN_SETTINGS", {"SECRET_KEY": TEST_SECRET_KEY, "ALGORITHM": "HS256"}):
            yield
    else:
        yield

# ====== Tests des fonctions de sécurité ======

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_create_access_token():
    """Test de la création d'un token d'accès"""
    # Création d'un token
    with patch.dict("app.security.token_versions", {}):
        token, version = create_access_token(TEST_USERNAME)
        
        # Vérification que le token est une chaîne non vide
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Décodage du token pour vérifier son contenu
        payload = jwt.decode(token, TEST_SECRET_KEY, algorithms=["HS256"])
        
        # Vérification des champs standards
        assert "sub" in payload
        assert payload["sub"] == TEST_USERNAME
        assert "exp" in payload
        
        # Vérification de la présence du numéro de version
        assert "token_version" in payload
        assert payload["token_version"] == version

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_verify_token():
    """Test de la vérification d'un token"""
    # Création d'un token valide
    with patch.dict("app.security.token_versions", {}):
        token, _ = create_access_token(TEST_USERNAME)
        
        # Vérification du token
        token_data, should_rotate = verify_token(token)
        
        # Vérification des données du token
        assert token_data.email == TEST_USERNAME
        
        # Le token est neuf, donc pas besoin de rotation
        assert should_rotate is False

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_verify_expired_token():
    """Test de la vérification d'un token expiré"""
    # Création d'un payload avec une date d'expiration passée
    exp_time = datetime.utcnow() - timedelta(minutes=30)
    payload = {
        "sub": TEST_USERNAME,
        "exp": exp_time.timestamp(),
        "iat": (exp_time - timedelta(minutes=15)).timestamp(),
        "jti": "test_id",
        "token_version": 1
    }
    
    # Création d'un token expiré
    with patch("jwt.decode", side_effect=jwt.ExpiredSignatureError):
        # Vérification que l'exception est levée
        with pytest.raises(HTTPException) as excinfo:
            verify_token("expired_token")
        
        assert excinfo.value.status_code == 401
        assert "Token has expired" in excinfo.value.detail

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_validate_image_file():
    """Test de la validation des fichiers image"""
    # Données valides (en-tête PNG)
    valid_png_header = b'\x89PNG\r\n\x1a\n'
    valid_png_data = valid_png_header + b'\x00' * 100  # Ajouter des données fictives
    
    # Données invalides
    invalid_data = b'not an image file'
    
    # Test avec des données valides
    assert validate_image_file(valid_png_data) is True
    
    # Test avec des données invalides
    assert validate_image_file(invalid_data) is False

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_log_security_event():
    """Test de la journalisation des événements de sécurité"""
    # Créer un mock direct pour le logger
    mock_logger = MagicMock()
    
    with patch("app.security.security_logger", mock_logger):
        # Test pour différents niveaux de log
        log_security_event("TEST_EVENT", "Test message")
        mock_logger.info.assert_called_once()
        
        log_security_event("WARNING_EVENT", "Test warning", "WARNING")
        mock_logger.warning.assert_called_once()
        
        log_security_event("ERROR_EVENT", "Test error", "ERROR")
        mock_logger.error.assert_called_once()

# ====== Tests des fonctions du modèle ======

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_load_model():
    """Test du chargement du modèle"""
    # Créer un mock pour torch.load
    mock_state_dict = {"test_layer": MagicMock()}
    
    # Créer un mock pour le modèle EfficientNetEmbedding
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None
    
    with patch("app.model_loader.torch.load", return_value=mock_state_dict), \
         patch("app.model_loader.EfficientNetEmbedding", return_value=mock_model):
        
        # Appeler la fonction à tester
        model = load_model()
        
        # Vérifier que le modèle a été correctement chargé
        assert model is mock_model
        mock_model.load_state_dict.assert_called_once_with(mock_state_dict)
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_preprocess_image():
    """Test du prétraitement des images"""
    # Créer une image test
    img = Image.new('L', (100, 100), 255)
    
    # Créer un mock du tensor pour le résultat
    mock_tensor = MagicMock()
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor
    
    # Remplacer la transformation par un mock
    with patch("app.model_loader.transform", return_value=mock_tensor):
        # Appeler la fonction à tester
        result = preprocess_image(img)
        
        # Vérifier que le résultat est le tensor traité
        assert result is mock_tensor
        mock_tensor.unsqueeze.assert_called_once_with(0)
        mock_tensor.to.assert_called_once()

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_get_embedding():
    """Test de la génération d'embeddings"""
    # Créer une image test
    img = Image.new('L', (100, 100), 255)
    
    # Créer un mock pour preprocess_image
    mock_tensor = MagicMock()
    
    # Créer un mock du modèle
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.cpu.return_value = mock_result
    mock_result.numpy.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_model.forward_one.return_value = mock_result
    
    with patch("app.model_loader.preprocess_image", return_value=mock_tensor):
        # Appeler la fonction à tester
        embedding = get_embedding(mock_model, img)
        
        # Vérifier que l'embedding a la bonne forme
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)  # La forme après la suppression de la dimension 0
        assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3]))

@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
def test_similarity_calculation():
    """Test du calcul de similarité"""
    # Créer deux embeddings avec une similarité connue
    embedding1 = np.array([1.0, 0.0, 0.0, 0.0])  # Vecteur unitaire dans la direction x
    embedding2 = np.array([0.707, 0.707, 0.0, 0.0])  # Vecteur unitaire à 45°
    
    # Mocker les embeddings de référence
    mock_refs = [
        ("test_class", embedding2)
    ]
    
    # Mocker cosine_similarity
    mock_similarity = np.array([[0.707]])
    
    with patch("app.similarity_search.reference_embeddings", mock_refs), \
         patch("app.similarity_search.cosine_similarity", return_value=mock_similarity):
        
        # Appeler la fonction à tester
        matches = get_top_matches(embedding1, k=1)
        
        # Vérifier le résultat
        assert isinstance(matches, list)
        assert len(matches) == 1
        assert matches[0]["class"] == "test_class"
        assert abs(matches[0]["similarity"] - 0.707) < 0.001 