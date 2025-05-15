"""
Tests pour l'API FastAPI de l'application E3_MettreDispositionIA.
Ce module contient des tests pour tous les points de terminaison de l'API.
"""

import pytest
from fastapi.testclient import TestClient
import io
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

# Ajouter le chemin parent aux chemins de recherche Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import de l'application FastAPI
from app.main import app

# Client de test - utiliser une approche plus simple
client = TestClient(app)  # Réessayer sans argument nommé

# Variables de test
TEST_USERNAME = "admin@example.com"
TEST_PASSWORD = "admin_password"
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_image.png")

# Fixtures pour les tests
@pytest.fixture
def test_image():
    """Crée une image de test en mémoire"""
    img = Image.new('L', (100, 100), color=255)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def auth_token():
    """Obtient un token d'authentification pour les tests"""
    response = client.post(
        "/token",
        data={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    return response.json().get("access_token")

# Mocks pour les tests
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock les dépendances externes pour isoler les tests"""
    # Mock pour le chargement du modèle
    with patch("app.main.load_model") as mock_load_model, \
         patch("app.main.load_references") as mock_load_references, \
         patch("app.main.get_embedding") as mock_get_embedding, \
         patch("app.main.get_top_matches") as mock_get_top_matches, \
         patch("app.main.find_matching_verres") as mock_find_matching_verres, \
         patch("app.main.get_verre_details") as mock_get_verre_details, \
         patch("app.main.ADMIN_EMAIL", TEST_USERNAME), \
         patch("app.main.ADMIN_PASSWORD", TEST_PASSWORD):
        
        # Configure les comportements des mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_get_embedding.return_value = np.random.random(256)
        mock_get_top_matches.return_value = [
            {"class": "classe_test_1", "similarity": 0.9},
            {"class": "classe_test_2", "similarity": 0.7},
        ]
        mock_find_matching_verres.return_value = [
            {"id": 1, "nom": "Verre Test 1", "tags": ["tag1", "tag2"]},
            {"id": 2, "nom": "Verre Test 2", "tags": ["tag2", "tag3"]},
        ]
        mock_get_verre_details.return_value = {
            "id": 1,
            "nom": "Verre Test Detail",
            "description": "Description du verre test",
            "tags": ["tag1", "tag2"],
            "image_url": "http://example.com/image.jpg"
        }
        
        yield

# Tests des points de terminaison
def test_token_endpoint_success():
    """Test de l'authentification réussie"""
    response = client.post(
        "/token",
        data={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_token_endpoint_failure():
    """Test de l'authentification échouée"""
    response = client.post(
        "/token",
        data={"username": "wrong@example.com", "password": "wrong_password"}
    )
    assert response.status_code == 401

def test_embedding_endpoint(auth_token, test_image):
    """Test du calcul d'embedding d'une image"""
    response = client.post(
        "/embedding",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("test_image.png", test_image, "image/png")}
    )
    assert response.status_code == 200
    assert "embedding" in response.json()
    assert isinstance(response.json()["embedding"], list)

def test_match_endpoint(auth_token, test_image):
    """Test de la recherche de correspondances pour une image"""
    response = client.post(
        "/match",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("test_image.png", test_image, "image/png")}
    )
    assert response.status_code == 200
    assert "matches" in response.json()
    matches = response.json()["matches"]
    assert isinstance(matches, list)
    assert len(matches) > 0
    assert "class_" in matches[0]
    assert "similarity" in matches[0]

def test_validate_prediction_endpoint(auth_token):
    """Test de la validation d'une prédiction"""
    # D'abord, effectuons une prédiction pour avoir quelque chose à valider
    with patch("app.main.monitor.add_temp_prediction") as mock_add_temp:
        with patch("app.main.monitor.validate_last_prediction", return_value=True) as mock_validate:
            # Faire une validation
            response = client.post(
                "/validate_prediction",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={"predicted_class": "classe_test_1"}
            )
            assert response.status_code == 200
            assert "status" in response.json()
            assert response.json()["status"] == "success"
            assert "message" in response.json()
            # Vérifier que la fonction de validation a été appelée
            mock_validate.assert_called_once_with("classe_test_1")

def test_search_tags_endpoint(auth_token):
    """Test de la recherche de verres par tags"""
    response = client.post(
        "/search_tags",
        headers={"Authorization": f"Bearer {auth_token}"},
        json=["tag1", "tag2"]
    )
    assert response.status_code == 200
    assert "results" in response.json()
    results = response.json()["results"]
    assert isinstance(results, list)
    assert len(results) > 0
    assert "nom" in results[0]
    assert "tags" in results[0]

def test_get_verre_endpoint(auth_token):
    """Test de la récupération des détails d'un verre"""
    response = client.get(
        "/verre/1",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert "verre" in response.json()
    verre = response.json()["verre"]
    assert "id" in verre
    assert "nom" in verre
    assert "description" in verre
    assert "tags" in verre

def test_unauthorized_access():
    """Test de l'accès non autorisé aux endpoints protégés"""
    endpoints = [
        ("/embedding", "POST", {"file": ("test_image.png", io.BytesIO(), "image/png")}),
        ("/match", "POST", {"file": ("test_image.png", io.BytesIO(), "image/png")}),
        ("/validate_prediction", "POST", {"json": {"predicted_class": "classe_test"}}),
        ("/search_tags", "POST", {"json": ["tag1"]}),
        ("/verre/1", "GET", {})
    ]
    
    for endpoint, method, kwargs in endpoints:
        if method == "GET":
            response = client.get(endpoint)
        else:
            if "file" in kwargs:
                response = client.post(endpoint, files=kwargs["file"])
            elif "json" in kwargs:
                response = client.post(endpoint, json=kwargs["json"])
            else:
                response = client.post(endpoint)
        
        assert response.status_code in [401, 403], f"Endpoint {endpoint} devrait être protégé"

def test_rate_limiting():
    """Test du limiteur de taux d'appels"""
    # Ce test est plus complexe et nécessiterait de contourner le limiteur
    # Pour tester correctement, on pourrait utiliser un mock pour le limiteur
    # Ici, nous vérifions simplement que le décorateur est présent en inspectant le code
    
    with patch("app.main.limiter.limit") as mock_limit:
        # Créer une nouvelle instance du client pour éviter les conflits avec les autres tests
        new_client = TestClient(app)
        response = new_client.post(
            "/token",
            data={"username": TEST_USERNAME, "password": TEST_PASSWORD}
        )
        # Vérifier que le décorateur a été appliqué (indirectement)
        assert mock_limit.called

def test_image_validation(auth_token):
    """Test de la validation des images"""
    # Créer un fichier non-image
    invalid_file = io.BytesIO(b"not an image")
    
    # Test avec un fichier non-image
    response = client.post(
        "/match",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("test.txt", invalid_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"] 