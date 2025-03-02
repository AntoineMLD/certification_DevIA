import pytest
from fastapi.testclient import TestClient
from api.app.main import app

# Créer un client de test
client = TestClient(app)

def test_token_endpoint():
    """
    Test de l'endpoint d'authentification
    """
    response = client.post(
        "/token",
        data={
            "username": "test@example.com",
            "password": "test123"
        }
    )
    assert response.status_code in [200, 401]

def test_get_verres_sans_auth():
    """
    Test de l'endpoint /verres sans authentification
    """
    response = client.get("/verres")
    assert response.status_code == 401

def test_structure_api():
    """
    Test de la structure de base de l'API
    """
    response = client.get("/")
    assert response.status_code == 404  # Racine non définie
    
    response = client.get("/docs")
    assert response.status_code == 200  # Documentation disponible 