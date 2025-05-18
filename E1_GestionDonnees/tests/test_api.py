"""
Tests pour l'API E1_GestionDonnees.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the API directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../api')))

from app.main import app
from app.models.database import Base
from app.config import settings

# Configuration de la base de données de test
TEST_SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Fournit une session de base de données de test."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def test_db():
    """Crée une base de données de test temporaire."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(test_db):
    """Crée un client de test pour l'API."""
    return TestClient(app)

def test_read_main(client):
    """Test de la route principale."""
    response = client.get("/")
    assert response.status_code == 200
    assert "API" in response.json()["message"]

def test_token_endpoint(client):
    """Test de l'endpoint de génération de token."""
    response = client.post(
        "/token",
        data={
            "username": settings.ADMIN_EMAIL,
            "password": settings.ADMIN_PASSWORD
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()
    assert response.json()["token_type"] == "bearer" 