"""
Tests pour les opérations de base de données.
"""
import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the API directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../api')))

from app.models.database import Base, Verre
from app.config import settings

# Configuration de la base de données de test
TEST_SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def test_db():
    """Crée une base de données de test temporaire."""
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)

def test_create_verre(test_db):
    """Test de la création d'un verre."""
    verre = Verre(
        nom="Test Verre",
        variante="Test Variante",
        hauteur_min=10,
        hauteur_max=20,
        indice=1.5,
        gravure="Test Gravure"
    )
    test_db.add(verre)
    test_db.commit()
    test_db.refresh(verre)

    assert verre.id is not None
    assert verre.nom == "Test Verre"
    assert verre.indice == 1.5

def test_read_verre(test_db):
    """Test de la lecture d'un verre."""
    # Créer un verre de test
    verre = Verre(
        nom="Test Verre",
        variante="Test Variante",
        hauteur_min=10,
        hauteur_max=20,
        indice=1.5,
        gravure="Test Gravure"
    )
    test_db.add(verre)
    test_db.commit()

    # Lire le verre
    db_verre = test_db.query(Verre).filter(Verre.nom == "Test Verre").first()
    assert db_verre is not None
    assert db_verre.nom == "Test Verre"
    assert db_verre.indice == 1.5 