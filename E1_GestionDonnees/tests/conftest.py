"""
Fixtures partagées pour les tests.
"""
import pytest
import os
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def env():
    """Charge les variables d'environnement de test."""
    load_dotenv("tests/.env.test")
    return None 