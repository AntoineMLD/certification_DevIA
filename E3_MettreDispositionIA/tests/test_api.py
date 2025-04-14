import os
import sys
import unittest
from fastapi.testclient import TestClient
import io
from PIL import Image
import numpy as np

# Ajouter le répertoire parent au chemin Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Importer l'application FastAPI et initialiser le client de test
from app.main import app
from app import init_app

# Client de test FastAPI
client = TestClient(app)

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialiser l'application (modèle et embeddings)
        init_app()
    
    def test_docs_endpoint(self):
        """Vérifier que la documentation Swagger est accessible"""
        response = client.get("/docs")
        self.assertEqual(response.status_code, 200)
    
    def test_authentication(self):
        """Tester l'authentification par token"""
        # Essayer d'obtenir un token avec des identifiants incorrects
        response = client.post("/token", data={"username": "invalid", "password": "invalid"})
        self.assertEqual(response.status_code, 401)
        
        # Obtenir un token avec des identifiants corrects
        response = client.post("/token", data={"username": "utilisateur", "password": "password123"})
        self.assertEqual(response.status_code, 200)
        
        # Vérifier que la réponse contient un token
        token_data = response.json()
        self.assertIn("access_token", token_data)
        self.assertIn("token_type", token_data)
        self.assertEqual(token_data["token_type"], "bearer")
    
    def test_get_gravures_unauthorized(self):
        """Tester l'accès non autorisé à la liste des gravures"""
        response = client.get("/gravures")
        self.assertEqual(response.status_code, 401)
    
    def test_get_gravures_authorized(self):
        """Tester l'accès autorisé à la liste des gravures"""
        # Obtenir un token
        auth_response = client.post("/token", data={"username": "utilisateur", "password": "password123"})
        token = auth_response.json()["access_token"]
        
        # Appeler l'endpoint avec le token
        response = client.get("/gravures", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        
        # Vérifier que la réponse est une liste non vide
        gravures = response.json()
        self.assertIsInstance(gravures, list)
        self.assertGreater(len(gravures), 0)
        
        # Vérifier le format d'une gravure
        gravure = gravures[0]
        self.assertIn("id", gravure)
        self.assertIn("code", gravure)
        self.assertIn("indice", gravure)
    
    def test_recognize_gravure(self):
        """Tester l'endpoint de reconnaissance de gravure"""
        # Obtenir un token
        auth_response = client.post("/token", data={"username": "utilisateur", "password": "password123"})
        token = auth_response.json()["access_token"]
        
        # Créer une image test (carré blanc avec un cercle noir)
        img = Image.new('L', (64, 64), 255)  # Image blanche
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        
        # Appeler l'endpoint de reconnaissance
        response = client.post(
            "/recognize",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("test_image.png", img_io, "image/png")}
        )
        
        # Vérifier que la réponse est valide
        self.assertEqual(response.status_code, 200)
        
        # Vérifier le format de la réponse
        result = response.json()
        self.assertIn("id", result)
        self.assertIn("code", result)
        self.assertIn("indice", result)
        self.assertIn("score", result)
        
        # Vérifier que le score est compris entre 0 et 1
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)

if __name__ == '__main__':
    unittest.main() 