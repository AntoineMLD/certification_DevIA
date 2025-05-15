import unittest
import sys
import os
import io
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock

# Ajouter le dossier parent au chemin pour pouvoir importer les modules de l'application
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les fonctions API à tester
from app.api_client import (
    get_similar_tags_from_api,
    validate_prediction,
    get_verres_by_tags_api,
    get_verre_details_api,
    get_verre_staging_details_api,
    get_model_headers,
    get_stored_token
)

class MockResponse:
    """
    Classe pour simuler les réponses API dans les tests
    """
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"Status code {self.status_code}")

class TestApiIntegration(unittest.TestCase):
    
    def setUp(self):
        """
        Configuration initiale pour chaque test
        """
        # Créer une petite image pour les tests
        self.test_image = Image.new('L', (224, 224), color=255)
        # Dessiner quelque chose de simple
        for x in range(50, 150):
            for y in range(50, 150):
                self.test_image.putpixel((x, y), 0)
    
    @patch('app.api_client.requests.post')
    def test_get_similar_tags_from_api(self, mock_post):
        """
        Tester l'envoi d'une image et la réception des tags similaires
        """
        # Configurer la réponse simulée
        mock_response = MockResponse({
            "matches": [
                {"class": "A", "similarity": 0.95},
                {"class_": "B", "similarity": 0.85},
                {"class": "C", "similarity": 0.75}
            ]
        }, 200)
        mock_post.return_value = mock_response
        
        # Appeler la fonction avec l'image de test
        result = get_similar_tags_from_api(self.test_image)
        
        # Vérifier que la demande POST a été effectuée
        mock_post.assert_called_once()
        
        # Vérifier que les résultats sont correctement normalisés
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["class"], "A")
        self.assertEqual(result[0]["similarity"], 0.95)
        # Vérifier que class_ est correctement transformé en class
        self.assertEqual(result[1]["class"], "B")
        self.assertEqual(result[1]["similarity"], 0.85)
    
    @patch('app.api_client.requests.post')
    def test_validate_prediction(self, mock_post):
        """
        Tester la validation d'une prédiction
        """
        # Configurer la réponse simulée
        mock_response = MockResponse({"status": "success", "message": "Prédiction validée"}, 200)
        mock_post.return_value = mock_response
        
        # Appeler la fonction avec une classe de test
        result = validate_prediction("A")
        
        # Vérifier que la demande POST a été effectuée avec les bons paramètres
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["predicted_class"], "A")
        
        # Vérifier le résultat
        self.assertEqual(result["status"], "success")
    
    @patch('app.api_client.requests.post')
    def test_get_verres_by_tags_api(self, mock_post):
        """
        Tester la recherche de verres par tags
        """
        # Configurer la réponse simulée
        mock_response = MockResponse({
            "results": [
                {"id": 1, "nom": "Verre 1", "tags": ["A", "B"]},
                {"id": 2, "nom": "Verre 2", "tags": ["A", "C"]}
            ]
        }, 200)
        mock_post.return_value = mock_response
        
        # Appeler la fonction avec des tags de test
        result = get_verres_by_tags_api(["A", "B"])
        
        # Vérifier que la demande POST a été effectuée avec les bons paramètres
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"], ["A", "B"])
        
        # Vérifier le résultat
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["nom"], "Verre 1")
    
    @patch('app.api_client.requests.get')
    def test_get_verre_details_api(self, mock_get):
        """
        Tester la récupération des détails d'un verre
        """
        # Configurer la réponse simulée
        mock_response = MockResponse({
            "verre": {
                "id": 1,
                "nom": "Verre Test",
                "fournisseur": "Fournisseur Test",
                "variante": "Variante Test",
                "indice": "1.50",
                "hauteur_min": "10",
                "hauteur_max": "20",
                "tags": ["A", "B", "C"]
            }
        }, 200)
        mock_get.return_value = mock_response
        
        # Appeler la fonction avec un ID de test
        result = get_verre_details_api(1)
        
        # Vérifier que la demande GET a été effectuée avec le bon URL
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertTrue("/verre/1" in args[0])
        
        # Vérifier le résultat
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["nom"], "Verre Test")
        self.assertEqual(len(result["tags"]), 3)
    
    @patch('app.api_client.requests.get')
    def test_get_verre_staging_details_api(self, mock_get):
        """
        Tester la récupération des détails de la table staging d'un verre
        """
        # Configurer la réponse simulée
        mock_response = MockResponse({
            "verre_staging": {
                "id": 1,
                "glass_name": "Nom Complet du Verre Test",
                "id_interne": "ABC123"
            }
        }, 200)
        mock_get.return_value = mock_response
        
        # Appeler la fonction avec un ID de test
        result = get_verre_staging_details_api(1)
        
        # Vérifier que la demande GET a été effectuée avec le bon URL
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertTrue("/verre_staging/1" in args[0])
        
        # Vérifier le résultat
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["glass_name"], "Nom Complet du Verre Test")
    
    @patch('app.api_client.get_stored_token')
    def test_get_model_headers_with_token(self, mock_get_token):
        """
        Tester la génération des en-têtes avec token
        """
        # Simuler un token stocké
        mock_get_token.return_value = "test_token_123"
        
        # Obtenir les en-têtes
        headers = get_model_headers()
        
        # Vérifier les en-têtes
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Authorization"], "Bearer test_token_123")
    
    @patch('app.api_client.get_stored_token')
    def test_get_model_headers_without_token(self, mock_get_token):
        """
        Tester la génération des en-têtes sans token
        """
        # Simuler l'absence de token
        mock_get_token.return_value = None
        
        # Obtenir les en-têtes
        headers = get_model_headers()
        
        # Vérifier les en-têtes
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertNotIn("Authorization", headers)

    @patch('app.api_client.requests.post')
    def test_api_error_handling(self, mock_post):
        """
        Tester la gestion des erreurs API
        """
        # Simuler une erreur 422
        mock_response = MockResponse({"detail": "Unprocessable Entity"}, 422)
        mock_post.return_value = mock_response
        
        # Vérifier que l'exception est bien levée
        with self.assertRaises(Exception):
            get_similar_tags_from_api(self.test_image)

if __name__ == '__main__':
    unittest.main() 