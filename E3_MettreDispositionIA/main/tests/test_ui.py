import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import io

# Ajouter le dossier parent au chemin pour pouvoir importer les modules de l'application
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Créer un mock complet pour Streamlit
streamlit_mock = MagicMock()
# Configurer le mock pour qu'il retourne ce qu'il faut pour les éléments d'interface
columns_mock = [MagicMock(), MagicMock()]
streamlit_mock.columns.return_value = columns_mock
# Session state avec attributs
session_state_mock = MagicMock()
session_state_mock.authenticated = True
streamlit_mock.session_state = session_state_mock
sys.modules['streamlit'] = streamlit_mock

# Patch pour les autres dépendances
sys.modules['streamlit_drawable_canvas'] = MagicMock()

# Désactiver check_authentication
with patch('app.auth.check_authentication'), \
     patch('app.app.st.columns', return_value=columns_mock):
    # Importer uniquement les fonctions spécifiques à tester, pas tout le module
    from app.app import find_symbol_image, extract_verre_name, get_full_verre_details

class TestUI(unittest.TestCase):
    
    @patch('os.path.exists')
    @patch('app.app.Image.open')
    def test_find_symbol_image_success(self, mock_image_open, mock_exists):
        """
        Tester la recherche d'image de symbole avec succès
        """
        # Simuler l'existence d'une image
        mock_exists.return_value = True
        
        # Simuler l'ouverture d'une image
        test_image = Image.new('L', (100, 100), color=255)
        mock_image_open.return_value = test_image
        
        # Appeler la fonction
        result = find_symbol_image("test_symbol")
        
        # Vérifier que l'image a été ouverte
        self.assertIsNotNone(result)
        self.assertEqual(result, test_image)
    
    @patch('os.path.exists')
    def test_find_symbol_image_not_found(self, mock_exists):
        """
        Tester la recherche d'image de symbole avec échec
        """
        # Simuler l'absence d'images
        mock_exists.return_value = False
        
        # Appeler la fonction
        result = find_symbol_image("nonexistent_symbol")
        
        # Vérifier que None a été retourné
        self.assertIsNone(result)
    
    def test_find_symbol_image_invalid_input(self):
        """
        Tester la recherche d'image avec des entrées invalides
        """
        # Tester avec une entrée None
        result = find_symbol_image(None)
        self.assertIsNone(result)
        
        # Tester avec une chaîne vide
        result = find_symbol_image("")
        self.assertIsNone(result)
        
        # Tester avec 'inconnu'
        result = find_symbol_image("inconnu")
        self.assertIsNone(result)
    
    def test_extract_verre_name(self):
        """
        Tester l'extraction du nom d'un verre
        """
        # Cas avec glass_name
        verre_data = {"glass_name": "Verre complet", "nom": "Verre"}
        result = extract_verre_name(verre_data)
        self.assertEqual(result, "Verre complet")
        
        # Cas avec nom seulement
        verre_data = {"nom": "Verre"}
        result = extract_verre_name(verre_data)
        self.assertEqual(result, "Verre")
        
        # Cas sans nom
        verre_data = {}
        result = extract_verre_name(verre_data)
        self.assertEqual(result, "Non spécifié")
    
    @patch('app.app.get_verre_details_api')
    @patch('app.app.get_verre_staging_details_api')
    def test_get_full_verre_details(self, mock_staging, mock_details):
        """
        Tester la récupération des détails complets d'un verre
        """
        # Simuler les données de base
        mock_details.return_value = {
            "id": 1,
            "nom": "Verre Test",
            "fournisseur": "Fournisseur Test"
        }
        
        # Simuler les données de staging
        mock_staging.return_value = {
            "id": 1,
            "glass_name": "Nom complet du verre"
        }
        
        # Appeler la fonction
        result = get_full_verre_details(1)
        
        # Vérifier la fusion des résultats
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["nom"], "Verre Test")
        self.assertEqual(result["glass_name"], "Nom complet du verre")
    
    @patch('app.app.get_verre_details_api')
    @patch('app.app.get_verre_staging_details_api')
    def test_get_full_verre_details_no_staging(self, mock_staging, mock_details):
        """
        Tester la récupération des détails sans données de staging
        """
        # Simuler les données de base
        mock_details.return_value = {
            "id": 1,
            "nom": "Verre Test",
            "fournisseur": "Fournisseur Test"
        }
        
        # Simuler l'absence de données de staging
        mock_staging.return_value = {}
        
        # Appeler la fonction
        result = get_full_verre_details(1)
        
        # Vérifier que seules les données de base sont présentes
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["nom"], "Verre Test")
        self.assertNotIn("glass_name", result)

if __name__ == '__main__':
    unittest.main() 