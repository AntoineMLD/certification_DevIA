import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Ajouter le dossier parent au chemin pour pouvoir importer les modules de l'application
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch Streamlit pour éviter les erreurs
streamlit_mock = MagicMock()
# Créer un objet qui simule une session_state avec attributs
session_state_mock = MagicMock()
session_state_mock.authenticated = True
session_state_mock.db_token = 'db_token_123'
streamlit_mock.session_state = session_state_mock
sys.modules['streamlit'] = streamlit_mock

# Importer les fonctions d'authentification à tester
from app.auth import login, logout

class TestAuth(unittest.TestCase):
    
    @patch('app.auth.requests.post')
    @patch('app.auth.store_model_api_token')
    def test_login_success(self, mock_store_token, mock_post):
        """
        Tester l'authentification réussie
        """
        # Simuler des réponses réussies pour les deux APIs
        mock_db_response = MagicMock()
        mock_db_response.status_code = 200
        mock_db_response.json.return_value = {"access_token": "db_token_123"}
        
        mock_model_response = MagicMock()
        mock_model_response.status_code = 200
        mock_model_response.json.return_value = {"access_token": "model_token_456"}
        
        # Configurer mock_post pour retourner différentes réponses selon l'URL
        def side_effect(url, data=None, **kwargs):
            if "8001/token" in url:  # DB API
                return mock_db_response
            elif "8000/token" in url:  # Model API
                return mock_model_response
            return MagicMock()
        
        mock_post.side_effect = side_effect
        
        # Appeler la fonction de login
        db_token, model_token = login("user@example.com", "password123")
        
        # Vérifier que les tokens sont correctement retournés
        self.assertEqual(db_token, "db_token_123")
        self.assertEqual(model_token, "model_token_456")
        
        # Vérifier que le token du modèle a été stocké
        mock_store_token.assert_called_once_with("model_token_456")
    
    @patch('app.auth.requests.post')
    @patch('app.auth.store_model_api_token')
    def test_login_failure(self, mock_store_token, mock_post):
        """
        Tester l'échec de l'authentification
        """
        # Simuler des réponses d'échec pour une des APIs
        mock_db_response = MagicMock()
        mock_db_response.status_code = 200
        mock_db_response.json.return_value = {"access_token": "db_token_123"}
        
        mock_model_response = MagicMock()
        mock_model_response.status_code = 401  # Échec d'authentification
        
        # Configurer mock_post pour retourner différentes réponses selon l'URL
        def side_effect(url, data=None, **kwargs):
            if "8001/token" in url:  # DB API
                return mock_db_response
            elif "8000/token" in url:  # Model API
                return mock_model_response
            return MagicMock()
        
        mock_post.side_effect = side_effect
        
        # Appeler la fonction de login
        db_token, model_token = login("user@example.com", "wrong_password")
        
        # Vérifier que les tokens sont None
        self.assertIsNone(db_token)
        self.assertIsNone(model_token)
        
        # Vérifier que le token du modèle n'a pas été stocké
        mock_store_token.assert_not_called()
    
    @patch('app.auth.requests.post')
    def test_login_exception(self, mock_post):
        """
        Tester la gestion des exceptions lors de l'authentification
        """
        # Simuler une exception
        mock_post.side_effect = Exception("Connection error")
        
        # Appeler la fonction de login avec le mock de streamlit
        with patch.object(streamlit_mock, 'error'):
            db_token, model_token = login("user@example.com", "password123")
            
            # Vérifier que l'erreur est affichée
            streamlit_mock.error.assert_called()
        
        # Vérifier que les tokens sont None
        self.assertIsNone(db_token)
        self.assertIsNone(model_token)
    
    @patch('app.auth.store_model_api_token')
    def test_logout(self, mock_store_token):
        """
        Tester la déconnexion
        """
        # Réinitialiser les mocks pour ce test spécifique
        # s'assurer que session_state est correctement configuré
        streamlit_mock.session_state.authenticated = True
        streamlit_mock.session_state.db_token = 'db_token_123'
        
        # Simuler un rerun pour éviter l'erreur
        with patch.object(streamlit_mock, 'rerun'):
            # Appeler la fonction de déconnexion
            logout()
            
            # Vérifier que les valeurs de session ont été changées
            self.assertFalse(streamlit_mock.session_state.authenticated)
            self.assertIsNone(streamlit_mock.session_state.db_token)
            
            # Vérifier que le token est supprimé
            mock_store_token.assert_called_once_with(None)

if __name__ == '__main__':
    unittest.main() 