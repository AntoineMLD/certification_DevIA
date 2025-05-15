import streamlit as st
import requests
from typing import Optional, Tuple
from app.api_client import store_token as store_model_api_token # Importer la fonction de stockage avec le bon chemin

# URLs des APIs
DB_API_URL = "http://localhost:8001"  # API de la base de données
MODEL_API_URL = "http://localhost:8000"  # API du modèle IA

def login(username: str, password: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Authentifie l'utilisateur auprès des deux APIs et retourne les tokens
    """
    try:
        # Authentification auprès de l'API de base de données
        db_response = requests.post(
            f"{DB_API_URL}/token",
            data={"username": username, "password": password}
        )
        
        # Authentification auprès de l'API du modèle
        model_response = requests.post(
            f"{MODEL_API_URL}/token",
            data={"username": username, "password": password}
        )
        
        if db_response.status_code == 200 and model_response.status_code == 200:
            model_token = model_response.json()["access_token"]
            store_model_api_token(model_token) # Stocker le token pour l'API modèle
            return (
                db_response.json()["access_token"],
                model_token
            )
        return None, None
    except Exception as e:
        st.error(f"Erreur de connexion: {e}")
        return None, None

def check_authentication():
    """
    Vérifie si l'utilisateur est authentifié
    Redirige vers la page de login si nécessaire
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.db_token = None
        # st.session_state.model_token n'est plus nécessaire ici si api_client gère son propre état

    if not st.session_state.authenticated:
        st.title("🔐 Connexion")
        
        with st.form("login_form"):
            username = st.text_input("Email")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter")
            
            if submit:
                db_token, model_token_from_login = login(username, password)
                if db_token and model_token_from_login: # model_token_from_login est déjà stocké par store_model_api_token dans login()
                    st.session_state.db_token = db_token
                    # st.session_state.model_token = model_token_from_login # Plus besoin de le stocker ici
                    st.session_state.authenticated = True
                    st.success("Connexion réussie!")
                    st.rerun()
                else:
                    st.error("Email ou mot de passe incorrect")
        
        st.stop()  # Arrête l'exécution si non authentifié

def logout():
    """
    Déconnecte l'utilisateur
    """
    st.session_state.authenticated = False
    st.session_state.db_token = None
    # st.session_state.model_token = None # Plus besoin
    store_model_api_token(None) # Effacer le token stocké dans api_client
    st.rerun() 