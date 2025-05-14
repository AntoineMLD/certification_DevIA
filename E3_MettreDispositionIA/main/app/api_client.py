import requests
import streamlit as st
from PIL import Image
import io

DB_API_URL = "http://localhost:8001"  # API de la base de données
MODEL_API_URL = "http://localhost:8000"  # API du modèle IA

def get_db_headers():
    """
    Retourne les headers avec le token d'authentification pour l'API de base de données
    """
    if "db_token" in st.session_state and st.session_state.db_token:
        return {
            "Authorization": f"Bearer {st.session_state.db_token}",
            "Content-Type": "application/json"
        }
    return {"Content-Type": "application/json"}

def get_model_headers():
    """
    Retourne les headers avec le token d'authentification pour l'API du modèle
    """
    if "model_token" in st.session_state and st.session_state.model_token:
        return {
            "Authorization": f"Bearer {st.session_state.model_token}",
            "Content-Type": "application/json"
        }
    return {"Content-Type": "application/json"}

def get_similar_tags_from_api(image: Image.Image, route: str = "/match"):
    """
    Envoie une image à l'API du modèle pour obtenir les symboles similaires.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    
    # Récupérer les headers d'authentification
    headers = get_model_headers()
    # Supprimer Content-Type car il sera automatiquement défini par requests pour les fichiers
    headers.pop("Content-Type", None)
    
    response = requests.post(
        f"{MODEL_API_URL}{route}",
        files={"file": ("image.png", buffered, "image/png")},
        headers=headers
    )
    response.raise_for_status()
    return response.json()["matches"]

def get_verres_by_tags_api(tags: list[str], route: str = "/search_tags"):
    """
    Envoie une liste de tags à l'API du modèle pour récupérer les verres correspondants.
    """
    response = requests.post(
        f"{MODEL_API_URL}{route}",
        json=tags,
        headers=get_model_headers()
    )
    response.raise_for_status()
    return response.json()["results"]

def get_verre_details_api(verre_id: int, route: str = "/verres"):
    """
    Récupère les détails d'un verre depuis l'API de base de données.
    """
    response = requests.get(
        f"{DB_API_URL}{route}/{verre_id}",
        headers=get_db_headers()
    )
    response.raise_for_status()
    return response.json()