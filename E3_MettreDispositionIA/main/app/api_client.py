import requests
import streamlit as st
from PIL import Image
import io
import logging
import os
from jose import jwt, JWTError
from datetime import datetime, timedelta

DB_API_URL = "http://localhost:8001"  # API de la base de données
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000")
TOKEN_KEY = os.getenv("TOKEN_KEY", "votre_cle_secrete_ici")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

logger = logging.getLogger(__name__)

def create_api_token(email: str, version: int = 1) -> str:
    to_encode = {
        "sub": email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "version": version
    }
    encoded_jwt = jwt.encode(to_encode, TOKEN_KEY, algorithm=ALGORITHM)
    return encoded_jwt

_session_token: str | None = None

def store_token(token: str):
    global _session_token
    _session_token = token
    logger.info("Token d'authentification stocké dans la session")

def get_stored_token() -> str | None:
    global _session_token
    if _session_token:
        logger.info("Token d'authentification trouvé dans la session")
    else:
        logger.warning("Aucun token d'authentification trouvé dans la session")
    return _session_token

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
    token = get_stored_token()
    headers = {
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def get_similar_tags_from_api(image: Image.Image, route: str = "/match"):
    """
    Envoie une image à l'API du modèle pour obtenir les symboles similaires.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    
    headers = get_model_headers()
    headers.pop("Content-Type", None)
    
    logger.info(f"Appel de l'API {MODEL_API_URL}{route} pour la correspondance d'image")
    response = requests.post(
        f"{MODEL_API_URL}{route}",
        files={"file": ("image.png", buffered, "image/png")},
        headers=headers
    )
    if response.status_code == 422:
        try:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}. Détails: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}. Réponse non JSON: {response.text}")
    response.raise_for_status()
    
    # Récupérer les résultats et normaliser les clés
    results = response.json()["matches"]
    
    # Normaliser les résultats pour gérer à la fois 'class' et 'class_'
    normalized_results = []
    for match in results:
        # Créer une copie du match pour ne pas modifier l'original
        normalized_match = dict(match)
        
        # S'assurer que la clé 'class' existe (même si on avait 'class_')
        if 'class_' in normalized_match and 'class' not in normalized_match:
            normalized_match['class'] = normalized_match['class_']
        
        # S'assurer que similarity existe
        if 'similarity' not in normalized_match:
            normalized_match['similarity'] = 0.0
            
        normalized_results.append(normalized_match)
    
    logger.info(f"Résultats normalisés: {normalized_results}")
    return normalized_results

def validate_prediction(predicted_class: str, route: str = "/validate_prediction"):
    """
    Envoie la classe prédite validée à l'API.
    """
    logger.info(f"Appel de l'API {MODEL_API_URL}{route} pour valider la classe: {predicted_class}")
    response = requests.post(
        f"{MODEL_API_URL}{route}",
        json={"predicted_class": predicted_class},
        headers=get_model_headers()
    )
    if response.status_code == 422:
        try:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}. Détails: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}. Réponse non JSON: {response.text}")
    response.raise_for_status()
    return response.json()

def get_verres_by_tags_api(tags: list[str], route: str = "/search_tags"):
    """
    Envoie une liste de tags à l'API du modèle pour récupérer les verres correspondants.
    """
    logger.info(f"Appel de l'API {MODEL_API_URL}{route} avec les tags: {tags}")
    response = requests.post(
        f"{MODEL_API_URL}{route}",
        json=tags,
        headers=get_model_headers()
    )
    if response.status_code == 422:
        try:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}. Détails: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}. Réponse non JSON: {response.text}")
    response.raise_for_status()
    return response.json()["results"]

def get_verre_details_api(verre_id: int, route: str = "/verre"):
    """
    Récupère les détails complets d'un verre par son ID.
    """
    logger.info(f"Appel de l'API {MODEL_API_URL}{route}/{verre_id} pour les détails du verre")
    response = requests.get(
        f"{MODEL_API_URL}{route}/{verre_id}",
        headers=get_model_headers()
    )
    if response.status_code == 422:
        try:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}/{verre_id}. Détails: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}/{verre_id}. Réponse non JSON: {response.text}")
    response.raise_for_status()
    return response.json()["verre"]

def get_verre_staging_details_api(verre_id: int, route: str = "/verre_staging"):
    """
    Récupère les détails d'un verre depuis la table staging par son ID.
    Cette table contient les informations complètes, notamment le champ glass_name.
    """
    logger.info(f"Appel de l'API {MODEL_API_URL}{route}/{verre_id} pour les détails du verre depuis staging")
    try:
        response = requests.get(
            f"{MODEL_API_URL}{route}/{verre_id}",
            headers=get_model_headers()
        )
        
        if response.status_code == 422:
            try:
                logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}/{verre_id}. Détails: {response.json()}")
            except requests.exceptions.JSONDecodeError:
                logger.error(f"Erreur 422 (Unprocessable Entity) de l'API {route}/{verre_id}. Réponse non JSON: {response.text}")
            return {}
            
        response.raise_for_status()
        
        # Obtenir les données et les journaliser
        data = response.json()
        verre_staging = data.get("verre_staging", {})
        
        if verre_staging:
            logger.info(f"Données de staging récupérées avec succès pour ID {verre_id}: glass_name={verre_staging.get('glass_name', 'Non disponible')}")
        else:
            logger.warning(f"Données de staging vides ou manquantes pour ID {verre_id}")
        
        return verre_staging
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données staging pour ID {verre_id}: {str(e)}")
        return {}