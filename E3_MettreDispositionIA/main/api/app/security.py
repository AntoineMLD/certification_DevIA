"""
Module de sécurité pour l'API
Gère la validation des entrées, les tokens et les logs de sécurité
"""

from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, EmailStr, constr
import jwt
from fastapi import HTTPException, status
import logging
import os
from logging.handlers import RotatingFileHandler

# Configuration du logging de sécurité
def setup_security_logging():
    # Créer le dossier logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)
    
    # Configurer le logger de sécurité
    security_logger = logging.getLogger("security")
    security_logger.setLevel(logging.INFO)
    
    # Créer un handler qui écrit dans un fichier avec rotation
    handler = RotatingFileHandler(
        "logs/security.log",
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5
    )
    
    # Formater les logs
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    security_logger.addHandler(handler)
    
    return security_logger

# Initialiser le logger
security_logger = setup_security_logging()

# Modèles de validation
class TokenData(BaseModel):
    email: EmailStr
    exp: datetime
    token_version: int

class UserCredentials(BaseModel):
    email: EmailStr
    password: str  # Validation simple du mot de passe

# Configuration des tokens
TOKEN_SETTINGS = {
    "SECRET_KEY": os.getenv("SECRET_KEY", "your-secret-key-for-jwt"),
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 30,
    "ROTATION_THRESHOLD_MINUTES": 25  # Rotation 5 minutes avant expiration
}

# Stockage des versions de token (en mémoire - à remplacer par une base de données en production)
token_versions = {}

def log_security_event(event_type: str, details: str, level: str = "INFO"):
    """
    Enregistre un événement de sécurité
    """
    log_message = f"{event_type} - {details}"
    if level == "INFO":
        security_logger.info(log_message)
    elif level == "WARNING":
        security_logger.warning(log_message)
    elif level == "ERROR":
        security_logger.error(log_message)

def create_access_token(email: str) -> tuple[str, int]:
    """
    Crée un nouveau token d'accès avec version
    """
    # Incrémenter ou initialiser la version du token pour cet utilisateur
    current_version = token_versions.get(email, 0) + 1
    token_versions[email] = current_version
    
    # Créer les données du token
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_SETTINGS["ACCESS_TOKEN_EXPIRE_MINUTES"])
    token_data = {
        "sub": email,
        "exp": expire,
        "token_version": current_version
    }
    
    # Générer le token
    token = jwt.encode(
        token_data,
        TOKEN_SETTINGS["SECRET_KEY"],
        algorithm=TOKEN_SETTINGS["ALGORITHM"]
    )
    
    log_security_event(
        "TOKEN_CREATED",
        f"New token created for {email} with version {current_version}"
    )
    
    return token, current_version

def verify_token(token: str) -> TokenData:
    """
    Vérifie un token et gère la rotation si nécessaire
    """
    try:
        # Décoder le token
        payload = jwt.decode(
            token,
            TOKEN_SETTINGS["SECRET_KEY"],
            algorithms=[TOKEN_SETTINGS["ALGORITHM"]]
        )
        
        # Valider les données du token
        token_data = TokenData(
            email=payload["sub"],
            exp=datetime.fromtimestamp(payload["exp"]),
            token_version=payload["token_version"]
        )
        
        # Vérifier la version du token
        if token_data.token_version != token_versions.get(token_data.email, 0):
            log_security_event(
                "TOKEN_INVALID_VERSION",
                f"Invalid token version for {token_data.email}",
                "WARNING"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token version is invalid"
            )
        
        # Vérifier si le token doit être renouvelé
        time_to_expire = token_data.exp - datetime.utcnow()
        should_rotate = time_to_expire <= timedelta(minutes=TOKEN_SETTINGS["ROTATION_THRESHOLD_MINUTES"])
        
        if should_rotate:
            log_security_event(
                "TOKEN_ROTATION",
                f"Token rotation needed for {token_data.email}"
            )
            return token_data, True
            
        return token_data, False
        
    except jwt.ExpiredSignatureError:
        log_security_event(
            "TOKEN_EXPIRED",
            "Token has expired",
            "WARNING"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError as e:
        log_security_event(
            "TOKEN_INVALID",
            f"Invalid token: {str(e)}",
            "ERROR"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate token"
        )

def validate_image_file(file_content: bytes, max_size: int = 5 * 1024 * 1024) -> bool:
    """
    Valide un fichier image
    """
    # Vérifier la taille du fichier
    if len(file_content) > max_size:
        log_security_event(
            "FILE_TOO_LARGE",
            f"File size {len(file_content)} exceeds maximum {max_size}",
            "WARNING"
        )
        return False
        
    # Vérifier les premiers octets pour le type de fichier
    allowed_signatures = [
        b'\xFF\xD8\xFF',  # JPEG
        b'\x89\x50\x4E\x47',  # PNG
    ]
    
    is_valid = any(file_content.startswith(sig) for sig in allowed_signatures)
    
    if not is_valid:
        log_security_event(
            "INVALID_FILE_TYPE",
            "File type not allowed",
            "WARNING"
        )
        
    return is_valid 