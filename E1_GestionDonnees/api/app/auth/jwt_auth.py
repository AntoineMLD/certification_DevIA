from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from ..config import settings

# Configuration de base pour la sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def check_password(plain_password: str, hashed_password: str) -> bool:
    """
    Vérifie si le mot de passe correspond au hash
    
    Args:
        plain_password: Mot de passe en clair
        hashed_password: Hash du mot de passe
    Returns:
        bool: True si le mot de passe correspond, False sinon
    """
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """
    Crée un hash sécurisé du mot de passe
    
    Args:
        password: Mot de passe à hasher
    Returns:
        str: Hash du mot de passe
    """
    return pwd_context.hash(password)

def create_token(user_data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Crée un token JWT pour l'utilisateur
    
    Args:
        user_data: Données de l'utilisateur à encoder dans le token
        expires_delta: Durée de validité du token (optionnel)
    Returns:
        str: Token JWT encodé
    """
    # Copie des données pour ne pas modifier l'original
    token_data = user_data.copy()
    
    # Calcul de la date d'expiration
    if expires_delta:
        expiration = datetime.utcnow() + expires_delta
    else:
        expiration = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Ajout de l'expiration aux données du token
    token_data.update({"exp": expiration})
    
    # Création du token JWT
    token = jwt.encode(
        token_data,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return token

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Vérifie le token JWT et retourne l'utilisateur
    
    Args:
        token: Token JWT à vérifier
    Returns:
        str: Email de l'utilisateur si le token est valide
    Raises:
        HTTPException: Si le token est invalide
    """
    # Préparation de l'erreur d'authentification
    auth_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Décodage du token
        token_data = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Extraction de l'email
        user_email = token_data.get("sub")
        if not user_email:
            raise auth_error
            
        # Vérification de l'admin
        if user_email != settings.ADMIN_EMAIL:
            raise auth_error
            
        return user_email
        
    except JWTError:
        raise auth_error

def check_user(email: str, password: str) -> bool:
    """
    Vérifie les identifiants de l'utilisateur
    
    Args:
        email: Email de l'utilisateur
        password: Mot de passe en clair
    Returns:
        bool: True si les identifiants sont valides, False sinon
    """
    return email == settings.ADMIN_EMAIL and password == settings.ADMIN_PASSWORD 