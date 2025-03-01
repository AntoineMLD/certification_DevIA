from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .config import settings
from .auth import jwt_auth
from .models.database import get_db, Verre
from .schemas import schemas
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_

# Création de l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Configuration CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/token", response_model=schemas.Token)
async def get_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Génère un token d'accès JWT
    
    Args:
        form_data: Données du formulaire de connexion
    Returns:
        dict: Token d'accès et son type
    Raises:
        HTTPException: Si les identifiants sont incorrects
    """
    # Vérification des identifiants
    if not jwt_auth.check_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Création du token
    token = jwt_auth.create_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/verres", response_model=List[schemas.VerreDetail])
async def get_all_glasses(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: str = Depends(jwt_auth.get_current_user)
):
    """
    Récupère la liste des verres avec pagination
    
    Args:
        skip: Nombre d'éléments à sauter
        limit: Nombre maximum d'éléments à retourner
        db: Session de base de données
        current_user: Utilisateur authentifié
    Returns:
        List[VerreDetail]: Liste des verres avec leurs détails
    """
    glasses = (
        db.query(Verre)
        .options(
            joinedload(Verre.fournisseur),
            joinedload(Verre.materiau),
            joinedload(Verre.gamme),
            joinedload(Verre.serie),
            joinedload(Verre.traitements)
        )
        .offset(skip)
        .limit(limit)
        .all()
    )
    return glasses

@app.get("/verres/{glass_id}", response_model=schemas.VerreDetail)
async def get_glass_by_id(
    glass_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(jwt_auth.get_current_user)
):
    """
    Récupère les détails d'un verre spécifique
    
    Args:
        glass_id: ID du verre à récupérer
        db: Session de base de données
        current_user: Utilisateur authentifié
    Returns:
        VerreDetail: Détails du verre
    Raises:
        HTTPException: Si le verre n'est pas trouvé
    """
    glass = (
        db.query(Verre)
        .options(
            joinedload(Verre.fournisseur),
            joinedload(Verre.materiau),
            joinedload(Verre.gamme),
            joinedload(Verre.serie),
            joinedload(Verre.traitements)
        )
        .filter(Verre.id == glass_id)
        .first()
    )
    
    if glass is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Verre non trouvé"
        )
    
    return glass

@app.get("/verres/search/", response_model=List[schemas.VerreDetail])
async def search_glasses(
    query: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(jwt_auth.get_current_user)
):
    """
    Recherche des verres par nom ou gravure
    
    Args:
        query: Texte à rechercher
        db: Session de base de données
        current_user: Utilisateur authentifié
    Returns:
        List[VerreDetail]: Liste des verres correspondant à la recherche
    """
    glasses = (
        db.query(Verre)
        .options(
            joinedload(Verre.fournisseur),
            joinedload(Verre.materiau),
            joinedload(Verre.gamme),
            joinedload(Verre.serie),
            joinedload(Verre.traitements)
        )
        .filter(
            or_(
                Verre.nom.ilike(f"%{query}%"),
                Verre.gravure.ilike(f"%{query}%")
            )
        )
        .all()
    )
    
    return glasses 