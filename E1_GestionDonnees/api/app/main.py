from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .config import settings
from .auth import jwt_auth
from .models.database import get_db, Verre
from .schemas import schemas
from sqlalchemy.orm import Session, joinedload, selectinload
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
            selectinload(Verre.fournisseur),
            selectinload(Verre.materiau),
            selectinload(Verre.gamme),
            selectinload(Verre.serie),
            selectinload(Verre.traitements)
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
    # Utiliser une seule requête optimisée avec selectinload
    glass = (
        db.query(Verre)
        .options(
            selectinload(Verre.fournisseur),
            selectinload(Verre.materiau),
            selectinload(Verre.gamme),
            selectinload(Verre.serie),
            selectinload(Verre.traitements)
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

@app.post("/verres", response_model=schemas.VerreDetail)
async def create_glass(
    verre: schemas.VerreCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(jwt_auth.get_current_user)
):
    """
    Crée un nouveau verre dans la base de données
    
    Args:
        verre: Données du verre à créer
        db: Session de base de données
        current_user: Utilisateur authentifié
    Returns:
        VerreDetail: Détails du verre créé
    Raises:
        HTTPException: Si les données sont invalides
    """
    try:
        # Création d'un nouveau verre avec les données reçues
        nouveau_verre = Verre(
            nom=verre.nom,
            variante=verre.variante,
            hauteur_min=verre.hauteur_min,
            hauteur_max=verre.hauteur_max,
            indice=verre.indice,
            gravure=verre.gravure,
            url_source=verre.url_source,
            fournisseur_id=verre.fournisseur_id,
            materiau_id=verre.materiau_id,
            gamme_id=verre.gamme_id,
            serie_id=verre.serie_id
        )
        
        # Ajout et sauvegarde dans la base de données
        db.add(nouveau_verre)
        db.commit()
        db.refresh(nouveau_verre)
        
        # Récupération du verre avec toutes ses relations
        verre_complet = (
            db.query(Verre)
            .options(
                joinedload(Verre.fournisseur),
                joinedload(Verre.materiau),
                joinedload(Verre.gamme),
                joinedload(Verre.serie),
                joinedload(Verre.traitements)
            )
            .filter(Verre.id == nouveau_verre.id)
            .first()
        )
        
        return verre_complet
        
    except Exception as e:
        # En cas d'erreur, on annule la transaction
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la création du verre : {str(e)}"
        )

@app.delete("/verres/{glass_id}")
async def delete_glass(
    glass_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(jwt_auth.get_current_user)
):
    """
    Supprime un verre de la base de données
    
    Args:
        glass_id: ID du verre à supprimer
        db: Session de base de données
        current_user: Utilisateur authentifié
    Returns:
        dict: Message de confirmation
    Raises:
        HTTPException: Si le verre n'est pas trouvé
    """
    try:
        # Recherche du verre à supprimer
        verre = db.query(Verre).filter(Verre.id == glass_id).first()
        
        # Vérification si le verre existe
        if verre is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Verre avec l'ID {glass_id} non trouvé"
            )
        
        # Suppression du verre
        db.delete(verre)
        db.commit()
        
        return {"message": f"Le verre avec l'ID {glass_id} a été supprimé avec succès"}
        
    except HTTPException:
        raise
    except Exception as e:
        # En cas d'erreur, on annule la transaction
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la suppression du verre : {str(e)}"
        ) 