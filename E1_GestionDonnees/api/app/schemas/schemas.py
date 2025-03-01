from pydantic import BaseModel
from typing import List, Optional

# Schémas de base
class TraitementBase(BaseModel):
    nom: str
    type: str

class FournisseurBase(BaseModel):
    nom: str

class MateriauBase(BaseModel):
    nom: str

class GammeBase(BaseModel):
    nom: str

class SerieBase(BaseModel):
    nom: str

class VerreBase(BaseModel):
    nom: str
    variante: Optional[str] = None
    hauteur_min: Optional[int] = None
    hauteur_max: Optional[int] = None
    indice: Optional[float] = None
    gravure: Optional[str] = None
    url_source: Optional[str] = None

# Schémas pour la création
class TraitementCreate(TraitementBase):
    pass

class FournisseurCreate(FournisseurBase):
    pass

class MateriauCreate(MateriauBase):
    pass

class GammeCreate(GammeBase):
    pass

class SerieCreate(SerieBase):
    pass

class VerreCreate(VerreBase):
    fournisseur_id: Optional[int] = None
    materiau_id: Optional[int] = None
    gamme_id: Optional[int] = None
    serie_id: Optional[int] = None

# Schémas pour la lecture
class Traitement(TraitementBase):
    id: int
    
    class Config:
        from_attributes = True

class Fournisseur(FournisseurBase):
    id: int
    
    class Config:
        from_attributes = True

class Materiau(MateriauBase):
    id: int
    
    class Config:
        from_attributes = True

class Gamme(GammeBase):
    id: int
    
    class Config:
        from_attributes = True

class Serie(SerieBase):
    id: int
    
    class Config:
        from_attributes = True

class Verre(VerreBase):
    id: int
    fournisseur_id: Optional[int] = None
    materiau_id: Optional[int] = None
    gamme_id: Optional[int] = None
    serie_id: Optional[int] = None
    
    class Config:
        from_attributes = True

# Schémas pour les relations
class VerreDetail(Verre):
    fournisseur: Optional[Fournisseur] = None
    materiau: Optional[Materiau] = None
    gamme: Optional[Gamme] = None
    serie: Optional[Serie] = None
    traitements: List[Traitement] = []

# Schémas pour l'authentification
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None 