from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Table, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from ..config import settings
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer la base de données avec des paramètres optimisés pour PostgreSQL
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Table d'association pour verres et traitements
verres_traitements = Table(
    'verres_traitements',
    Base.metadata,
    Column('verre_id', Integer, ForeignKey('verres.id'), primary_key=True),
    Column('traitement_id', Integer, ForeignKey('traitements.id'), primary_key=True)
)

class Traitement(Base):
    """Modèle pour la table des traitements"""
    __tablename__ = "traitements"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False)
    type = Column(String, nullable=False)
    
    # Relation avec les verres
    verres = relationship("Verre", secondary=verres_traitements, back_populates="traitements")

class Fournisseur(Base):
    """Modèle pour la table des fournisseurs"""
    __tablename__ = "fournisseurs"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, unique=True)
    
    # Relation avec les verres
    verres = relationship("Verre", back_populates="fournisseur")

class Materiau(Base):
    """Modèle pour la table des matériaux"""
    __tablename__ = "materiaux"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, unique=True)
    
    # Relation avec les verres
    verres = relationship("Verre", back_populates="materiau")

class Gamme(Base):
    """Modèle pour la table des gammes"""
    __tablename__ = "gammes"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, unique=True)
    
    # Relation avec les verres
    verres = relationship("Verre", back_populates="gamme")

class Serie(Base):
    """Modèle pour la table des séries"""
    __tablename__ = "series"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, unique=True)
    
    # Relation avec les verres
    verres = relationship("Verre", back_populates="serie")

class Verre(Base):
    """Modèle pour la table des verres"""
    __tablename__ = "verres"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, index=True)
    variante = Column(String)
    hauteur_min = Column(Integer)
    hauteur_max = Column(Integer)
    indice = Column(Float)
    gravure = Column(String)
    url_source = Column(String)
    
    # Clés étrangères avec index
    fournisseur_id = Column(Integer, ForeignKey("fournisseurs.id"), index=True)
    materiau_id = Column(Integer, ForeignKey("materiaux.id"), index=True)
    gamme_id = Column(Integer, ForeignKey("gammes.id"), index=True)
    serie_id = Column(Integer, ForeignKey("series.id"), index=True)
    
    # Relations avec lazy="joined" pour éviter les requêtes N+1
    fournisseur = relationship("Fournisseur", back_populates="verres", lazy="joined")
    materiau = relationship("Materiau", back_populates="verres", lazy="joined")
    gamme = relationship("Gamme", back_populates="verres", lazy="joined")
    serie = relationship("Serie", back_populates="verres")
    traitements = relationship("Traitement", secondary=verres_traitements, back_populates="verres")

# Fonction pour obtenir une session de base de données
def get_db():
    """Crée une nouvelle session de base de données"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 