from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from ..config import settings

# Créer la base de données
engine = create_engine(settings.DATABASE_URL)
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
    nom = Column(String, nullable=False)
    variante = Column(String)
    hauteur_min = Column(Integer)
    hauteur_max = Column(Integer)
    indice = Column(Float)
    gravure = Column(String)
    url_source = Column(String)
    
    # Clés étrangères
    fournisseur_id = Column(Integer, ForeignKey("fournisseurs.id"))
    materiau_id = Column(Integer, ForeignKey("materiaux.id"))
    gamme_id = Column(Integer, ForeignKey("gammes.id"))
    serie_id = Column(Integer, ForeignKey("series.id"))
    
    # Relations
    fournisseur = relationship("Fournisseur", back_populates="verres")
    materiau = relationship("Materiau", back_populates="verres")
    gamme = relationship("Gamme", back_populates="verres")
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