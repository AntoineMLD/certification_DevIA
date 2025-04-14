from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import datetime
import jwt
from passlib.context import CryptContext
import uvicorn
import numpy as np
from PIL import Image
import io
import torch

# Importer les modules de l'application
from .model import SiameseNetwork, load_model
from .embeddings_manager import EmbeddingsManager
from .ui import create_ui
from config import (
    DEFAULT_MODEL_PATH,
    EMBEDDINGS_PATH,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    API_HOST,
    API_PORT,
    API_RELOAD,
    DEFAULT_USERS
)

# Variables globales pour le modèle et le gestionnaire d'embeddings
model = None
embeddings_manager = None

# Initialisation de l'application FastAPI
app = FastAPI(title="API de Reconnaissance de Gravures Optiques")

# Ajouter le middleware CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration sécurité
# Utilisation d'une valeur par défaut uniquement pour le développement
# En production, utiliser une variable d'environnement
SECRET_KEY = SECRET_KEY
ALGORITHM = ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = ACCESS_TOKEN_EXPIRE_MINUTES

# Gestionnaire de mots de passe pour le hachage
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialisation de l'authentification OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Modèle Pydantic pour les utilisateurs
class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

# Modèle Pydantic pour les détails de l'utilisateur
class UserInDB(User):
    hashed_password: str

# Base d'utilisateurs simulée (remplacer par une vraie base de données en production)
fake_users_db = DEFAULT_USERS

# Modèle Pydantic pour le token
class Token(BaseModel):
    access_token: str
    token_type: str

# Modèle Pydantic pour une gravure
class Gravure(BaseModel):
    id: int
    code: str
    indice: float

# Fonction pour vérifier un mot de passe
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Fonction pour obtenir un utilisateur
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

# Fonction pour authentifier un utilisateur
def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Fonction pour créer un token d'accès
def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Fonction pour obtenir l'utilisateur actuel
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return user

# Endpoint pour l'authentification et obtention du token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Fonction de prétraitement d'image
def preprocess_image(image):
    # Convertir en niveaux de gris
    if image.mode != 'L':
        image = image.convert('L')
    
    # Redimensionner à une taille uniforme (ex: 64x64)
    image = image.resize((64, 64))
    
    # Normaliser les valeurs des pixels
    img_array = np.array(image) / 255.0
    
    return img_array

# Endpoint pour reconnaître une gravure
@app.post("/recognize")
async def recognize_gravure(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        # Vérifier que le modèle et le gestionnaire d'embeddings sont initialisés
        if model is None:
            raise HTTPException(status_code=500, detail="Le modèle n'a pas été initialisé correctement")
        if embeddings_manager is None:
            raise HTTPException(status_code=500, detail="Le gestionnaire d'embeddings n'a pas été initialisé correctement")
            
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prétraiter l'image
        processed_image = preprocess_image(image)
        
        # Convertir en tensor
        image_tensor = torch.from_numpy(processed_image).float().unsqueeze(0).unsqueeze(0)
        
        # Calculer l'embedding avec le modèle
        with torch.no_grad():
            embedding = model.forward_one(image_tensor).cpu().numpy()[0]
        
        # Trouver la gravure la plus proche
        results = embeddings_manager.find_closest_gravure(embedding)
        
        if not results:
            raise HTTPException(status_code=404, detail="Aucune gravure correspondante trouvée")
        
        id_gravure, similarity = results[0]
        gravure_info = embeddings_manager.get_gravure_info(id_gravure)
        
        return {
            "id": id_gravure,
            "code": gravure_info.get('code'),
            "indice": gravure_info.get('indice'),
            "score": float(similarity)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour récupérer la liste des gravures
@app.get("/gravures", response_model=List[Gravure])
async def get_gravures(current_user: User = Depends(get_current_user)):
    gravures = embeddings_manager.get_all_gravures()
    return [Gravure(id=g['id'], code=g['code'], indice=g['indice']) for g in gravures]

# Événement de démarrage pour initialiser l'application
@app.on_event("startup")
async def startup_event():
    # Initialiser l'application (modèle et embeddings)
    init_app()
    
    # Créer l'interface Gradio
    ui = create_ui()
    
    # Monter l'interface Gradio
    from fastapi.staticfiles import StaticFiles
    import gradio as gr
    
    # Monter l'application Gradio
    app.mount("/gradio", gr.routes.App.create_app(ui), name="gradio_app")

# Fonction pour initialiser l'application (modèle et embeddings)
def init_app():
    global model, embeddings_manager
    
    # Vérifier si CUDA est disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {DEFAULT_MODEL_PATH}")
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = load_model(DEFAULT_MODEL_PATH, device=device)
        print("Modèle chargé avec succès")
    else:
        # Créer un modèle non entraîné pour la démonstration
        model = SiameseNetwork()
        print("Modèle d'exemple créé (non entraîné)")
    
    # Charger les embeddings
    print(f"Chargement des embeddings depuis {EMBEDDINGS_PATH}")
    embeddings_manager = EmbeddingsManager(embeddings_path=EMBEDDINGS_PATH)
    
    # Si aucun embedding n'a été chargé, créer des embeddings factices pour la démonstration
    if not embeddings_manager.embeddings_dict:
        print("Aucun embedding trouvé. Création d'embeddings factices pour la démonstration.")
        # Créer quelques embeddings factices
        import numpy as np
        embeddings_manager.add_embedding(1, np.random.rand(128), {'code': 'Varilux', 'indice': 1.67, 'filename': 'varilux_1.67.jpg'})
        embeddings_manager.add_embedding(2, np.random.rand(128), {'code': 'Essilor', 'indice': 1.6, 'filename': 'essilor_1.6.jpg'})
        embeddings_manager.add_embedding(3, np.random.rand(128), {'code': 'Zeiss', 'indice': 1.5, 'filename': 'zeiss_1.5.jpg'})
        
        # Sauvegarder les embeddings factices
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        embeddings_manager.save_embeddings(EMBEDDINGS_PATH)
    
    print(f"Application initialisée avec {len(embeddings_manager.embeddings_dict)} embeddings de gravures")

# Pour exécuter l'application en local
if __name__ == "__main__":
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD) 