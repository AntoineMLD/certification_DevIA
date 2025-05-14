import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import load_model, preprocess_image, get_embedding
from app.similarity_search import get_top_matches, load_references, reference_embeddings
from app.database import find_matching_verres, get_verre_details
import io 
from PIL import Image
from datetime import datetime, timedelta
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de sécurité
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-jwt")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Configuration OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialisation du limiteur de taux
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les origines exactes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

model = load_model()
# Charger les références au démarrage
load_references(model)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email != ADMIN_EMAIL:
            raise credentials_exception
        return email
    except JWTError:
        raise credentials_exception

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_EMAIL or form_data.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/embedding")
@limiter.limit("5/minute")
async def get_image_embedding(request: Request, file: UploadFile = File(...), token: str = Depends(verify_token)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    embedding = get_embedding(model, img)
    return {"embedding": embedding.tolist()}

@app.post("/match")
@limiter.limit("5/minute")
async def get_best_match(request: Request, file: UploadFile = File(...), token: str = Depends(verify_token)):
    logger.info("Début du traitement de la requête /match")
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    embedding = get_embedding(model, img)
    logger.info(f"Embedding calculé. Recherche des meilleures correspondances parmi {len(reference_embeddings)} références")
    matches = get_top_matches(embedding)
    logger.info(f"Correspondances trouvées: {matches}")
    return {"matches": matches}

@app.post("/search_tags")
@limiter.limit("10/minute")
async def search_tags(request: Request, tags: list[str], token: str = Depends(verify_token)):
    logger.info(f"Recherche de verres pour les tags: {tags}")
    results = find_matching_verres(tags)
    logger.info(f"Résultats trouvés: {len(results)} verres")
    return {"results": results}

@app.get("/verre/{verre_id}")
@limiter.limit("20/minute")
async def get_verre(request: Request, verre_id: int, token: str = Depends(verify_token)):
    """
    Récupère les détails complets d'un verre par son ID
    """
    logger.info(f"Récupération des détails du verre ID: {verre_id}")
    verre = get_verre_details(verre_id)
    
    if verre:
        logger.info(f"Détails du verre trouvés: {verre.get('nom', 'inconnu')}")
        return {"verre": verre}
    else:
        logger.warning(f"Verre non trouvé avec ID: {verre_id}")
        return {"error": "Verre non trouvé"} 
