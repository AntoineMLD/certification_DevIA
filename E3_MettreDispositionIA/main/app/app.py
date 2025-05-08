import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import sys
import time

# Ajout du répertoire parent au chemin Python
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

from models.efficientnet_triplet import EfficientNetEmbedding

# --- Config ---
MODEL_PATH = os.path.join(main_dir, "models", "efficientnet_triplet.pth")
REFERENCE_DIR = os.path.join(main_dir, "data", "oversampled_gravures")
EMBEDDING_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
STROKE_WIDTH = 3  # Épaisseur du trait du pinceau
NUM_RESULTS = 10  # Nombre de résultats à afficher

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Load Model ---
@st.cache_resource
def load_model():
    model = EfficientNetEmbedding(embedding_dim=EMBEDDING_DIM, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- Load Reference Embeddings ---
@st.cache_data
def load_reference_embeddings(_model):
    refs = []
    # Utiliser un dictionnaire pour éviter les doublons
    class_embeddings = {}
    
    for cls in os.listdir(REFERENCE_DIR):
        cls_path = os.path.join(REFERENCE_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
            
        # Chercher d'abord une image avec le même nom que le dossier (en .jpg ou .png)
        preferred_img_path_jpg = os.path.join(cls_path, f"{cls}.jpg")
        preferred_img_path_png = os.path.join(cls_path, f"{cls}.png")
        
        if os.path.exists(preferred_img_path_png):
            # Utiliser l'image .png avec le même nom que le dossier
            img_path = preferred_img_path_png
        elif os.path.exists(preferred_img_path_jpg):
            # Utiliser l'image .jpg avec le même nom que le dossier
            img_path = preferred_img_path_jpg
        else:
            # Sinon, prendre la première image disponible (.jpg ou .png)
            img_paths_jpg = glob.glob(os.path.join(cls_path, "*.jpg"))
            img_paths_png = glob.glob(os.path.join(cls_path, "*.png"))
            img_paths = img_paths_jpg + img_paths_png
            
            if not img_paths:
                continue
            img_path = img_paths[0]
        
        try:
            img = Image.open(img_path).convert("L")  # Convertir en niveaux de gris
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                emb = _model.forward_one(img_tensor).cpu().numpy()[0]
                
            # Stocker l'embedding et l'image
            class_embeddings[cls] = (emb, img, img_path)
        except Exception as e:
            continue
    
    # Convertir le dictionnaire en liste
    for cls, (emb, img, path) in class_embeddings.items():
        refs.append((emb, cls, img, path))
        
    return refs

# --- UI ---
st.title("Gravure à main levée - Recherche intelligente")

st.write("Dessinez une gravure ci-dessous pour la comparer aux références existantes.")

# Initialiser les variables de session
if 'results' not in st.session_state:
    st.session_state.results = None

# Dessin utilisateur
canvas_result = st_canvas(
    fill_color="rgba(255,255,255,0)",
    stroke_width=STROKE_WIDTH,
    stroke_color="black",
    background_color="white",
    width=IMAGE_SIZE,
    height=IMAGE_SIZE,
    drawing_mode="freedraw",
    key="canvas"
)

# Créer deux colonnes pour les boutons
col1, col2 = st.columns(2)

# Bouton pour effacer le canvas
with col1:
    if st.button("🗑️ Effacer le dessin"):
        st.session_state.results = None
        st.experimental_rerun()

# Bouton pour rechercher les gravures similaires
with col2:
    search_button = st.button("🔍 Rechercher les gravures similaires")

# Traiter le dessin si le bouton de recherche est cliqué
if search_button and canvas_result.image_data is not None:
    img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")  # Convertir en niveaux de gris
    st.image(img, caption="Votre dessin", width=150)
    
    with st.spinner("Recherche des gravures similaires..."):
        # --- Embedding du dessin ---
        model = load_model()
        reference_embeddings = load_reference_embeddings(model)

        user_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            user_emb = model.forward_one(user_tensor).cpu().numpy()

        # --- Similarité ---
        similarities = [
            (cosine_similarity(user_emb, emb.reshape(1, -1))[0][0], cls, ref_img, path)
            for emb, cls, ref_img, path in reference_embeddings
        ]

        top_results = sorted(similarities, key=lambda x: x[0], reverse=True)[:NUM_RESULTS]
        st.session_state.results = top_results

# Afficher les résultats s'ils existent
if st.session_state.results is not None:
    st.subheader(f"Top {NUM_RESULTS} gravures similaires trouvées :")
    
    # Afficher les résultats dans une grille
    # Créer 5 colonnes pour afficher les images sur 2 lignes
    for i in range(0, NUM_RESULTS, 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(st.session_state.results):
                score, cls, ref_img, path = st.session_state.results[i + j]
                with cols[j]:
                    st.image(ref_img, caption=f"{cls} - Similarité : {score:.2f}", width=160)