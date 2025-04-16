import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import sys
from datetime import datetime
from PIL import Image
import numpy as np
import io
import glob
from pathlib import Path
import torch
import cv2
import time
import torchvision.transforms as transforms

# Import direct depuis le module local
from app.model import load_model
# Importer aussi le nouveau modèle CNN
# Importer le modèle EfficientNet
from app.efficientnet_model import load_model as load_efficientnet_model, extract_embedding as extract_efficientnet_embedding

# Constantes
CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
OUTPUT_SIZE = 64
BRUSH_SIZE = 3
SYMBOL_TYPE = "symbole"  # Dossier unique pour tous les dessins
IMAGE_SIZE = 64  # Taille des images attendue par le modèle

# CSS pour améliorer l'interface mobile
mobile_css = """
<style>
    /* Styles généraux */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Titre principal */
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sous-titres */
    h2, h3 {
        font-size: 1.3rem !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Boutons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
    }
    
    /* Canvas de dessin */
    .canvas-container {
        margin: 0 auto;
        display: block;
        border: 2px solid #4a4a4a;
        border-radius: 10px;
    }
    
    /* Images */
    img {
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Zone de résultat */
    .prediction-box {
        margin-top: 1rem;
        padding: 0.8rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Score */
    .score-label {
        font-size: 0.85rem;
        color: #555;
    }
    
    /* Ajustements pour écran très petit */
    @media (max-width: 480px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
</style>
"""

# Charger le modèle
@st.cache_resource
def load_model_only(use_efficientnet=True, use_cnn=True):
    """Charge le modèle pour la reconnaissance des gravures"""
    try:
        # Utiliser un expander pour cacher les logs par défaut
        with st.expander("Détails du chargement du modèle (cliquez pour voir)", expanded=False):
            st.info("Chargement du modèle en cours...")
        
        # Obtenir le chemin absolu du répertoire principal du projet
        project_dir = os.path.abspath(os.path.dirname(__file__))
        
        # Définir le device
        device = "cuda" else:  # siamese ou autre
        # Transformations basiques
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
    
    # Appliquer les transformations
    img_tensor = transform(image)
    
    # Ajouter la dimension du batch
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

@st.cache_data
def load_reference_images(_model, _device, _model_type):
    """Charge les images de référence et génère leurs embeddings"""
    # Utiliser un expander pour cacher les logs par défaut
    with st.expander("Détails du chargement (cliquez pour voir)", expanded=False):
        st.info("Chargement des images de référence...")
    
    reference_images = []
    
    # Obtenir le chemin absolu du répertoire principal du projet
    project_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Chemins des dossiers contenant les images
    processed_dir = os.path.join(project_dir, "data", "processed")
    augmented_dir = os.path.join(project_dir, "data", "augmented_gravures")
    raw_dir = os.path.join(project_dir, "data", "raw_gravures")
    
    # Liste des dossiers à vérifier
    dirs_to_check = [processed_dir, augmented_dir, raw_dir]
    
    def process_directory(base_dir):
        """Traite un dossier pour extraire les images et leurs embeddings"""
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("reference_images")
        
        elelif model_type == "efficientnet":
                query_embedding = model.forward_one(image_tensor.to(device)).cpu().numpy()[0]
        
        # Trouver les images similaires
        if reference_images:
            similarity_results = find_similar_images(query_embedding, reference_images, top_n=top_n)
            
            # Afficher la classe la plus probable dans une boîte stylisée
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; margin-top: 0;'>Résultat: {similarity_results['most_likely_class']}</h2>", unsafe_allow_html=True)
            
            # Afficher les 3 meilleurs scores de classe
            st.markdown("<p style='font-weight: bold;'>Meilleurs scores:</p>", unsafe_allow_html=True)
            top_classes = list(similarity_results['class_votes'].items())[:3]
            for class_name, score in top_classes:
                # Utiliser des barres de progression pour visualiser les scores
                st.markdown(f"<span class='score-label'>{class_name}</span>", unsafe_allow_html=True)
                st.progress(min(score * 1.5, 1.0))  # Multiplier par 1.5 pour mieux visualiser
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Afficher les images les plus similaires
            st.markdown("<h3 style='text-align: center;'>Images similaires</h3>", unsafe_allow_html=True)
            
            # Afficher 2 images par ligne sur mobile
            for i in range(0, len(similarity_results['top_matches']), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(similarity_results['top_matches']):
                        match = similarity_results['top_matches'][i + j]
                        with cols[j]:
                            img = Image.open(match['path'])
                            st.image(img, caption=match['class'], use_column_width=True)
                            st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>Score: {match['similarity']:.2f}</p>", unsafe_allow_html=True)
        else:
            st.warning("Aucune image de référence disponible pour la comparaison.")
    
    # Afficher des informations sur l'application en bas
    with st.expander("À propos", expanded=False):
        st.write("""
        Cette application utilise un modèle d'intelligence artificielle pour reconnaître 
        des gravures à partir de dessins faits à la main.
        
        Le modèle a été entraîné sur un ensemble de gravures de référence et utilise 
        une technique de deep learning pour la reconnaissance.
        """)
        
        model_type_display = {
            "efficientnet": "EFFICIENTNET + TRIPLET LOSS",
            "cnn": "CNN AVEC CLASSIFICATION",
            "siamese": "RÉSEAU SIAMOIS"
        }.get(model_type, model_type.upper())
        
        st.write(f"**Modèle:** {model_type_display}")
        st.write(f"**Images de référence:** {len(reference_images)}")
        st.write(f"**Calcul:** {device.upper()}")

if __name__ == "__main__":
    ensure_directories()
    main() 