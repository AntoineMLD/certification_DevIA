import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from api_client import get_similar_tags_from_api, get_verres_by_tags_api, get_verre_details_api, validate_prediction
import pandas as pd
import numpy as np
import os
import requests
from io import BytesIO
from auth import check_authentication, logout
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérifier l'authentification avant tout
check_authentication()

# Définir le chemin absolu vers le répertoire des images de référence
REF_IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../api/data/oversampled_gravures"))

# --- Configuration de base ---
st.set_page_config(page_title="Recherche de Verres", layout="wide")

# Bouton de déconnexion dans la barre latérale
with st.sidebar:
    if st.button("📤 Déconnexion"):
        logout()

st.title("🔍 Recherche de verres par symboles et tags via API")

IMAGE_SIZE = 224
STROKE_WIDTH = 3
NUM_RESULTS = 10

# --- Initialisation session state ---
for key in ["results", "selected_tags", "matched_verres", "selected_verre_id", "selected_verre_details", "search_performed"]:
    if key not in st.session_state:
        st.session_state[key] = [] if 'tags' in key or 'verres' in key else None
        if 'performed' in key:
            st.session_state[key] = False

# --- Colonne de gauche : dessin, sélection et recherche ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🎨 Dessinez une gravure")
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

    if st.button("🗑️ Effacer le dessin"):
        st.session_state.results = None
        st.experimental_rerun()

    if canvas_result.image_data is not None:
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
        st.image(img, caption="Votre dessin", width=150)

        if st.button("🔍 Rechercher les symboles similaires"):
            with st.spinner("Envoi à l'API pour comparaison..."):
                try:
                    results = get_similar_tags_from_api(img)
                    st.session_state.results = results
                    st.success("Résultats reçus")
                except Exception as e:
                    st.error(f"Erreur API : {e}")

    if st.session_state.results:
        st.subheader(f"Top {NUM_RESULTS} symboles similaires trouvés")
        cols = st.columns(3)
        for idx, res in enumerate(st.session_state.results[:NUM_RESULTS]):
            with cols[idx % 3]:
                st.write(f"Tag : {res['class']} — Similarité : {res['similarity']:.2f}")
                
                # Afficher l'image correspondante
                try:
                    # Chemin de l'image de référence
                    ref_img_path = f"{REF_IMG_DIR}/{res['class']}/{res['class']}.png"
                    ref_img = Image.open(ref_img_path)
                    st.image(ref_img, width=100, caption=res['class'])
                except Exception as e:
                    st.warning(f"Image non trouvée: {e}")
                
                if st.button("Ajouter", key=f"add_{idx}"):
                    try:
                        logger.info(f"[UI] Clic sur le bouton Ajouter pour la classe: {res['class']}")
                        # Valider la prédiction auprès de l'API
                        validate_prediction(res['class'])
                        logger.info(f"[UI] Validation réussie pour la classe: {res['class']}")
                        # Ajouter aux tags sélectionnés
                        if res['class'] not in st.session_state.selected_tags:
                            st.session_state.selected_tags.append(res['class'])
                            logger.info(f"[UI] Tag {res['class']} ajouté aux tags sélectionnés")
                            st.experimental_rerun()
                    except Exception as e:
                        logger.error(f"[UI] Erreur lors de la validation de {res['class']}: {str(e)}")
                        st.error(f"Erreur lors de la validation: {e}")
                        if res['class'] not in st.session_state.selected_tags:
                            st.session_state.selected_tags.append(res['class'])
                            st.experimental_rerun()

    # --- Tags sélectionnés ---
    st.subheader("🏷️ Tags sélectionnés")
    if st.session_state.selected_tags:
        st.write(", ".join(st.session_state.selected_tags))
        if st.button("🧹 Réinitialiser les tags"):
            st.session_state.selected_tags = []
            st.session_state.matched_verres = []
            st.session_state.search_performed = False
            st.experimental_rerun()
    else:
        st.info("Aucun tag sélectionné.")

    # --- Saisie manuelle de tags ---
    st.text_input("Ajouter les lettres ou chiffres de la gravure", key="manual_input")
    if st.button("➕ Ajouter ces tags"):
        # Si l'entrée contient des virgules, la diviser en tags séparés
        if ',' in st.session_state.manual_input:
            manual_tags = [t.strip() for t in st.session_state.manual_input.split(',') if t.strip()]
        else:
            # Sinon, considérer l'entrée entière comme un seul tag
            manual_tags = [st.session_state.manual_input.strip()] if st.session_state.manual_input.strip() else []
            
        st.session_state.selected_tags.extend([t for t in manual_tags if t not in st.session_state.selected_tags])
        st.experimental_rerun()

    # --- Rechercher les verres associés ---
    if st.button("📦 Rechercher les verres correspondants") and st.session_state.selected_tags:
        with st.spinner("Recherche dans la base de données..."):
            try:
                verres = get_verres_by_tags_api(st.session_state.selected_tags)
                st.session_state.matched_verres = verres
                st.session_state.search_performed = True
            except Exception as e:
                st.error(f"Erreur API : {e}")

    # --- Affichage des résultats ---
    if st.session_state.search_performed:
        if st.session_state.matched_verres:
            st.subheader(f"🔎 {len(st.session_state.matched_verres)} verres trouvés")
            
            # Créer un DataFrame avec plus d'informations
            df = pd.DataFrame([{
                "ID": v.get("id"),
                "Fournisseur": v.get("fournisseur", ""),
                "Nom": v.get("nom", ""),
                "Variante": v.get("variante", ""),
                "Hauteur Min": v.get("hauteur_min", ""),
                "Hauteur Max": v.get("hauteur_max", ""),
                "Indice": v.get("indice", ""),
                "Tags": ", ".join(v.get("tags", []))
            } for v in st.session_state.matched_verres])
            
            # Afficher le tableau plus complet
            st.dataframe(df)
            
            # Afficher les images si disponibles
            st.subheader("Images des verres trouvés")
            cols = st.columns(3)  # 3 images par ligne
            col_idx = 0
            
            for idx, verre in enumerate(st.session_state.matched_verres):
                gravure = verre.get("gravure", "")
                if gravure and isinstance(gravure, str) and gravure.startswith(("http://", "https://")):
                    with cols[col_idx % 3]:
                        st.markdown(f"**Verre #{verre['id']} - {verre.get('fournisseur', '')} {verre.get('variante', '')}**")
                        try:
                            # Essayer d'afficher l'image
                            st.image(gravure, width=150)
                            col_idx += 1
                        except Exception as e:
                            st.error(f"Erreur d'affichage de l'image: {e}")
                            # Afficher l'URL en cas d'échec
                            st.markdown(f"[Voir l'image]({gravure})")
        else:
            st.warning("Aucun verre ne correspond aux tags.")

# --- Colonne de droite : détail d'un verre ---
with col_right:
    st.markdown("## Détails du verre sélectionné")
    
    # Ajouter une fonction pour afficher les détails d'un verre
    def display_verre_details(verre):
        st.markdown(f"### {verre.get('nom', 'Verre')} {verre.get('variante', '')}")
        
        # Informations générales
        st.markdown("#### Informations générales")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**ID:** {verre.get('id')}")
            st.markdown(f"**Fournisseur:** {verre.get('fournisseur_nom', 'Non spécifié')}")
            st.markdown(f"**Indice:** {verre.get('indice', 'Non spécifié')}")
        
        with col2:
            # Hauteurs
            if verre.get('hauteur_min') or verre.get('hauteur_max'):
                hauteur_min = verre.get('hauteur_min', 'N/A')
                hauteur_max = verre.get('hauteur_max', 'N/A')
                st.markdown(f"**Hauteur:** {hauteur_min} - {hauteur_max}")
            
            # URL source
            if verre.get('url_source'):
                st.markdown(f"**Source:** [Lien]({verre.get('url_source')})")
        
        # Matériau
        if verre.get('materiau'):
            st.markdown("#### Matériau")
            materiau = verre['materiau']
            st.markdown(f"**Nom:** {materiau.get('nom', 'Non spécifié')}")
            st.markdown(f"**Description:** {materiau.get('description', 'Non spécifié')}")
        
        # Série
        if verre.get('serie'):
            st.markdown("#### Série")
            serie = verre['serie']
            st.markdown(f"**Nom:** {serie.get('nom', 'Non spécifié')}")
            st.markdown(f"**Description:** {serie.get('description', 'Non spécifié')}")
        
        # Traitements
        if verre.get('traitements'):
            st.markdown("#### Traitements")
            traitements = verre['traitements']
            for t in traitements:
                st.markdown(f"- **{t.get('nom', 'Sans nom')}:** {t.get('description', 'Sans description')}")
        
        # Tags
        if verre.get('tags'):
            st.markdown("#### Tags")
            st.markdown(f"{', '.join(verre.get('tags', []))}")
        
        # Image de gravure
        gravure = verre.get('gravure', '')
        if gravure and isinstance(gravure, str) and gravure.startswith(("http://", "https://")):
            st.markdown("### Image")
            try:
                st.image(gravure, width=250)
            except:
                st.markdown(f"[Voir l'image]({gravure})")
    
    # Permettre la sélection d'un verre depuis le tableau
    if st.session_state.matched_verres:
        verre_ids = [v["id"] for v in st.session_state.matched_verres]
        verre_names = [f"{v.get('id')} - {v.get('fournisseur', '')} {v.get('nom', '')} {v.get('variante', '')}" for v in st.session_state.matched_verres]
        
        selected_verre_name = st.selectbox("Sélectionnez un verre pour voir les détails", [""] + verre_names)
        
        if selected_verre_name:
            try:
                selected_idx = verre_names.index(selected_verre_name)
                selected_verre_id = verre_ids[selected_idx]
                
                with st.spinner("Chargement des détails complets..."):
                    # Récupérer les détails complets du verre depuis l'API
                    verre_details = get_verre_details_api(selected_verre_id)
                    
                display_verre_details(verre_details)
            except Exception as e:
                st.error(f"Erreur lors du chargement des détails: {e}")
    else:
        st.info("Aucun verre trouvé. Recherchez des verres pour voir les détails.")
