import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from api_client import get_similar_tags_from_api, get_verres_by_tags_api, get_verre_details_api, get_verre_staging_details_api, validate_prediction
import pandas as pd
import numpy as np
import os
import requests
import glob
import re
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

# Fonction pour trouver une image pour un symbole donné
def find_symbol_image(symbol_name):
    """
    Cherche une image pour un symbole donné en essayant plusieurs formats et méthodes.
    
    Args:
        symbol_name (str): Nom du symbole/classe
        
    Returns:
        PIL.Image ou None: L'image trouvée ou None si aucune image n'est trouvée
    """
    # Vérifier si le nom du symbole est valide
    if not symbol_name or not isinstance(symbol_name, str) or symbol_name.lower() == 'inconnu':
        logger.warning(f"Nom de symbole invalide: {symbol_name}")
        return None
    
    logger.info(f"Recherche d'image pour le symbole: {symbol_name}")
    
    # Chemins possibles à essayer
    paths_to_try = []
    
    # 1. Essayer les formats courants dans le sous-dossier du symbole
    for ext in ['.png', '.jpg', '.jpeg']:
        paths_to_try.append(os.path.join(REF_IMG_DIR, symbol_name, f"{symbol_name}{ext}"))
    
    # 2. Essayer les formats courants directement dans le dossier de référence
    for ext in ['.png', '.jpg', '.jpeg']:
        paths_to_try.append(os.path.join(REF_IMG_DIR, f"{symbol_name}{ext}"))
    
    # Essayer chaque chemin
    for path in paths_to_try:
        if os.path.exists(path):
            logger.info(f"Image trouvée: {path}")
            try:
                return Image.open(path)
            except Exception as e:
                logger.error(f"Erreur lors de l'ouverture de l'image {path}: {e}")
                # Continuer avec le prochain chemin
    
    # Si aucun chemin exact n'a fonctionné, essayer de trouver n'importe quelle image dans le dossier du symbole
    symbol_dir = os.path.join(REF_IMG_DIR, symbol_name)
    if os.path.exists(symbol_dir) and os.path.isdir(symbol_dir):
        logger.info(f"Recherche d'images dans le dossier: {symbol_dir}")
        # Chercher tous les fichiers image
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(symbol_dir, ext)))
        
        # Utiliser le premier fichier trouvé s'il y en a
        if image_files:
            logger.info(f"Image trouvée dans le dossier: {image_files[0]}")
            try:
                return Image.open(image_files[0])
            except Exception as e:
                logger.error(f"Erreur lors de l'ouverture de l'image {image_files[0]}: {e}")
    
    logger.warning(f"Aucune image trouvée pour le symbole: {symbol_name}")
    return None

# Fonction pour extraire le bon nom d'un verre
def extract_verre_name(verre_data):
    """
    Extrait le nom correct d'un verre à partir des données
    
    Args:
        verre_data (dict): Données du verre
        
    Returns:
        str: Nom correctement formaté du verre
    """
    # Essayer d'abord glass_name puis nom sinon utiliser 'Non spécifié'
    nom_complet = verre_data.get('glass_name', verre_data.get('nom', 'Non spécifié'))
    
    # Pour ce cas particulier, on veut utiliser le nom complet
    # puisqu'il n'y a pas de format numérique X.YY
    return nom_complet

# Fonction pour récupérer les détails complets d'un verre (y compris le glass_name de staging)
def get_full_verre_details(verre_id):
    """
    Récupère les détails complets d'un verre, y compris les informations de la table staging
    
    Args:
        verre_id (int): ID du verre
        
    Returns:
        dict: Détails complets du verre combinant les données des tables verres et staging
    """
    try:
        # Récupérer les détails de base du verre
        logger.info(f"Récupération des détails de base pour le verre {verre_id}")
        verre_details = get_verre_details_api(verre_id)
        
        if not verre_details:
            logger.error(f"Impossible de récupérer les détails de base pour le verre {verre_id}")
            return None
        
        logger.info(f"Détails de base récupérés pour le verre {verre_id}: {verre_details.get('nom', 'N/A')}")
        
        # Tenter de récupérer les détails de staging qui contient glass_name
        try:
            logger.info(f"Tentative de récupération des détails de staging pour le verre {verre_id}")
            staging_details = get_verre_staging_details_api(verre_id)
            
            # Journaliser les détails de staging bruts
            logger.info(f"Détails de staging bruts reçus: {staging_details}")
            
            # Si on a récupéré les détails de staging, les fusionner avec les détails de base
            if staging_details and isinstance(staging_details, dict):
                glass_name = staging_details.get('glass_name')
                logger.info(f"Détails staging trouvés pour verre {verre_id}: glass_name = {glass_name}")
                
                # Fusionner les deux dictionnaires (les clés de staging ont priorité)
                for key, value in staging_details.items():
                    if value is not None:  # Ne pas écraser avec des valeurs None
                        verre_details[key] = value
                        
                # Vérifier que le champ glass_name a bien été fusionné
                logger.info(f"Après fusion, le champ glass_name est: {verre_details.get('glass_name', 'Non défini')}")
            else:
                logger.warning(f"Aucun détail staging trouvé pour le verre {verre_id} ou format inattendu")
        except Exception as e:
            logger.warning(f"Impossible de récupérer les détails de staging pour le verre {verre_id}: {str(e)}")
            logger.exception("Détail de l'erreur:")
        
        return verre_details
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détails du verre {verre_id}: {str(e)}")
        logger.exception("Détail de l'erreur:")
        return None

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
                    logger.error(f"Erreur lors de l'appel API: {e}", exc_info=True)

    if st.session_state.results:
        st.subheader(f"Top {NUM_RESULTS} symboles similaires trouvés")
        cols = st.columns(3)
        for idx, res in enumerate(st.session_state.results[:NUM_RESULTS]):
            with cols[idx % 3]:
                # Extraction du nom de classe avec méthode get sécurisée
                class_name = res.get('class', 'inconnu')
                similarity = res.get('similarity', 0.0)
                
                # Journaliser l'information pour aider au débogage
                logger.info(f"Traitement du résultat {idx}: class_name={class_name}, similarity={similarity}")
                
                st.write(f"Tag : {class_name} — Similarité : {similarity:.2f}")
                
                # Afficher l'image correspondante
                try:
                    # Trouver et afficher l'image
                    ref_img = find_symbol_image(class_name)
                    if ref_img:
                        st.image(ref_img, width=100, caption=class_name)
                    else:
                        st.warning(f"Aucune image trouvée pour {class_name}")
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de l'image pour {class_name}: {e}", exc_info=True)
                    st.warning(f"Erreur lors du chargement de l'image: {e}")
                
                if st.button("Ajouter", key=f"add_{idx}"):
                    try:
                        logger.info(f"[UI] Clic sur le bouton Ajouter pour la classe: {class_name}")
                        
                        # Vérifier si class_name est valide avant de l'envoyer à l'API
                        if not class_name or class_name == 'inconnu':
                            logger.warning(f"[UI] Tentative de validation d'une classe non valide: {class_name}")
                            st.error("Impossible de valider une classe inconnue ou vide")
                            continue
                            
                        # Valider la prédiction auprès de l'API
                        validate_prediction(class_name)
                        logger.info(f"[UI] Validation réussie pour la classe: {class_name}")
                        
                        # Ajouter aux tags sélectionnés
                        if class_name not in st.session_state.selected_tags:
                            st.session_state.selected_tags.append(class_name)
                            logger.info(f"[UI] Tag {class_name} ajouté aux tags sélectionnés")
                            st.experimental_rerun()
                    except Exception as e:
                        logger.error(f"[UI] Erreur lors de la validation de {class_name}: {str(e)}")
                        st.error(f"Erreur lors de la validation: {e}")
                        # Quand même ajouter le tag si l'API échoue, mais que le tag est valide
                        if class_name != 'inconnu' and class_name not in st.session_state.selected_tags:
                            st.session_state.selected_tags.append(class_name)
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
        # Récupérer le nom brut du verre depuis la table staging
        nom_staging = verre.get('glass_name')
        
        # Afficher le nom complet du verre comme titre principal
        if nom_staging:
            st.markdown(f"### {nom_staging}")
        else:
            st.warning("⚠️ glass_name non disponible dans la table staging")
            nom_verre = verre.get('nom', 'Verre')
            variante = verre.get('variante', '')
            st.markdown(f"### {nom_verre} {variante}")
            
            # Essayons de montrer ce qui est disponible pour le débogage
            st.expander("Déboguer les données disponibles").write(f"Clés disponibles: {', '.join(verre.keys())}")
        
        # Afficher un bouton pour recharger les informations
        if st.button("🔄 Actualiser les données"):
            verre_id = verre.get("id")
            if verre_id:
                st.info(f"Actualisation des données pour le verre #{verre_id}...")
                verre_details = get_full_verre_details(verre_id)
                st.session_state.selected_verre_details = verre_details
                st.experimental_rerun()
        
        # Informations générales
        st.markdown("#### Informations générales")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**ID:** {verre.get('id')}")
            st.markdown(f"**Fournisseur:** {verre.get('fournisseur_nom', 'Non spécifié')}")
            st.markdown(f"**Nom de base:** {verre.get('nom', 'Non spécifié')}")
            st.markdown(f"**Nom complet (staging):** {nom_staging if nom_staging else '⚠️ Non disponible'}")
            st.markdown(f"**Variante:** {verre.get('variante', '')}")
            st.markdown(f"**Indice:** {verre.get('indice', 'Non spécifié')}")
            
        with col2:
            # Debug info avec l'API
            db_info = st.expander("Informations de débogage")
            with db_info:
                st.markdown(f"**ID verre utilisé pour staging:** {verre.get('id')}")
                st.markdown(f"**ID interne:** {verre.get('id_interne', 'Non disponible')}")
                
                # Afficher un bouton pour tester directement l'API staging
                if st.button("🔍 Tester l'API Staging"):
                    verre_id = verre.get("id")
                    if verre_id:
                        try:
                            staging_details = get_verre_staging_details_api(verre_id)
                            st.json(staging_details)
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
            
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
        # Chercher d'abord les détails complets pour chaque verre
        verre_details_list = []
        for verre in st.session_state.matched_verres:
            try:
                # Récupérer les détails complets du verre via l'API
                verre_id = verre.get('id')
                verre_details = get_full_verre_details(verre_id)
                
                # Journaliser les détails pour le débogage
                logger.info(f"Détails récupérés pour le verre {verre_id}: glass_name: {verre_details.get('glass_name', 'Non spécifié')}")
                
                verre_details_list.append(verre_details)
            except Exception as e:
                # En cas d'erreur, utiliser les informations limitées disponibles
                logger.error(f"Erreur lors de la récupération des détails du verre {verre.get('id')}: {e}")
                verre_details_list.append(verre)
        
        # Créer une liste d'options pour le sélecteur avec le nom complet du verre de staging
        verre_options = []
        for v in verre_details_list:
            verre_id = v.get('id', 'N/A')
            # Récupérer directement le glass_name de staging
            glass_name = v.get('glass_name', v.get('nom', 'Non spécifié'))
            verre_options.append(f"#{verre_id} - {glass_name}")
        
        # Ajouter une option vide en premier
        verre_options.insert(0, "Sélectionnez un verre...")
        
        # Créer le sélecteur
        selected_option = st.selectbox("Sélectionner un verre pour voir les détails", verre_options)
        
        # Si une option valide est sélectionnée
        if selected_option != "Sélectionnez un verre...":
            # Extraire l'ID du verre depuis l'option sélectionnée
            verre_id_str = selected_option.split(" - ")[0].replace("#", "")
            try:
                verre_id = int(verre_id_str)
                
                # Trouver le verre sélectionné dans la liste des détails complets
                selected_verre = next((v for v in verre_details_list if v.get("id") == verre_id), None)
                
                if selected_verre:
                    # Les détails complets sont déjà disponibles, pas besoin d'appel API supplémentaire
                    st.session_state.selected_verre_details = selected_verre
                    # Afficher les détails
                    display_verre_details(selected_verre)
                else:
                    # Si pour une raison quelconque nous n'avons pas les détails, essayer de les récupérer
                    try:
                        verre_details = get_full_verre_details(verre_id)
                        st.session_state.selected_verre_details = verre_details
                        # Afficher les détails
                        display_verre_details(verre_details)
                    except Exception as e:
                        st.error(f"Erreur lors de la récupération des détails du verre: {e}")
                        # Chercher dans la liste originale
                        fallback_verre = next((v for v in st.session_state.matched_verres if v.get("id") == verre_id), None)
                        if fallback_verre:
                            display_verre_details(fallback_verre)
                        else:
                            st.error("Impossible de trouver les détails du verre sélectionné.")
            except ValueError:
                st.error(f"Erreur lors de la conversion de l'ID du verre: {verre_id_str}")
    else:
        if st.session_state.search_performed:
            st.info("Aucun verre trouvé. Essayez avec d'autres tags.")
        else:
            st.info("Recherchez des verres pour afficher les détails.")
