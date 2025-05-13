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
import sqlite3
import json
import pandas as pd

# --- Configuration de base ---
st.set_page_config(page_title="Recherche de Verres", layout="wide")

# Ajout du répertoire parent au chemin Python
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

# Essayer d'importer le modèle
try:
    from models.efficientnet_triplet import EfficientNetEmbedding
except ImportError as e:
    st.error(f"Erreur d'importation du modèle: {e}")

# --- Config ---
MODEL_PATH = os.path.join(main_dir, "models", "efficientnet_triplet.pth")
REFERENCE_DIR = os.path.join(main_dir, "data", "oversampled_gravures")
EMBEDDING_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
STROKE_WIDTH = 3  # Épaisseur du trait du pinceau
NUM_RESULTS = 10  # Nombre de résultats à afficher
DB_PATH = "../../../E1_GestionDonnees/Base_de_donnees/france_optique.db"  # Chemin vers la BDD

# Activer le débogage
DEBUG = True

# --- Vérification de la base de données ---
st.sidebar.markdown("## Diagnostic de la base de données")
db_status = st.sidebar.empty()

if os.path.exists(DB_PATH):
    db_status.success(f"✅ Base de données trouvée : {DB_PATH}")
    
    # Tester la connexion
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Lister les tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        st.sidebar.success(f"✅ Connexion réussie - {len(tables)} tables trouvées")
        st.sidebar.write(f"Tables: {', '.join(table_names)}")
        
        # Vérifier si la table tags existe
        if 'tags' in table_names:
            # Compter les entrées
            cursor.execute("SELECT COUNT(*) FROM tags")
            count = cursor.fetchone()[0]
            st.sidebar.success(f"✅ Table 'tags' trouvée avec {count} enregistrements")
            
            # Exemple de données
            st.sidebar.write("Exemples d'entrées dans la table tags:")
            cursor.execute("SELECT verre_id, tags FROM tags LIMIT 3")
            for verre_id, tags_json in cursor.fetchall():
                try:
                    tags = json.loads(tags_json) if tags_json else []
                    st.sidebar.write(f"- Verre #{verre_id}: {tags}")
                except:
                    st.sidebar.write(f"- Verre #{verre_id}: Format JSON invalide")
        else:
            st.sidebar.error("❌ Table 'tags' non trouvée!")
            
        conn.close()
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de connexion à la base: {e}")
else:
    db_status.error(f"❌ Base de données introuvable: {DB_PATH}")
    st.sidebar.write(f"Chemin absolu: {os.path.abspath(DB_PATH)}")
    st.sidebar.write(f"Répertoire courant: {os.getcwd()}")

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

# --- Search Tags in Database ---
def search_verres_by_tags(tags):
    debug_container = st.empty()
    debug_info = []
    
    debug_info.append(f"🔍 Recherche des verres contenant TOUS les tags suivants: {tags}")
    
    if not os.path.exists(DB_PATH):
        debug_info.append("❌ Base de données non trouvée!")
        debug_container.error("\n".join(debug_info))
        return []
    
    if not tags:
        debug_info.append("⚠️ Aucun tag fourni pour la recherche")
        debug_container.warning("\n".join(debug_info))
        return []
    
    try:
        # Connexion à la base de données
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Récupérer tous les verres avec leurs tags
        query = """
            SELECT v.id, v.nom, v.variante, v.hauteur_min, v.hauteur_max, v.indice, v.gravure, v.url_source, 
                   f.nom as fournisseur_nom, t.tags 
            FROM verres v
            JOIN tags t ON v.id = t.verre_id
            LEFT JOIN fournisseurs f ON v.fournisseur_id = f.id
            WHERE t.tags IS NOT NULL
        """
        cursor.execute(query)
        
        all_verres = cursor.fetchall()
        debug_info.append(f"📋 Nombre total de verres avec tags: {len(all_verres)}")
        
        # Liste pour stocker les résultats
        matched_verres = []
        
        # Pour chaque verre, vérifier s'il contient TOUS les tags recherchés
        for verre in all_verres:
            verre_id, nom, variante, hauteur_min, hauteur_max, indice, gravure, url_source, fournisseur_nom, verre_tags_json = verre
            
            try:
                # Convertir les tags JSON en liste Python
                verre_tags = json.loads(verre_tags_json) if verre_tags_json else []
                
                # Convertir tous les tags du verre en minuscules pour une comparaison insensible à la casse
                verre_tags_lower = [str(tag).lower() for tag in verre_tags]
                
                # Liste pour suivre les correspondances trouvées
                found_matches = []
                
                # Vérifier si CHAQUE tag recherché est présent dans les tags du verre
                all_tags_found = True
                for search_tag in tags:
                    search_tag_lower = search_tag.lower()
                    tag_found = False
                    
                    # Chercher une correspondance pour ce tag spécifique
                    for i, verre_tag_lower in enumerate(verre_tags_lower):
                        if search_tag_lower == verre_tag_lower:
                            found_matches.append((search_tag, verre_tags[i]))
                            tag_found = True
                            break
                    
                    # Si un des tags recherchés n'est pas trouvé, ce verre ne correspond pas à tous les critères
                    if not tag_found:
                        all_tags_found = False
                        break
                
                # Si tous les tags ont été trouvés, ajouter ce verre aux résultats
                if all_tags_found:
                    matched_verres.append({
                        "id": verre_id,
                        "nom": nom,
                        "variante": variante,
                        "hauteur_min": hauteur_min,
                        "hauteur_max": hauteur_max,
                        "indice": indice,
                        "gravure": gravure,
                        "url_source": url_source,
                        "fournisseur": fournisseur_nom,
                        "tags": verre_tags,
                        "matches": found_matches
                    })
                    
                    # Ajouter un log pour le débogage
                    debug_info.append(f"✓ Verre #{verre_id} correspond à TOUS les tags recherchés")
                    
            except json.JSONDecodeError:
                debug_info.append(f"⚠️ Format JSON invalide pour le verre #{verre_id}")
                continue
        
        # Afficher le résumé des résultats
        debug_info.append(f"\n✅ Recherche terminée - {len(matched_verres)} verres correspondent à TOUS les tags")
        debug_container.info("\n".join(debug_info))
        
        conn.close()
        return matched_verres
    except sqlite3.Error as e:
        debug_info.append(f"❌ Erreur SQL: {str(e)}")
        debug_container.error("\n".join(debug_info))
        return []

# --- Get Verre Details ---
def get_verre_details(verre_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Récupérer les détails du verre
        cursor.execute("""
            SELECT v.id, v.nom, v.variante, v.gravure, v.indice, v.url_source, 
                   f.nom as fournisseur_nom, t.tags 
            FROM verres v
            LEFT JOIN tags t ON v.id = t.verre_id
            LEFT JOIN fournisseurs f ON v.fournisseur_id = f.id
            WHERE v.id = ?
        """, (verre_id,))
        
        verre = cursor.fetchone()
        conn.close()
        
        if verre:
            verre_id, nom, variante, gravure, indice, url_source, fournisseur_nom, verre_tags_json = verre
            tags = []
            
            if verre_tags_json:
                try:
                    tags = json.loads(verre_tags_json)
                except json.JSONDecodeError:
                    tags = []
                    
            return {
                "id": verre_id,
                "nom": nom,
                "variante": variante,
                "gravure": gravure,
                "indice": indice,
                "url_source": url_source,
                "fournisseur": fournisseur_nom,
                "tags": tags
            }
        
        return None
    except sqlite3.Error as e:
        st.error(f"Erreur lors de la récupération des détails du verre: {e}")
        return None

# --- UI ---
st.title("Recherche de verres par symboles et tags")

# Initialiser les variables de session
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_tags' not in st.session_state:
    st.session_state.selected_tags = []
if 'matched_verres' not in st.session_state:
    st.session_state.matched_verres = []
if 'selected_verre_id' not in st.session_state:
    st.session_state.selected_verre_id = None
if 'selected_verre_details' not in st.session_state:
    st.session_state.selected_verre_details = None
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# Fonction pour traiter et afficher une image à partir d'une URL
def display_image_from_url(url, width=None, caption=None):
    if not url:
        return False
    
    # Vérifier si l'URL est valide
    if not url.startswith(('http://', 'https://')):
        # Si ce n'est pas une URL, on considère que c'est du texte
        return False
    
    try:
        # Essayer plusieurs méthodes d'affichage
        # 1. Affichage direct avec st.image
        st.image(url, width=width, caption=caption)
        
        # 2. Toujours afficher un lien direct pour l'image
        st.markdown(f"""
        ### Lien direct vers l'image
        Si l'image ne s'affiche pas correctement ci-dessus, vous pouvez [cliquer ici pour la voir]({url})
        """)
        
        # 3. Pour les images de si.france-optique.com, on ajoute une iframe
        if "france-optique.com" in url:
            st.markdown(f"""
            <div style="margin-top: 10px; margin-bottom: 10px;">
                <p>Aperçu via iframe :</p>
                <iframe src="{url}" width="{width or 300}" height="300" style="border: none;"></iframe>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Afficher l'image dans un composant HTML img
            st.markdown(f"""
            <div style="margin-top: 10px; margin-bottom: 10px;">
                <p>Aperçu via balise img :</p>
                <img src="{url}" width="{width or 300}" style="max-width: 100%;" />
            </div>
            """, unsafe_allow_html=True)
        
        return True
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image: {str(e)}")
        st.warning("Tentative d'affichage alternatif...")
        
        # Fallback pour les images qui ne peuvent pas être chargées directement
        st.markdown(f"""
        <div style="margin-top: 10px; margin-bottom: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
            <p>L'image n'a pas pu être chargée directement. Essayez ces alternatives :</p>
            <p><a href="{url}" target="_blank">1. Ouvrir l'image dans un nouvel onglet</a></p>
            <p>2. Aperçu direct (peut ne pas fonctionner) :</p>
            <img src="{url}" width="{width or 300}" style="max-width: 100%;" />
        </div>
        """, unsafe_allow_html=True)
        
        return False

# Fonction pour déterminer si une valeur est une URL d'image
def is_image_url(value):
    if not value:
        return False
    return isinstance(value, str) and value.startswith(('http://', 'https://'))

# Fonction pour afficher les détails d'un verre
def display_verre_details():
    if st.session_state.selected_verre_details:
        verre = st.session_state.selected_verre_details
        st.markdown("### Informations détaillées du verre")
        
        # Afficher l'image si la gravure est une URL
        if is_image_url(verre.get('gravure')):
            if display_image_from_url(verre['gravure'], width=250, caption="Image du verre"):
                st.success("Image chargée depuis le champ gravure")
        
        st.markdown(f"**ID :** {verre['id']}")
        st.markdown(f"**Fournisseur :** {verre['fournisseur'] or 'Non spécifié'}")
        st.markdown(f"**Nom :** {verre['nom'] or 'Non spécifié'}")
        st.markdown(f"**Variante :** {verre['variante'] or 'Non spécifiée'}")
        
        # Afficher la gravure en fonction de son type
        if verre.get('gravure'):
            if is_image_url(verre['gravure']):
                st.markdown(f"**Gravure :** URL d'image")
                st.markdown(f"**URL gravure :** [Lien]({verre['gravure']})")
            else:
                st.markdown(f"**Gravure :** {verre['gravure']}")
        else:
            st.markdown("**Gravure :** Non spécifiée")
            
        st.markdown(f"**Indice :** {verre['indice'] or 'Non spécifié'}")
        
        # Afficher les tags
        if verre['tags']:
            if isinstance(verre['tags'], list):
                tags_str = ", ".join([str(tag) for tag in verre['tags']])
                st.markdown(f"**Tags :** {tags_str}")
            else:
                st.markdown(f"**Tags :** {verre['tags']}")
        else:
            st.markdown("**Tags :** Aucun tag")

# Fonction pour mettre à jour les résultats de recherche sans recharger la page
def search_verres():
    with st.spinner("Recherche des verres correspondants..."):
        matched_verres = search_verres_by_tags(st.session_state.selected_tags)
        st.session_state.matched_verres = matched_verres
        st.session_state.search_performed = True

# Fonction pour sélectionner un verre et afficher ses détails
def select_verre(verre_id, verre_details):
    st.session_state.selected_verre_id = verre_id
    st.session_state.selected_verre_details = verre_details

# Créer un layout avec deux colonnes principales
col_left, col_right = st.columns([2, 1])

with col_left:
    # Afficher les tags sélectionnés
    st.subheader("Tags sélectionnés")
    if st.session_state.selected_tags:
        st.write(", ".join(st.session_state.selected_tags))
        st.info("La recherche trouvera les verres qui contiennent TOUS ces tags.")
    else:
        st.write("Aucun tag sélectionné.")
    
    # Option pour effacer tous les tags
    if st.button("🗑️ Effacer tous les tags"):
        st.session_state.selected_tags = []
        st.session_state.matched_verres = []
        st.session_state.search_performed = False
        st.experimental_rerun()
    
    # Saisie manuelle de tags
    st.subheader("Ajouter des tags manuellement")
    tag_input = st.text_input("Entrez des tags séparés par des virgules")
    if st.button("Ajouter ces tags"):
        if tag_input:
            new_tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
            st.session_state.selected_tags.extend(new_tags)
            # Supprimer les doublons
            st.session_state.selected_tags = list(set(st.session_state.selected_tags))
            st.experimental_rerun()
    
    # Bouton pour rechercher des verres avec les tags sélectionnés
    if st.button("🔍 Rechercher des verres avec TOUS ces tags") and st.session_state.selected_tags:
        search_verres()
    
    # Afficher les verres correspondants s'ils existent (conservés en session)
    if st.session_state.search_performed:
        results_container = st.container()
        with results_container:
            if len(st.session_state.matched_verres) > 0:
                st.subheader(f"Verres correspondants ({len(st.session_state.matched_verres)} trouvés):")
                
                # Créer les données pour le tableau
                data = []
                for verre in st.session_state.matched_verres:
                    # Ajouter les données du verre au tableau
                    data.append({
                        "Fournisseur": verre.get('fournisseur', ''),
                        "Variante": verre.get('variante', ''),
                        "Hauteur Min": verre.get('hauteur_min', ''),
                        "Hauteur Max": verre.get('hauteur_max', ''),
                        "Indice": verre.get('indice', ''),
                        "Gravure": verre.get('gravure', '')
                    })
                
                # Convertir en DataFrame pour l'affichage en tableau
                df = pd.DataFrame(data)
                
                # Afficher le tableau dans le format demandé
                display_df = df[["Fournisseur", "Variante", "Hauteur Min", "Hauteur Max", "Indice", "Gravure"]]
                st.table(display_df)
                
                # Afficher les images sous le tableau
                st.subheader("Images des verres trouvés")
                cols_img = st.columns(3)  # 3 images par ligne
                col_idx = 0
                
                for i, verre in enumerate(st.session_state.matched_verres):
                    if is_image_url(verre.get('gravure')):
                        with cols_img[col_idx % 3]:
                            if display_image_from_url(verre['gravure'], width=150, caption=f"Verre #{verre['id']} - {verre.get('fournisseur', '')} {verre.get('variante', '')}"):
                                st.button(f"Détails", key=f"img_details_{i}", on_click=select_verre, args=(verre['id'], verre))
                        col_idx += 1
            else:
                st.warning("Aucun verre ne correspond aux tags sélectionnés.")
                st.info("Vérifiez les informations de diagnostic dans la barre latérale.")
                    
    st.markdown("---")
    st.subheader("Dessiner pour rechercher des symboles")
    st.write("Dessinez une gravure ci-dessous pour la comparer aux références existantes.")
    
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
        search_button = st.button("🔍 Rechercher les symboles similaires")
    
    # Traiter le dessin si le bouton de recherche est cliqué
    if search_button and canvas_result.image_data is not None:
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")  # Convertir en niveaux de gris
        st.image(img, caption="Votre dessin", width=150)
        
        with st.spinner("Recherche des symboles similaires..."):
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
        st.subheader(f"Top {NUM_RESULTS} symboles similaires trouvés :")
        st.write("Cliquez sur un symbole pour l'ajouter à votre liste de tags")
        
        # Utiliser une disposition en grille plus adaptée pour éviter le chevauchement
        num_cols = 3  # Réduire à 3 colonnes au lieu de 5
        
        for i in range(0, min(len(st.session_state.results), NUM_RESULTS), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i + j
                if idx < len(st.session_state.results):
                    score, cls, ref_img, path = st.session_state.results[idx]
                    with cols[j]:
                        # Créer un conteneur pour l'image et le bouton
                        container = st.container()
                        container.image(ref_img, caption=f"{cls}", width=120)
                        container.write(f"Similarité : {score:.2f}")
                        if container.button(f"Ajouter", key=f"add_{i}_{j}"):
                            if cls not in st.session_state.selected_tags:
                                st.session_state.selected_tags.append(cls)
                                st.experimental_rerun()

# Dans la colonne de droite, afficher le panneau permanent pour les détails du verre
with col_right:
    details_container = st.container()
    with details_container:
        st.markdown("## Détails du verre")
        
        # Si un verre est sélectionné, afficher les détails
        if st.session_state.selected_verre_id is not None:
            # Vérifier si les détails du verre sont déjà chargés, sinon les charger
            if st.session_state.selected_verre_details is None:
                verre_details = get_verre_details(st.session_state.selected_verre_id)
                if verre_details:
                    st.session_state.selected_verre_details = verre_details
            
            # Afficher les détails du verre
            display_verre_details()
        else:
            st.info("Sélectionnez un verre pour voir ses détails.")
            
        # Bouton pour effacer la sélection
        if st.session_state.selected_verre_id is not None:
            if st.button("Fermer les détails"):
                st.session_state.selected_verre_id = None
                st.session_state.selected_verre_details = None
                st.experimental_rerun()