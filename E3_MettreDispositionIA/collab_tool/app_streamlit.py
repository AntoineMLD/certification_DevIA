import streamlit as st
import os
from PIL import Image
from datetime import datetime
import hashlib
from supabase import create_client
import requests
from io import BytesIO

# Configuration
# Chemin vers le dossier d'images local
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
GITHUB_RAW_URL = "https://raw.githubusercontent.com/AntoineMLD/certification_DevIA/main/E3_MettreDispositionIA/images/"

# Configuration Supabase
SUPABASE_URL = st.secrets["supabase_url"]
SUPABASE_KEY = st.secrets["supabase_key"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuration de l'application
MAX_DESCRIPTION_LENGTH = 500  # Limite de caractères par description
MIN_DESCRIPTION_LENGTH = 3   # Minimum de caractères requis

def init_db():
    """Vérifie que la table existe dans Supabase."""
    # Note: Avec Supabase, la table doit être créée manuellement dans l'interface
    pass

def compute_hash(verre_id, description, timestamp):
    """Calcule un hash pour vérifier l'intégrité des données."""
    data = f"{verre_id}{description}{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()

def get_descriptions():
    """Récupère toutes les descriptions depuis Supabase."""
    try:
        # Récupère toutes les descriptions triées par verre_id et date
        response = supabase.table('descriptions').select('*').order('verre_id').order('created_at').execute()
        
        # Organise les descriptions par verre
        descriptions = {}
        for record in response.data:
            verre_id = str(record['verre_id'])
            description = record['description']
            timestamp = record['created_at']
            stored_hash = record['hash']
            
            # Vérifie l'intégrité des données
            computed_hash = compute_hash(verre_id, description, timestamp)
            if computed_hash != stored_hash:
                st.error(f"Attention: Données potentiellement corrompues pour le verre {verre_id}")
                continue
                
            if verre_id not in descriptions:
                descriptions[verre_id] = []
            descriptions[verre_id].append(description)
        
        return descriptions
    except Exception as e:
        st.error(f"Erreur lors de la récupération des descriptions: {e}")
        return {}

def get_image_from_github(filename):
    """Récupère une image depuis le dossier local."""
    try:
        image_path = os.path.join(IMAGES_DIR, filename)
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image: {e}")
        return None

def validate_description(description):
    """Valide une description avant de la sauvegarder."""
    if len(description) < MIN_DESCRIPTION_LENGTH:
        st.error(f"La description doit faire au moins {MIN_DESCRIPTION_LENGTH} caractères.")
        return False
    if len(description) > MAX_DESCRIPTION_LENGTH:
        st.error(f"La description ne doit pas dépasser {MAX_DESCRIPTION_LENGTH} caractères.")
        return False
    # Vérification basique anti-spam/injection
    if '<script>' in description.lower() or 'javascript:' in description.lower():
        return False
    return True

def save_description(verre_id, description):
    """Sauvegarde une description dans Supabase avec validation."""
    try:
        if not validate_description(description):
            return False
            
        # Ajoute un horodatage et un hash pour la sécurité
        timestamp = datetime.now().isoformat()
        hash_value = compute_hash(verre_id, description, timestamp)
        
        # Insère les données dans Supabase
        data = {
            'verre_id': verre_id,
            'description': description,
            'created_at': timestamp,
            'hash': hash_value
        }
        
        response = supabase.table('descriptions').insert(data).execute()
        return True if response.data else False
    
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde: {e}")
        return False

def get_image_list():
    """Récupère la liste des images et les trie par nombre de descriptions croissant."""
    images = []
    descriptions = get_descriptions()
    
    # Parcourir le dossier d'images local
    try:
        for filename in os.listdir(IMAGES_DIR):
            if filename.endswith('.png'):
                try:
                    image_id = int(filename.split('_')[0])
                    # Compte le nombre de descriptions pour cette image
                    description_count = len(descriptions.get(str(image_id), []))
                    
                    images.append({
                        'id': image_id,
                        'filename': filename,
                        'description_count': description_count
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier d'images: {e}")
        return []
    
    # Trie les images par nombre de descriptions croissant, puis par ID
    return sorted(images, key=lambda x: (x['description_count'], x['id']))

def resize_image(image, scale_percent=80):
    """Redimensionne l'image selon un pourcentage."""
    width = int(image.size[0] * scale_percent / 100)
    height = int(image.size[1] * scale_percent / 100)
    return image.resize((width, height), Image.Resampling.LANCZOS)

def main():
    # Initialiser la base de données au démarrage
    init_db()
    
    st.set_page_config(
        page_title="Description des Verres",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # CSS personnalisé pour réduire les espaces
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.5rem;
                padding-bottom: 0rem;
            }
            .element-container {
                margin-bottom: 0.2rem;
            }
            .stButton button {
                width: 100%;
                padding: 0.2rem !important;
            }
            .stMarkdown p {
                margin-bottom: 0.2rem;
                font-size: 0.9rem;
            }
            div[data-testid="column"] {
                padding: 0rem;
            }
            .st-emotion-cache-1v0mbdj.e115fcil1 {
                margin-top: -1rem;
            }
            .st-emotion-cache-1kyxreq.e115fcil2 {
                margin-top: -2rem;
            }
            h1 {
                margin-bottom: 0.5rem !important;
                font-size: 1.5rem !important;
            }
            .stTextArea textarea {
                height: 80px !important;
            }
            .stAlert {
                padding: 0.3rem;
                margin: 0.2rem 0;
            }
            div[data-testid="stExpander"] {
                margin-bottom: 0.2rem;
            }
            /* Contrôle de la taille de l'image */
            img {
                max-width: 600px !important;
                margin: 0 auto;
                display: block;
            }
            /* Centrer la légende de l'image */
            .stImage caption {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Container principal avec largeur maximale
    with st.container():
        col_main = st.columns([6, 4])[0]  # Utilise 60% de la largeur
        
        with col_main:
            st.title("Description des Verres")
            
            # Récupérer la liste des images
            images = get_image_list()
            
            if not images:
                st.warning("Aucune image trouvée")
                return
            
            # Gérer l'index de l'image courante
            if 'current_index' not in st.session_state:
                st.session_state.current_index = 0
            
            # Afficher l'image courante
            current_image = images[st.session_state.current_index]
            image_path = GITHUB_RAW_URL + current_image['filename']
            
            # Navigation et informations
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("←", disabled=st.session_state.current_index == 0):
                    st.session_state.current_index -= 1
                    st.rerun()
            
            with col2:
                description_count = current_image['description_count']
                color = "red" if description_count == 0 else "orange" if description_count < 3 else "green"
                st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <p style='margin:0;font-size:0.8rem;'>Image {st.session_state.current_index + 1}/{len(images)}</p>
                        <p style='color: {color}; margin:0;font-size:0.8rem;'>{description_count} description{'s' if description_count != 1 else ''}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col3:
                if st.button("→", disabled=st.session_state.current_index == len(images) - 1):
                    st.session_state.current_index += 1
                    st.rerun()
            
            # Afficher l'image
            image = get_image_from_github(current_image['filename'])
            if image:
                st.image(image, caption=f"Verre {current_image['id']}")
            
            # Afficher les descriptions existantes
            descriptions = get_descriptions()
            existing_descriptions = descriptions.get(str(current_image['id']), [])
            if not isinstance(existing_descriptions, list):
                existing_descriptions = [existing_descriptions]
            
            if existing_descriptions:
                with st.expander("Descriptions existantes", expanded=True):
                    for desc in existing_descriptions:
                        st.info(desc)
            
            # Formulaire pour ajouter une description
            with st.form("description_form", clear_on_submit=True):
                description = st.text_area(
                    "Description",
                    height=80,
                    placeholder="Décrivez ce que vous voyez sur le verre...",
                    label_visibility="collapsed"
                )
                submit = st.form_submit_button(
                    "Enregistrer et Suivant →",
                    use_container_width=True
                )
                
                if submit and description:
                    if save_description(current_image['id'], description):
                        st.success("Enregistré!")
                        if st.session_state.current_index < len(images) - 1:
                            st.session_state.current_index += 1
                        st.rerun()

if __name__ == '__main__':
    main() 