import streamlit as st
import os
import json
from PIL import Image

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_descriptions():
    """Charge les descriptions existantes depuis le fichier JSON."""
    descriptions_file = os.path.join(UPLOAD_FOLDER, 'descriptions.json')
    if os.path.exists(descriptions_file):
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_image_list():
    """Récupère la liste des images et les trie par nombre de descriptions croissant."""
    images = []
    descriptions = get_descriptions()
    
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.endswith('.png'):
            try:
                image_id = int(filename.split('_')[0])
                # Compte le nombre de descriptions pour cette image
                description_count = 0
                if str(image_id) in descriptions:
                    if isinstance(descriptions[str(image_id)], list):
                        description_count = len(descriptions[str(image_id)])
                    else:
                        description_count = 1
                
                images.append({
                    'id': image_id,
                    'filename': filename,
                    'description_count': description_count
                })
            except (ValueError, IndexError):
                continue
    
    # Trie les images par nombre de descriptions croissant, puis par ID
    return sorted(images, key=lambda x: (x['description_count'], x['id']))

def save_description(verre_id, description):
    """Sauvegarde une description dans le fichier JSON."""
    descriptions_file = os.path.join(UPLOAD_FOLDER, 'descriptions.json')
    descriptions = get_descriptions()
    
    # Convertir l'ID en string pour le stockage JSON
    verre_id = str(verre_id)
    
    # Si c'est la première description pour cette image, créer une liste
    if verre_id not in descriptions:
        descriptions[verre_id] = []
    elif not isinstance(descriptions[verre_id], list):
        # Convertir une ancienne description unique en liste
        descriptions[verre_id] = [descriptions[verre_id]]
    
    # Ajouter la nouvelle description à la liste
    descriptions[verre_id].append(description)
    
    # Sauvegarder les descriptions
    with open(descriptions_file, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)
    
    return True

def resize_image(image, scale_percent=80):
    """Redimensionne l'image selon un pourcentage."""
    width = int(image.size[0] * scale_percent / 100)
    height = int(image.size[1] * scale_percent / 100)
    return image.resize((width, height), Image.Resampling.LANCZOS)

def main():
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
            image_path = os.path.join(IMAGES_FOLDER, current_image['filename'])
            
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
            image = Image.open(image_path)
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