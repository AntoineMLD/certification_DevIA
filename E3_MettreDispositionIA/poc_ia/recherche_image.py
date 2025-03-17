import os
import json
import numpy as np
from supabase import create_client
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import azure.cognitiveservices.speech as speechsdk

# Configuration Supabase
try:
    SUPABASE_URL = st.secrets["supabase_url"]
    SUPABASE_KEY = st.secrets["supabase_key"]
except Exception as e:
    st.error("Erreur: Veuillez configurer les secrets Streamlit avec les informations de connexion à Supabase.")
    SUPABASE_URL = "https://rgmumgolnowpilenhdvb.supabase.co"
    SUPABASE_KEY = "votre_clé_supabase"  # Sera remplacé par les secrets

# Configuration Azure Speech Services
try:
    AZURE_SPEECH_KEY = st.secrets["azure_speech_key"]
    AZURE_SPEECH_REGION = st.secrets["azure_speech_region"]
    AZURE_CONFIGURED = True
except Exception as e:
    AZURE_CONFIGURED = False
    st.warning("Azure Speech Services non configuré. La reconnaissance vocale ne sera pas disponible.")

# Chemin vers le dossier d'images
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")

# Initialisation du modèle de traitement du langage
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_supabase_client():
    """Crée et retourne un client Supabase."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_descriptions_from_supabase():
    """Récupère toutes les descriptions depuis Supabase."""
    supabase = get_supabase_client()
    response = supabase.table('descriptions').select('*').execute()
    
    # Organiser les descriptions par verre_id
    descriptions_by_id = {}
    for record in response.data:
        verre_id = str(record['verre_id'])
        description = record['description']
        
        if verre_id not in descriptions_by_id:
            descriptions_by_id[verre_id] = []
        
        descriptions_by_id[verre_id].append(description)
    
    return descriptions_by_id

def get_image_path(verre_id):
    """Retourne le chemin de l'image correspondant à l'ID."""
    for filename in os.listdir(IMAGES_DIR):
        if filename.startswith(f"{verre_id}_"):
            return os.path.join(IMAGES_DIR, filename)
    return None

def encode_descriptions(descriptions_by_id, model):
    """Encode toutes les descriptions en vecteurs."""
    all_descriptions = []
    id_mapping = []
    
    for verre_id, descriptions in descriptions_by_id.items():
        for desc in descriptions:
            all_descriptions.append(desc)
            id_mapping.append(verre_id)
    
    # Encoder toutes les descriptions
    embeddings = model.encode(all_descriptions)
    
    return embeddings, id_mapping, all_descriptions

def find_best_match(query, embeddings, id_mapping, descriptions, model, top_n=3):
    """Trouve les meilleures correspondances pour une requête."""
    # Encoder la requête
    query_embedding = model.encode([query])[0]
    
    # Calculer la similarité avec toutes les descriptions
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Trouver les indices des meilleures correspondances
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        results.append({
            'verre_id': id_mapping[idx],
            'description': descriptions[idx],
            'similarity': similarities[idx]
        })
    
    return results

def recognize_speech_azure():
    """Utilise Azure Speech Services pour reconnaître la parole."""
    if not AZURE_CONFIGURED:
        st.error("Azure Speech Services n'est pas configuré.")
        return None
    
    # Créer un placeholder pour afficher le statut
    status_placeholder = st.empty()
    status_placeholder.info("Écoutez... Parlez maintenant.")
    
    # Configurer le service de reconnaissance vocale
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "fr-FR"  # Langue française
    
    # Créer un reconnaisseur de parole
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # Variable pour stocker le résultat
    result_text = None
    
    # Callback pour la reconnaissance terminée
    def recognized_cb(evt):
        nonlocal result_text
        result_text = evt.result.text
        status_placeholder.success(f"Reconnu: {result_text}")
    
    # Configurer les callbacks
    speech_recognizer.recognized.connect(recognized_cb)
    
    # Démarrer la reconnaissance
    speech_recognizer.start_recognizing()
    
    # Attendre jusqu'à 10 secondes pour la reconnaissance
    import time
    start_time = time.time()
    while result_text is None and time.time() - start_time < 10:
        time.sleep(0.1)
    
    # Arrêter la reconnaissance
    speech_recognizer.stop_recognizing()
    
    if result_text is None:
        status_placeholder.error("Aucune parole détectée. Veuillez réessayer.")
    
    return result_text

def main():
    st.title("Recherche d'image par description vocale")
    
    # Charger le modèle
    model = load_model()
    
    # Récupérer les descriptions
    descriptions_by_id = get_descriptions_from_supabase()
    
    if not descriptions_by_id:
        st.warning("Aucune description trouvée dans la base de données.")
        return
    
    # Encoder les descriptions
    embeddings, id_mapping, all_descriptions = encode_descriptions(descriptions_by_id, model)
    
    # Interface utilisateur
    st.write("Décrivez ce que vous voyez sur le verre :")
    
    # Option 1: Entrée texte (pour tester)
    query = st.text_area("Description", height=100)
    
    # Option 2: Entrée vocale avec Azure
    st.write("Ou utilisez la reconnaissance vocale :")
    if st.button("🎤 Parler"):
        speech_text = recognize_speech_azure()
        if speech_text:
            query = speech_text
            st.session_state.query = query
    
    # Utiliser la valeur de la session si disponible
    if 'query' in st.session_state:
        query = st.session_state.query
    
    if st.button("Rechercher") and query:
        with st.spinner("Recherche en cours..."):
            # Trouver les meilleures correspondances
            results = find_best_match(query, embeddings, id_mapping, all_descriptions, model)
            
            # Afficher les résultats
            st.subheader("Résultats de la recherche")
            
            for i, result in enumerate(results):
                verre_id = result['verre_id']
                similarity = result['similarity'] * 100
                
                st.write(f"**Match #{i+1}** (Confiance: {similarity:.1f}%)")
                
                # Afficher l'image
                image_path = get_image_path(verre_id)
                if image_path:
                    image = Image.open(image_path)
                    st.image(image, caption=f"Verre {verre_id}", width=300)
                else:
                    st.error(f"Image pour le verre {verre_id} non trouvée.")
                
                # Afficher la description correspondante
                st.info(f"Description: {result['description']}")
                st.markdown("---")

if __name__ == "__main__":
    main() 