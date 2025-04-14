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

# Ajouter le chemin parent pour les imports depuis E3_MettreDispositionIA
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from E3_MettreDispositionIA.app.model import load_model

# Constantes
CANVAS_WIDTH = 300
CANVAS_HEIGHT = 300
OUTPUT_SIZE = 64
BRUSH_SIZE = 3
SYMBOL_TYPE = "symbole"  # Dossier unique pour tous les dessins
IMAGE_SIZE = 64  # Taille des images attendue par le modèle

# Charger le modèle
@st.cache_resource
def load_model_only():
    try:
        # Obtenir le chemin absolu du répertoire principal du projet
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # Construire le chemin absolu vers le modèle
        model_path = os.path.join(project_dir, "E3_MettreDispositionIA", "model", "best_siamese_model.pt")
        
        st.write(f"Chemin du modèle: {model_path}")
        
        # Vérifier si le fichier du modèle existe
        if not os.path.isfile(model_path):
            model_path = os.path.join(project_dir, "E3_MettreDispositionIA", "model", "siamese_model.pt")
            st.write(f"Modèle principal non trouvé, tentative avec modèle alternatif: {model_path}")
            
            if not os.path.isfile(model_path):
                st.error("Aucun modèle trouvé!")
                return None, "cpu"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(model_path, device=device, embedding_dim=256)
        return model, device
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, "cpu"

# Prétraiter l'image pour le modèle
def preprocess_image(img):
    # Convertir en niveaux de gris si nécessaire
    if img.mode != "L":
        img = img.convert("L")
    
    # Amélioration des contrastes
    img_array = np.array(img)
    
    # Appliquer un filtre gaussien pour réduire le bruit
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # Améliorer le contraste avec une égalisation d'histogramme
    img_array = cv2.equalizeHist(img_array)
    
    # Redimensionner à la taille attendue par le modèle
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Normaliser les valeurs
    img_array = img_array / 255.0
    
    # Préparer pour PyTorch (B, C, H, W)
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
    
    return img_tensor

# Charger toutes les images de référence et générer leurs embeddings
@st.cache_data
def load_reference_images(_model, _device):
    st.info("Chargement des images de référence...")
    
    reference_images = []
    
    # Obtenir le chemin absolu du répertoire principal du projet
    # Remonter d'un niveau par rapport au répertoire poc_smartphone
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Afficher le chemin du projet pour débogage
    st.write(f"Chemin du projet: {project_dir}")
    
    # Chemins des dossiers contenant les images
    processed_dir = os.path.join(project_dir, "E3_MettreDispositionIA", "data", "processed")
    augmented_dir = os.path.join(project_dir, "E3_MettreDispositionIA", "data", "augmented_gravures")
    raw_dir = os.path.join(project_dir, "E3_MettreDispositionIA", "data", "raw_gravures")
    
    # Liste des dossiers à vérifier
    dirs_to_check = [
        processed_dir,
        augmented_dir,
        raw_dir
    ]
    
    # Fonction pour traiter un dossier
    def process_directory(base_dir):
        if not os.path.exists(base_dir):
            st.warning(f"Le dossier {base_dir} n'existe pas.")
            return
            
        st.success(f"Dossier trouvé: {base_dir}")
            
        # Trouver toutes les classes (sous-dossiers)
        classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        if not classes:
            st.warning(f"Aucune classe trouvée dans {base_dir}")
            return
            
        st.write(f"Classes trouvées: {', '.join(classes)}")
        
        for class_name in classes:
            class_dir = os.path.join(base_dir, class_name)
            
            # Trouver toutes les images de cette classe (limiter à 10 par classe pour éviter de surcharger)
            image_files = glob.glob(os.path.join(class_dir, "*.png")) + \
                         glob.glob(os.path.join(class_dir, "*.jpg")) + \
                         glob.glob(os.path.join(class_dir, "*.jpeg"))
            
            if not image_files:
                st.warning(f"Aucune image trouvée dans {class_dir}")
                continue
                
            st.write(f"Classe {class_name}: {len(image_files)} images trouvées")
            
            # Prendre un échantillon aléatoire si trop d'images (max 10 par classe)
            if len(image_files) > 10:
                import random
                random.shuffle(image_files)
                image_files = image_files[:10]
            
            for img_path in image_files:
                try:
                    # Charger et prétraiter l'image
                    img = Image.open(img_path).convert('L')
                    img_tensor = preprocess_image(img)
                    img_tensor = img_tensor.to(_device)
                    
                    # Générer l'embedding
                    with torch.no_grad():
                        embedding = _model.forward_one(img_tensor).cpu().numpy()[0]
                    
                    # Ajouter à notre liste
                    reference_images.append({
                        'path': img_path,
                        'class': class_name,
                        'embedding': embedding
                    })
                        
                except Exception as e:
                    st.warning(f"Erreur lors du traitement de {img_path}: {e}")
    
    # Traiter chaque dossier
    for dir_path in dirs_to_check:
        process_directory(dir_path)
    
    st.success(f"{len(reference_images)} images de référence chargées")
    return reference_images

# Générer plusieurs variantes de l'image dessinée pour la reconnaissance
def generate_variants(img):
    """Génère plusieurs variantes de l'image pour améliorer la reconnaissance"""
    variants = []
    img_array = np.array(img)
    
    # Image originale
    variants.append(img_array)
    
    # Rotation à 5 degrés
    rows, cols = img_array.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
    rotated = cv2.warpAffine(img_array, M, (cols, rows))
    variants.append(rotated)
    
    # Rotation à -5 degrés
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -5, 1)
    rotated = cv2.warpAffine(img_array, M, (cols, rows))
    variants.append(rotated)
    
    # Léger décalage
    M = np.float32([[1, 0, 3], [0, 1, 3]])
    shifted = cv2.warpAffine(img_array, M, (cols, rows))
    variants.append(shifted)
    
    # Légère dilatation
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(img_array, kernel, iterations=1)
    variants.append(dilated)
    
    # Léger érosion
    eroded = cv2.erode(img_array, kernel, iterations=1)
    variants.append(eroded)
    
    return variants

# Fonction pour trouver les images les plus similaires avec vote
def find_similar_images(query_embedding, reference_images, top_n=5):
    if not reference_images:
        return []
    
    # Tableau pour stocker les votes
    class_votes = {}
    
    # Calculer la similarité avec tous les embeddings et le score ajusté
    similarities = []
    
    for ref_img in reference_images:
        # Distance euclidienne
        distance = np.linalg.norm(query_embedding - ref_img['embedding'])
        
        # Convertir en similarité
        # Une fonction exponentielle décroissante plus agressive
        similarity = np.exp(-distance * 1.5)  # Facteur 1.5 pour accentuer les différences
        
        # Augmenter les similarités élevées et diminuer les faibles
        boosted_similarity = similarity**2 if similarity > 0.5 else similarity**3
        
        # Ajouter à la liste des similarités
        similarities.append((ref_img, boosted_similarity))
        
        # Comptabiliser les votes pour chaque classe
        class_name = ref_img['class']
        if class_name not in class_votes:
            class_votes[class_name] = 0
        class_votes[class_name] += boosted_similarity
    
    # Trier par similarité décroissante
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Trier les classes par votes
    sorted_classes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    top_classes = [cls for cls, _ in sorted_classes[:top_n]]
    
    # Filtrer pour ne garder que les meilleures images des classes les plus votées
    filtered_results = []
    
    for class_name in top_classes:
        # Trouver la meilleure image de cette classe
        best_match = None
        best_score = -1
        
        for ref_img, similarity in similarities:
            if ref_img['class'] == class_name and similarity > best_score:
                best_match = (ref_img, similarity)
                best_score = similarity
        
        if best_match:
            filtered_results.append(best_match)
    
    return filtered_results

# Fonction pour créer les répertoires nécessaires
def ensure_directories():
    base_dir = Path("drawings")
    base_dir.mkdir(exist_ok=True)
    
    # Un seul dossier pour tous les dessins
    symbol_dir = base_dir / SYMBOL_TYPE
    symbol_dir.mkdir(exist_ok=True)
    
    return base_dir

# Fonction pour enregistrer l'image
def save_image(image_data):
    # Créer les répertoires
    drawings_dir = ensure_directories()
    
    # Générer un nom de fichier unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SYMBOL_TYPE}_{timestamp}.png"
    filepath = drawings_dir / SYMBOL_TYPE / filename
    
    # Convertir les données canvas en image PIL
    img_array = image_data.image_data
    
    # Convertir l'array numpy en image PIL
    img = Image.fromarray(img_array.astype('uint8'))
    
    # Convertir en noir et blanc si nécessaire
    if img.mode != "L":
        img = img.convert("L")
    
    # Redimensionner à 64x64 pixels
    img_resized = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    
    # Sauvegarder l'image
    img_resized.save(filepath)
    
    return filepath, img_resized

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Reconnaissance de Gravures",
    page_icon="✏️",
    layout="centered"
)

def main():
    st.title("Reconnaissance de gravures optiques")
    
    # Charger le modèle
    model, device = load_model_only()
    
    # Charger les images de référence
    if model is not None:
        reference_images = load_reference_images(model, device)
    else:
        reference_images = []
        st.error("Impossible de charger les images de référence sans modèle.")
    
    # Initialiser l'état de la session pour le canvas
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    # Initialiser l'état pour les résultats de reconnaissance
    if 'recognition_results' not in st.session_state:
        st.session_state.recognition_results = None
    
    # Zone de dessin
    st.subheader("Zone de dessin")
    st.caption("Dessinez votre symbole ici")
    
    # Canvas dessinable avec taille de pinceau fixe à 3
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=BRUSH_SIZE,
        stroke_color="black",
        background_color="white",
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Enregistrer et Reconnaître", type="primary"):
            if canvas_result.image_data is not None:
                # Vérifier s'il y a un dessin (pas juste un canvas blanc)
                if not np.all(canvas_result.image_data == 255):
                    filepath, img_resized = save_image(canvas_result)
                    st.success(f"Gravure enregistrée: {filepath.name}")
                    
                    # Reconnaissance de la gravure
                    if model is not None and reference_images:
                        with st.spinner("Reconnaissance en cours..."):
                            # Générer plusieurs variantes de l'image
                            variants = generate_variants(img_resized)
                            
                            # Pour chaque variante, calculer un embedding et faire la reconnaissance
                            all_results = []
                            
                            for variant in variants:
                                variant_img = Image.fromarray(variant.astype('uint8'))
                                img_tensor = preprocess_image(variant_img)
                                img_tensor = img_tensor.to(device)
                                
                                with torch.no_grad():
                                    embedding = model.forward_one(img_tensor).cpu().numpy()[0]
                                
                                # Calculer les similarités pour cette variante
                                variant_results = find_similar_images(embedding, reference_images, top_n=5)
                                all_results.extend(variant_results)
                            
                            # Fusionner les résultats et prendre les 5 meilleures classes
                            # Créer un dictionnaire pour stocker la meilleure similarité par classe
                            class_best = {}
                            for ref_img, similarity in all_results:
                                class_name = ref_img['class']
                                if class_name not in class_best or similarity > class_best[class_name][1]:
                                    class_best[class_name] = (ref_img, similarity)
                            
                            # Convertir en liste et trier
                            combined_results = list(class_best.values())
                            combined_results.sort(key=lambda x: x[1], reverse=True)
                            
                            # Prendre les 5 premières classes différentes
                            results = combined_results[:5]
                            st.session_state.recognition_results = results
                    else:
                        st.warning("Le modèle n'a pas été chargé correctement ou aucune image de référence n'est disponible.")
                    
                    # Effacer après sauvegarde
                    st.session_state.canvas_key += 1
                    st.rerun()
                else:
                    st.warning("Le canvas est vide, rien à enregistrer.")
    
    with col2:
        if st.button("Effacer"):
            # Générer une nouvelle clé pour forcer la réinitialisation du canvas
            st.session_state.canvas_key += 1
            st.session_state.recognition_results = None
            st.rerun()
    
    # Afficher les résultats de reconnaissance
    if st.session_state.recognition_results is not None:
        st.subheader("Résultats de la reconnaissance")
        
        results = st.session_state.recognition_results
        
        if results:
            # Utiliser deux lignes pour afficher les 5 meilleures correspondances
            # Première ligne: 3 colonnes
            # Deuxième ligne: 2 colonnes
            
            if len(results) > 0:
                st.write("Top 5 correspondances (classes différentes):")
                
                # Première ligne avec 3 colonnes
                row1_cols = st.columns(3)
                
                # Traiter les 3 premiers résultats
                for i in range(min(3, len(results))):
                    with row1_cols[i]:
                        ref_img, similarity = results[i]
                        class_name = ref_img['class']
                        img_path = ref_img['path']
                        
                        st.metric(f"Match #{i+1}", class_name, f"{similarity:.2%}")
                        
                        # Essayer d'afficher l'image
                        try:
                            if os.path.isfile(img_path):
                                img = Image.open(img_path)
                                st.image(img, caption=f"Similarité: {similarity:.2%}")
                            else:
                                st.write(f"Image introuvable")
                        except Exception as e:
                            st.write(f"Erreur d'affichage")
                
                # Deuxième ligne avec 2 colonnes si nécessaire
                if len(results) > 3:
                    row2_cols = st.columns(2)
                    
                    # Traiter les 2 résultats restants
                    for i in range(3, min(5, len(results))):
                        col_idx = i - 3  # 0 ou 1
                        with row2_cols[col_idx]:
                            ref_img, similarity = results[i]
                            class_name = ref_img['class']
                            img_path = ref_img['path']
                            
                            st.metric(f"Match #{i+1}", class_name, f"{similarity:.2%}")
                            
                            # Essayer d'afficher l'image
                            try:
                                if os.path.isfile(img_path):
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"Similarité: {similarity:.2%}")
                                else:
                                    st.write(f"Image introuvable")
                            except Exception as e:
                                st.write(f"Erreur d'affichage")
        else:
            st.info("Aucune correspondance trouvée.")
    
    # Instructions minimales
    with st.expander("Instructions"):
        st.markdown("""
        1. Dessinez un symbole dans la zone de dessin
        2. Cliquez sur 'Enregistrer et Reconnaître' pour identifier le symbole
        3. Cliquez sur 'Effacer' pour recommencer
        """)

if __name__ == "__main__":
    main() 