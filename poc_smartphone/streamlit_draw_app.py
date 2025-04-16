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
IMAGE_SIZE = 64  # Taille des images attendue par le modèle

# Charger le modèle
@st.cache_resource
def load_model_only():
    """Charge le modèle Siamese pour la reconnaissance des gravures"""
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

def preprocess_image(image):
    """Prétraite l'image pour le modèle"""
    # Convertir en niveaux de gris si nécessaire
    if image.mode != "L":
        image = image.convert("L")
    
    # Amélioration des contrastes
    img_array = np.array(image)
    
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

@st.cache_data
def load_reference_images(_model, _device):
    """Charge les images de référence et génère leurs embeddings"""
    st.info("Chargement des images de référence...")
    
    reference_images = []
    
    # Obtenir le chemin absolu du répertoire principal du projet
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Afficher le chemin du projet pour débogage
    st.write(f"Chemin du projet: {project_dir}")
    
    # Chemins des dossiers contenant les images
    processed_dir = os.path.join(project_dir, "E3_MettreDispositionIA", "data", "processed")
    augmented_dir = os.path.join(project_dir, "E3_MettreDispositionIA", "data", "augmented_gravures")
    raw_dir = os.path.join(project_dir, "E3_MettreDispositionIA", "data", "raw_gravures")
    
    # Liste des dossiers à vérifier
    dirs_to_check = [processed_dir, augmented_dir, raw_dir]
    
    def process_directory(base_dir):
        """Traite un dossier pour extraire les images et leurs embeddings"""
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
            
            # Trouver toutes les images de cette classe
            image_files = glob.glob(os.path.join(class_dir, "*.png")) + \
                         glob.glob(os.path.join(class_dir, "*.jpg")) + \
                         glob.glob(os.path.join(class_dir, "*.jpeg"))
            
            if not image_files:
                st.warning(f"Aucune image trouvée dans {class_dir}")
                continue
                
            st.write(f"Classe {class_name}: {len(image_files)} images trouvées")
            
            # Limiter à 10 images par classe pour éviter de surcharger
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

def generate_variants(image):
    """Génère plusieurs variantes de l'image pour améliorer la reconnaissance"""
    variants = []
    img_array = np.array(image)
    
    # Image originale
    variants.append(img_array)
    
    # Rotation à 5 degrés
    rows, cols = img_array.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
    rotated = cv2.warpAffine(img_array, rotation_matrix, (cols, rows))
    variants.append(rotated)
    
    # Rotation à -5 degrés
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), -5, 1)
    rotated = cv2.warpAffine(img_array, rotation_matrix, (cols, rows))
    variants.append(rotated)
    
    # Léger décalage
    translation_matrix = np.float32([[1, 0, 3], [0, 1, 3]])
    shifted = cv2.warpAffine(img_array, translation_matrix, (cols, rows))
    variants.append(shifted)
    
    # Légère dilatation
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(img_array, kernel, iterations=1)
    variants.append(dilated)
    
    # Légère érosion
    eroded = cv2.erode(img_array, kernel, iterations=1)
    variants.append(eroded)
    
    return variants

def find_similar_images(query_embedding, reference_images, top_n=5):
    """Trouve les images les plus similaires en utilisant un système de vote"""
    if not reference_images:
        return []
    
    # Tableau pour stocker les votes
    class_votes = {}
    
    # Calculer la similarité avec tous les embeddings
    similarities = []
    
    for ref_img in reference_images:
        # Distance euclidienne
        distance = np.linalg.norm(query_embedding - ref_img['embedding'])
        
        # Convertir en similarité (fonction exponentielle décroissante)
        similarity = np.exp(-distance * 1.5)  # Facteur 1.5 pour accentuer les différences
        
        # Améliorer les similarités
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
        # Prendre les meilleures images de cette classe
        class_images = [(img, score) for img, score in similarities if img['class'] == class_name]
        filtered_results.extend(class_images[:3])  # Max 3 images par classe
    
    # Retrier par similarité
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    
    # Limiter au nombre demandé
    return filtered_results[:top_n]

def save_image(image_data, symbol_type="symbole"):
    """Sauvegarde l'image dessinée dans le dossier spécifié
    
    Args:
        image_data: Données de l'image à sauvegarder
        symbol_type: Type de symbole/dossier où sauvegarder l'image
    
    Returns:
        Le chemin du fichier sauvegardé ou None en cas d'erreur
    """
    # Créer le dossier drawings s'il n'existe pas
    drawings_dir = os.path.join(os.path.dirname(__file__), "drawings")
    if not os.path.exists(drawings_dir):
        os.makedirs(drawings_dir)
    
    # Créer le sous-dossier pour le type de symbole
    symbol_dir = os.path.join(drawings_dir, symbol_type)
    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)
    
    # Générer un nom de fichier unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol_type}_{timestamp}.png"
    filepath = os.path.join(symbol_dir, filename)
    
    try:
        # Convertir l'image en niveaux de gris et la sauvegarder
        pil_image = Image.fromarray(image_data).convert("L")
        pil_image.save(filepath)
        st.success(f"Image sauvegardée: {filepath}")
        return filepath
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde de l'image: {e}")
        return None

def main():
    """Fonction principale de l'application"""
    # Titre de l'application
    st.title("Reconnaissance de Gravures Optiques")
    st.write("Dessinez une gravure optique pour la reconnaître")
    
    # Charger le modèle
    model, device = load_model_only()
    
    if model is None:
        st.error("Impossible de charger le modèle. Vérifiez les logs pour plus d'informations.")
        return
    
    # Chargement des images de référence
    reference_images = load_reference_images(_model=model, _device=device)
    
    if not reference_images:
        st.warning("Aucune image de référence trouvée. La reconnaissance ne sera pas possible.")
    
    # Interface de dessin
    st.subheader("Zone de dessin")
    st.write("Utilisez la souris pour dessiner une gravure")
    
    # Paramètres du canvas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        drawing_mode = st.selectbox(
            "Mode de dessin:",
            ("freedraw", "line", "rect", "circle"),
            index=0
        )
        
        stroke_width = st.slider("Largeur du trait:", 1, 10, BRUSH_SIZE)
    
    with col2:
        # Ajouter un sélecteur pour le dossier de sauvegarde
        symbol_type = st.selectbox(
            "Dossier de sauvegarde:",
            ("symbole", "autres", "cercle", "triangle", "carré", "losange"),
            index=0
        )
        
        save_option = st.checkbox("Sauvegarder après reconnaissance", value=True)
        
        # Bouton pour sauvegarder immédiatement sans reconnaissance
        st.write("ou")
        save_only_mode = st.checkbox("Mode sauvegarde uniquement", value=False,
                                    help="Activer pour sauvegarder sans faire de reconnaissance")
    
    # Canvas pour le dessin
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="black",
        background_color="white",
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    
    # Actions sur le dessin
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Effacer"):
            # Réinitialiser le canvas (via le key)
            st.experimental_rerun()
    
    with col2:
        # Bouton pour sauvegarder directement
        if st.button("Sauvegarder maintenant"):
            if canvas_result.image_data is None:
                st.warning("Veuillez dessiner quelque chose d'abord!")
            else:
                saved_path = save_image(canvas_result.image_data, symbol_type)
                if saved_path:
                    st.success(f"Image sauvegardée dans le dossier '{symbol_type}'")
    
    # Si mode sauvegarde uniquement, on s'arrête ici
    if save_only_mode:
        st.info("Mode sauvegarde uniquement activé. Appuyez sur 'Sauvegarder maintenant' pour enregistrer votre dessin.")
        return
    
    # Reconnaissance de la gravure
    if st.button("Reconnaître la gravure"):
        if canvas_result.image_data is None:
            st.warning("Veuillez dessiner quelque chose d'abord!")
            return
        
        try:
            # Récupérer l'image du canvas
            img_data = canvas_result.image_data
            
            # Convertir en niveaux de gris
            img_gray = Image.fromarray(img_data).convert("L")
            
            # Afficher l'image
            st.image(img_gray, caption="Gravure dessinée", width=150)
            
            # Générer des variantes de l'image
            variants = generate_variants(img_gray)
            
            # Collecter les votes de tous les variants
            all_results = []
            
            # Fonction pour calculer l'embedding et trouver les similarités
            with st.spinner("Analyse en cours..."):
                # Traiter chaque variante
                for i, variant in enumerate(variants):
                    # Convertir en PIL Image si ce n'est pas déjà le cas
                    if not isinstance(variant, Image.Image):
                        variant = Image.fromarray(variant)
                    
                    # Prétraiter l'image pour le modèle
                    img_tensor = preprocess_image(variant)
                    img_tensor = img_tensor.to(device)
                    
                    # Calculer l'embedding avec le modèle
                    with torch.no_grad():
                        embedding = model.forward_one(img_tensor).cpu().numpy()[0]
                    
                    # Trouver les images similaires
                    results = find_similar_images(embedding, reference_images, top_n=3)
                    all_results.extend(results)
                
                # Agréger les résultats en comptant les occurrences de chaque classe
                class_votes = {}
                for img, score in all_results:
                    class_name = img['class']
                    if class_name not in class_votes:
                        class_votes[class_name] = 0
                    class_votes[class_name] += score
                
                # Trier les classes par nombre de votes
                sorted_classes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
                
                # Afficher les résultats
                st.subheader("Résultats de la reconnaissance")
                
                if sorted_classes:
                    # Prendre la meilleure correspondance
                    best_match = sorted_classes[0][0]
                    confidence = sorted_classes[0][1] / sum(score for _, score in sorted_classes)
                    
                    st.success(f"Gravure reconnue: **{best_match}** (confiance: {confidence:.2%})")
                    
                    # Afficher toutes les classes détectées avec leur score
                    st.write("Toutes les correspondances:")
                    for class_name, votes in sorted_classes:
                        normalized_score = votes / sum(score for _, score in sorted_classes)
                        st.write(f"- {class_name}: {normalized_score:.2%}")
                    
                    # Afficher les images les plus similaires
                    st.subheader("Images similaires")
                    
                    # Trouver les meilleures images pour chaque classe
                    shown_images = set()
                    cols = st.columns(3)
                    col_idx = 0
                    
                    for img, score in sorted(all_results, key=lambda x: x[1], reverse=True):
                        img_path = img['path']
                        if img_path not in shown_images and len(shown_images) < 3:
                            shown_images.add(img_path)
                            try:
                                image = Image.open(img_path)
                                with cols[col_idx]:
                                    st.image(image, caption=f"{img['class']} ({score:.2f})", width=100)
                                    col_idx = (col_idx + 1) % 3
                            except Exception as e:
                                st.warning(f"Impossible d'afficher l'image {img_path}: {e}")
                else:
                    st.warning("Aucune correspondance trouvée")
            
            # Sauvegarder l'image si demandé
            if save_option:
                saved_path = save_image(img_data, symbol_type)
                if saved_path:
                    st.write(f"Image sauvegardée dans le dossier '{symbol_type}' pour référence ultérieure")
        
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la reconnaissance: {e}")
            st.error("Vérifiez que le modèle est correctement chargé et que les images de référence sont disponibles.")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main() 