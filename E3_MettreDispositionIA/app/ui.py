import gradio as gr
import numpy as np
import torch
from PIL import Image
import os
import io

from . import model, embeddings_manager
from config import IMAGE_SIZE

def preprocess_image(image):
    """
    Prétraite l'image dessinée dans l'interface Gradio
    """
    # Convertir l'image en niveaux de gris si nécessaire
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Image RGB, convertir en niveaux de gris
        # On convertit d'abord en PIL
        pil_image = Image.fromarray(image)
        pil_image = pil_image.convert('L')
        image = np.array(pil_image)
    
    # Redimensionner à la taille attendue par le modèle
    pil_image = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Normaliser les valeurs
    image_array = np.array(pil_image) / 255.0
    
    return image_array

def predict_gravure(drawing):
    """
    Fonction de prédiction pour l'interface Gradio
    """
    if drawing is None:
        return "Veuillez dessiner une gravure", None, None
    
    try:
        # Prétraiter l'image dessinée
        processed_image = preprocess_image(drawing)
        
        # Convertir en tensor pour le modèle
        image_tensor = torch.from_numpy(processed_image).float().unsqueeze(0).unsqueeze(0)
        
        # Calculer l'embedding
        embedding = model.forward_one(image_tensor).detach().numpy()[0]
        
        # Trouver les gravures les plus proches
        results = embeddings_manager.find_closest_gravure(embedding, top_k=3)
        
        if not results:
            return "Aucune gravure trouvée dans la base", None, None
        
        # Récupérer les informations des 3 meilleures correspondances
        result_texts = []
        for idx, (id_gravure, similarity) in enumerate(results):
            info = embeddings_manager.get_gravure_info(id_gravure)
            result_texts.append(f"{idx+1}. {info['code']} (indice: {info['indice']}) - Similarité: {similarity:.2f}")
        
        # Obtenir la meilleure correspondance
        best_id, best_score = results[0]
        best_info = embeddings_manager.get_gravure_info(best_id)
        
        # Construire le chemin de l'image correspondante (s'il existe)
        image_path = os.path.join("./data/processed", best_info.get('filename', ''))
        image = None
        if os.path.exists(image_path):
            image = Image.open(image_path)
        
        # Retourner le résultat sous forme de texte, l'image de la gravure, et le score
        return "\n".join(result_texts), image, f"{best_score:.2f}"
    
    except Exception as e:
        return f"Erreur lors de la reconnaissance: {str(e)}", None, None

def create_ui():
    """
    Crée l'interface utilisateur Gradio
    """
    # Interface avec un canvas de dessin et les résultats
    interface = gr.Interface(
        fn=predict_gravure,
        inputs=gr.Sketchpad(),
        outputs=[
            gr.Textbox(label="Résultats"),
            gr.Image(label="Gravure reconnue"),
            gr.Label(label="Score")
        ],
        title="Reconnaissance de Gravures Optiques",
        description="""
        Dessinez une gravure optique et le système identifiera la gravure correspondante.
        
        Les gravures sont généralement de petits symboles gravés sur les verres de lunettes,
        indiquant le fabricant et parfois l'indice de réfraction.
        """,
        examples=[],
        allow_flagging="never"
    )
    
    return interface 