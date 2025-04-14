import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
import numpy as np
import io
import os
import torch
import sys

# Ajouter le chemin pour les imports d'app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.model import SiameseNetwork, load_model
from app.embeddings_manager import EmbeddingsManager
from config import (
    BEST_MODEL_PATH,
    EMBEDDINGS_PATH, 
    CANVAS_WIDTH, 
    CANVAS_HEIGHT, 
    DEFAULT_BRUSH_SIZE,
    IMAGE_SIZE
)


class GravureDrawApp:
    """Application de dessin et reconnaissance de gravures optiques"""
    
    def __init__(self, root):
        """Initialise l'application
        
        Args:
            root: Fenêtre principale Tkinter
        """
        self.root = root
        self.root.title("Application de Dessin de Gravures")
        
        # Initialiser les composants
        self.model = None
        self.embeddings_manager = None
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.brush_size_var = None
        self.result_text = None
        self.canvas = None
        
        # État du dessin
        self.prev_x = None
        self.prev_y = None
        self.image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Charger le modèle
        self._load_model()
        
        # Configurer l'interface
        self._setup_ui()
    
    def _load_model(self):
        """Charge le modèle et les embeddings"""
        try:
            # Vérifier que les fichiers existent
            if not os.path.exists(BEST_MODEL_PATH):
                messagebox.showerror("Erreur", f"Fichier modèle non trouvé: {BEST_MODEL_PATH}")
                return
                
            if not os.path.exists(EMBEDDINGS_PATH):
                messagebox.showerror("Erreur", f"Fichier embeddings non trouvé: {EMBEDDINGS_PATH}")
                return
            
            # Charger le modèle
            self.model = load_model(BEST_MODEL_PATH)
            self.model.eval()  # Mettre en mode évaluation
            
            # Charger le gestionnaire d'embeddings
            self.embeddings_manager = EmbeddingsManager()
            self.embeddings_manager.load_embeddings(EMBEDDINGS_PATH)
            
            print(f"Modèle chargé depuis {BEST_MODEL_PATH}")
            print(f"Embeddings chargés depuis {EMBEDDINGS_PATH}")
            print(f"Nombre de gravures: {len(self.embeddings_manager.get_all_gravures())}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'initialisation du modèle: {str(e)}")
    
    def _setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self._setup_drawing_area(main_frame)
        self._setup_control_panel(main_frame)
    
    def _setup_drawing_area(self, parent):
        """Configure la zone de dessin
        
        Args:
            parent: Widget parent
        """
        # Frame de dessin
        draw_frame = ttk.LabelFrame(parent, text="Zone de dessin", padding="5")
        draw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas de dessin
        self.canvas = tk.Canvas(
            draw_frame, 
            width=CANVAS_WIDTH, 
            height=CANVAS_HEIGHT,
            bg="white", 
            bd=2, 
            relief=tk.SUNKEN
        )
        self.canvas.pack(padx=5, pady=5)
        
        # Événements de dessin
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
    
    def _setup_control_panel(self, parent):
        """Configure le panneau de contrôle
        
        Args:
            parent: Widget parent
        """
        # Frame de contrôle
        ctrl_frame = ttk.LabelFrame(parent, text="Contrôles", padding="5")
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Contrôle de la taille du pinceau
        self._setup_brush_size_control(ctrl_frame)
        
        # Boutons d'action
        self._setup_action_buttons(ctrl_frame)
        
        # Zone de résultats
        self._setup_results_area(ctrl_frame)
    
    def _setup_brush_size_control(self, parent):
        """Configure le contrôle de taille du pinceau
        
        Args:
            parent: Widget parent
        """
        # Frame pour la taille du pinceau
        size_frame = ttk.Frame(parent)
        size_frame.pack(fill=tk.X, pady=5)
        
        # Label et slider
        ttk.Label(size_frame, text="Taille du pinceau:").pack(side=tk.LEFT, padx=5)
        self.brush_size_var = tk.IntVar(value=self.brush_size)
        
        brush_size_scale = ttk.Scale(
            size_frame, 
            from_=1, 
            to=20, 
            orient=tk.HORIZONTAL,
            variable=self.brush_size_var, 
            command=self.update_brush_size
        )
        brush_size_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def _setup_action_buttons(self, parent):
        """Configure les boutons d'action
        
        Args:
            parent: Widget parent
        """
        # Frame pour les boutons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Boutons
        ttk.Button(btn_frame, text="Effacer", command=self.clear_canvas).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Reconnaître", command=self.recognize_drawing).pack(fill=tk.X, pady=2)
    
    def _setup_results_area(self, parent):
        """Configure la zone de résultats
        
        Args:
            parent: Widget parent
        """
        # Frame pour les résultats
        result_frame = ttk.LabelFrame(parent, text="Résultats", padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Zone de texte pour les résultats
        self.result_text = tk.Text(result_frame, height=10, width=30, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def start_draw(self, event):
        """Commence le dessin quand le bouton gauche est pressé
        
        Args:
            event: Événement de la souris
        """
        # Réinitialiser les résultats quand on commence un nouveau dessin
        self.result_text.delete(1.0, tk.END)
        self.prev_x = event.x
        self.prev_y = event.y
    
    def draw_line(self, event):
        """Dessine une ligne lors du déplacement de la souris
        
        Args:
            event: Événement de la souris
        """
        if not (self.prev_x and self.prev_y):
            return
            
        # Dessiner sur le canvas Tkinter
        self.canvas.create_line(
            self.prev_x, self.prev_y, event.x, event.y,
            width=self.brush_size, 
            fill="black", 
            capstyle=tk.ROUND, 
            smooth=tk.TRUE
        )
        
        # Dessiner sur l'image PIL
        self.draw.line(
            [(self.prev_x, self.prev_y), (event.x, event.y)],
            fill=0,  # Noir pour l'image en niveaux de gris
            width=self.brush_size
        )
        
        # Mettre à jour les coordonnées précédentes
        self.prev_x = event.x
        self.prev_y = event.y
    
    def clear_canvas(self):
        """Efface le canvas et réinitialise l'image"""
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=255)
        self.draw = ImageDraw.Draw(self.image)
        # Vider aussi la zone de résultats
        self.result_text.delete(1.0, tk.END)
    
    def update_brush_size(self, event=None):
        """Met à jour la taille du pinceau"""
        self.brush_size = self.brush_size_var.get()
    
    def preprocess_image(self, image):
        """Prétraite l'image pour la reconnaissance
        
        Args:
            image: Image PIL à prétraiter
            
        Returns:
            numpy.ndarray: Image prétraitée
        """
        # Redimensionner à 64x64 pixels
        img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        
        # Convertir en niveaux de gris si ce n'est pas déjà le cas
        if img_resized.mode != 'L':
            img_resized = img_resized.convert('L')
        
        # Normaliser les valeurs des pixels entre 0 et 1
        img_array = np.array(img_resized) / 255.0
        
        return img_array
    
    def _get_image_embedding(self, processed_image):
        """Calcule l'embedding d'une image prétraitée
        
        Args:
            processed_image: Image prétraitée
            
        Returns:
            numpy.ndarray: Embedding de l'image
        """
        # Convertir en tensor
        image_tensor = torch.from_numpy(processed_image).float().unsqueeze(0).unsqueeze(0)
        
        # Calculer l'embedding avec le modèle
        with torch.no_grad():
            embedding = self.model.forward_one(image_tensor).cpu().numpy()[0]
            
        return embedding
    
    def _display_recognition_results(self, results):
        """Affiche les résultats de la reconnaissance
        
        Args:
            results: Liste de tuples (id_gravure, score de similarité)
        """
        self.result_text.delete(1.0, tk.END)
        
        # Afficher les meilleures correspondances
        for i, (id_gravure, similarity) in enumerate(results):
            gravure_info = self.embeddings_manager.get_gravure_info(id_gravure)
            code = gravure_info.get('code', 'inconnu')
            self.result_text.insert(tk.END, f"Classe reconnue: {code}\n")
            self.result_text.insert(tk.END, f"Score: {similarity:.4f}\n\n")
    
    def recognize_drawing(self):
        """Reconnaît le dessin avec le modèle local"""
        # Vérifier que le modèle est chargé
        if self.model is None or self.embeddings_manager is None:
            messagebox.showwarning("Attention", "Le modèle n'est pas initialisé")
            return
        
        try:
            # Prétraiter l'image
            processed_image = self.preprocess_image(self.image)
            
            # Obtenir l'embedding
            embedding = self._get_image_embedding(processed_image)
            
            # Trouver les gravures les plus proches
            results = self.embeddings_manager.find_closest_gravure(embedding, top_k=3)
            
            if not results:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Aucune gravure correspondante trouvée")
                return
            
            # Afficher les résultats
            self._display_recognition_results(results)
            
            # Sauvegarder l'image pour référence
            self.image.save("dernier_dessin.jpg")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Erreur: {str(e)}")


def main():
    """Point d'entrée de l'application"""
    root = tk.Tk()
    app = GravureDrawApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 