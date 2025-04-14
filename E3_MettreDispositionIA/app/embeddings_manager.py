import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
import torch.nn.functional as F

from .model import get_embedding


class EmbeddingsManager:
    """
    Gestionnaire pour les embeddings des gravures
    """
    def __init__(self, embeddings_path: str = None):
        self.embeddings_dict = {}  # {id_gravure: embedding}
        self.gravures_info = {}    # {id_gravure: {code, indice, etc.}}
        
        if embeddings_path and os.path.exists(embeddings_path):
            self.load_embeddings(embeddings_path)
    
    def load_embeddings(self, file_path: str) -> bool:
        """
        Charge les embeddings depuis un fichier pickle
        
        Args:
            file_path: Chemin du fichier pickle contenant les embeddings
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            print(f"Chargement des embeddings depuis {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings_dict = data.get('embeddings', {})
                self.gravures_info = data.get('info', {})
            print(f"Embeddings chargés avec succès: {len(self.embeddings_dict)} gravures")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des embeddings: {e}")
            return False
    
    def save_embeddings(self, file_path: str) -> bool:
        """
        Sauvegarde les embeddings dans un fichier pickle
        
        Args:
            file_path: Chemin où sauvegarder le fichier pickle
            
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            print(f"Sauvegarde des embeddings dans {file_path}")
            data = {
                'embeddings': self.embeddings_dict,
                'info': self.gravures_info
            }
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Embeddings sauvegardés avec succès: {len(self.embeddings_dict)} gravures")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_embedding(self, id_gravure: int, embedding: np.ndarray, info: dict) -> None:
        """
        Ajoute un embedding et les informations associées à la gravure
        
        Args:
            id_gravure: Identifiant unique de la gravure
            embedding: Vecteur d'embedding numpy
            info: Dictionnaire avec les informations de la gravure (code, indice, etc.)
        """
        self.embeddings_dict[id_gravure] = embedding
        self.gravures_info[id_gravure] = info
    
    def remove_embedding(self, id_gravure: int) -> bool:
        """
        Supprime un embedding et les informations associées
        
        Args:
            id_gravure: Identifiant de la gravure à supprimer
            
        Returns:
            True si supprimé avec succès, False si non trouvé
        """
        if id_gravure in self.embeddings_dict:
            del self.embeddings_dict[id_gravure]
            del self.gravures_info[id_gravure]
            return True
        return False
    
    def find_closest_gravure(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Tuple[int, float]]:
        """
        Trouve les gravures les plus proches d'un embedding de requête
        
        Args:
            query_embedding: Embedding de la gravure recherchée
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de tuples (id_gravure, score_similarité) triés par similarité décroissante
        """
        if not self.embeddings_dict:
            return []
        
        # Calculer la similarité avec tous les embeddings stockés
        similarities = []
        
        for id_gravure, embedding in self.embeddings_dict.items():
            # Distance euclidienne
            distance = np.linalg.norm(query_embedding - embedding)
            # Convertir en similarité (plus la distance est petite, plus la similarité est grande)
            similarity = np.exp(-distance)  # Fonction exponentielle décroissante
            similarities.append((id_gravure, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les top_k résultats
        return similarities[:top_k]
    
    def get_gravure_info(self, id_gravure: int) -> Optional[dict]:
        """
        Récupère les informations associées à une gravure
        
        Args:
            id_gravure: Identifiant de la gravure
            
        Returns:
            Dictionnaire d'informations ou None si non trouvé
        """
        return self.gravures_info.get(id_gravure)
    
    def get_all_gravures(self) -> List[dict]:
        """
        Récupère toutes les informations des gravures
        
        Returns:
            Liste de dictionnaires d'informations pour chaque gravure
        """
        return [{'id': id_gravure, **info} for id_gravure, info in self.gravures_info.items()]
    
    def generate_embeddings(self, model, images_dir: str, save_path: str = None) -> None:
        """
        Génère les embeddings pour toutes les images dans un répertoire
        
        Args:
            model: Modèle siamois pour générer les embeddings
            images_dir: Répertoire contenant les images de gravures
            save_path: Chemin où sauvegarder les embeddings (optionnel)
        """
        # Réinitialiser les dictionnaires
        self.embeddings_dict = {}
        self.gravures_info = {}
        
        # Parcourir les fichiers d'images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        id_counter = 1
        
        print(f"Recherche des images dans {images_dir}")
        files_list = os.listdir(images_dir)
        print(f"Nombre de fichiers trouvés: {len(files_list)}")
        
        # Filtrer les fichiers d'images
        image_files = [f for f in files_list if any(f.lower().endswith(ext) for ext in image_extensions)]
        print(f"Nombre d'images à traiter: {len(image_files)}")
        
        processed_images = 0
        for filename in image_files:
            # Vérifier si c'est une image
            file_path = os.path.join(images_dir, filename)
            
            try:
                print(f"Traitement de l'image {filename}...")
                
                # Charger et prétraiter l'image
                image = Image.open(file_path).convert('L')  # Convertir en niveaux de gris
                image = image.resize((64, 64))  # Redimensionner
                
                # Convertir en tensor
                image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Ajouter dimensions batch et canal
                
                # Générer l'embedding
                embedding = get_embedding(model, image_tensor)
                
                # Extraire le code de la gravure du nom de fichier
                # Format supposé: marque_indice.jpg (ex: varilux_1.67.jpg)
                name_parts = os.path.splitext(filename)[0].split('_')
                
                code = name_parts[0] if len(name_parts) > 0 else "Unknown"
                indice = float(name_parts[1]) if len(name_parts) > 1 and name_parts[1].replace('.', '', 1).isdigit() else 0.0
                
                # Ajouter à notre dictionnaire
                self.add_embedding(id_counter, embedding, {
                    'code': code,
                    'indice': indice,
                    'filename': filename
                })
                
                id_counter += 1
                processed_images += 1
                
                # Afficher la progression
                if processed_images % 10 == 0 or processed_images == len(image_files):
                    print(f"Progression: {processed_images}/{len(image_files)} images traitées")
                
            except Exception as e:
                print(f"Erreur avec l'image {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Traitement terminé: {processed_images}/{len(image_files)} images traitées avec succès")
        
        # Sauvegarder si un chemin est fourni
        if save_path:
            success = self.save_embeddings(save_path)
            if success:
                print(f"Embeddings sauvegardés dans {save_path}")
            else:
                print(f"Échec de la sauvegarde des embeddings dans {save_path}")
        else:
            print("Aucun chemin de sauvegarde fourni, les embeddings n'ont pas été sauvegardés")
    
    def print_stats(self):
        """
        Affiche les statistiques des embeddings
        """
        if not self.embeddings_dict:
            print("Aucun embedding trouvé.")
            return
            
        print(f"Nombre total d'embeddings: {len(self.embeddings_dict)}")
        
        # Examiner la structure des embeddings
        id_sample = next(iter(self.embeddings_dict.keys()))
        print(f"Structure d'un élément (ID: {id_sample}):")
        data = self.embeddings_dict[id_sample]
        print(f"Type: {type(data)}")
        print(f"Structure: {data}")
        
        # Si c'est un tuple, examiner le contenu
        if isinstance(data, tuple):
            print(f"Longueur du tuple: {len(data)}")
            for i, item in enumerate(data):
                print(f"Item {i}: Type = {type(item)}")
                if i == 1 and isinstance(item, dict):  # Si c'est un dictionnaire de métadonnées
                    print(f"Contenu du dictionnaire: {item}")
                    
        # Compter les classes
        class_counts = {}
        
        try:
            for id_gravure, data in self.embeddings_dict.items():
                # Extraire les informations sur la classe
                info = None
                if isinstance(data, tuple) and len(data) >= 2:
                    embedding, info = data
                
                # Si pas d'info ou si l'info n'est pas un dictionnaire, passer à l'élément suivant
                if info is None or not isinstance(info, dict):
                    print(f"ID {id_gravure}: Pas d'informations valides")
                    continue
                
                # Extraire la classe
                class_name = info.get('code', 'inconnu')
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
            
            # Afficher les résultats
            if class_counts:
                print("\nDistribution des classes:")
                for class_name, count in class_counts.items():
                    print(f"  - {class_name}: {count} images ({count/len(self.embeddings_dict)*100:.1f}%)")
            else:
                print("\nAucune information de classe trouvée dans les embeddings.")
                
        except Exception as e:
            print(f"Erreur lors de l'analyse des embeddings: {e}")
            import traceback
            traceback.print_exc() 