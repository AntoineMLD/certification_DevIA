import os
import sys
import unittest
import torch
import numpy as np

# Ajouter le répertoire parent au chemin Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.model import SiameseNetwork, ContrastiveLoss

class TestSiameseModel(unittest.TestCase):
    def setUp(self):
        # Initialiser le modèle pour les tests
        self.model = SiameseNetwork(embedding_dim=64)
        self.model.eval()
        
        # Créer quelques tensors d'entrée factices
        self.input1 = torch.randn(2, 1, 64, 64)  # Batch de 2 images
        self.input2 = torch.randn(2, 1, 64, 64)
        
        # Créer une loss
        self.criterion = ContrastiveLoss(margin=1.0)
    
    def test_model_output_shape(self):
        """Vérifier que le modèle produit des tensors de la bonne forme"""
        # Test avec une seule entrée
        output1 = self.model.forward_one(self.input1)
        self.assertEqual(output1.shape, (2, 64), "La forme de l'embedding devrait être (batch_size, embedding_dim)")
        
        # Test avec deux entrées
        output1, output2 = self.model(self.input1, self.input2)
        self.assertEqual(output1.shape, (2, 64), "La forme de l'embedding devrait être (batch_size, embedding_dim)")
        self.assertEqual(output2.shape, (2, 64), "La forme de l'embedding devrait être (batch_size, embedding_dim)")
    
    def test_embeddings_normalization(self):
        """Vérifier que les embeddings sont normalisés (norme L2 = 1)"""
        output = self.model.forward_one(self.input1)
        
        # Calculer la norme L2 pour chaque vecteur d'embedding
        norms = torch.norm(output, p=2, dim=1)
        
        # Vérifier que toutes les normes sont proches de 1
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, delta=1e-6, 
                                  msg="Les embeddings devraient être normalisés avec une norme L2 de 1")
    
    def test_contrastive_loss(self):
        """Vérifier que la loss contrastive fonctionne correctement"""
        # Créer des embeddings factices
        embedding1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        embedding2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        
        # Définir les labels (0 = différent, 1 = même)
        labels_same = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        
        # Calculer la loss
        loss = self.criterion(embedding1, embedding2, labels_same)
        
        # Vérifier que la loss est un scalaire
        self.assertEqual(loss.dim(), 0, "La loss devrait être un scalaire")
        
        # Vérifier que la loss est positive
        self.assertGreaterEqual(loss.item(), 0, "La loss contrastive devrait être positive")
    
    def test_same_image_similarity(self):
        """Vérifier que l'embedding d'une image est plus proche d'elle-même que d'une autre image"""
        # Créer une image factice et sa version avec du bruit
        original_image = torch.randn(1, 1, 64, 64)
        noisy_version = original_image + 0.1 * torch.randn(1, 1, 64, 64)
        different_image = torch.randn(1, 1, 64, 64)
        
        # Calculer les embeddings
        with torch.no_grad():
            embedding_original = self.model.forward_one(original_image)
            embedding_noisy = self.model.forward_one(noisy_version)
            embedding_different = self.model.forward_one(different_image)
        
        # Calculer les distances euclidiennes
        dist_to_self = torch.nn.functional.pairwise_distance(embedding_original, embedding_noisy)
        dist_to_different = torch.nn.functional.pairwise_distance(embedding_original, embedding_different)
        
        # La distance à soi-même (avec bruit) devrait être plus petite qu'à une image différente
        self.assertLess(dist_to_self.item(), dist_to_different.item(), 
                       "La distance à la version bruitée de l'image devrait être plus petite qu'à une image différente")

if __name__ == '__main__':
    unittest.main() 