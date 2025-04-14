import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random

class SiameseNetwork(nn.Module):
    """
    Réseau siamois pour la comparaison d'images de gravures
    Architecture équilibrée avec régularisation
    """
    def __init__(self, embedding_dim=128, dropout_rate=0.3):
        """
        Args:
            embedding_dim: Dimension de l'embedding de sortie
            dropout_rate: Taux de dropout pour la régularisation
        """
        super(SiameseNetwork, self).__init__()
        
        # Première couche convolutionnelle
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Deuxième couche convolutionnelle
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Troisième couche convolutionnelle
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Calculer la taille des features après la convolution
        feature_size = 8 * 8 * 128
        
        # Couches fully connected
        self.fc1 = nn.Linear(feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.prelu_fc1 = nn.PReLU()
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, embedding_dim)
        self.bn_fc2 = nn.BatchNorm1d(embedding_dim)
    
    def forward_one(self, x):
        """
        Forward pass pour une image
        """
        # Couches convolutionnelles avec PReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Aplatir
        x = x.view(x.size(0), -1)
        
        # Couches fully connected avec PReLU
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.prelu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        
        # Normaliser l'embedding
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass pour une paire d'images
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        return output1, output2

class ContrastiveLoss(nn.Module):
    """
    Loss contrastive pour le réseau siamois
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        Calcule la loss contrastive
        
        Args:
            output1: Premier embedding
            output2: Deuxième embedding
            label: 1 si les deux images sont de la même classe, 0 sinon
        """
        # Calculer la distance euclidienne entre les embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Calculer la loss contrastive
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                     label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive

class TripletLoss(nn.Module):
    """
    Loss triplet pour améliorer la structure des embeddings
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Calculer les distances
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        # Calculer la loss triplet avec une marge
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)

def load_model(model_path, device='cpu', embedding_dim=128, dropout_rate=0.2):
    """
    Charge un modèle siamois à partir d'un fichier
    
    Args:
        model_path: Chemin vers le fichier du modèle
        device: Device sur lequel charger le modèle (cpu ou cuda)
        embedding_dim: Dimension de l'embedding
        
    Returns:
        Le modèle chargé
    """
    model = SiameseNetwork(embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_embedding(model, image_tensor, device='cpu'):
    """
    Calcule l'embedding d'une image
    
    Args:
        model: Modèle siamois
        image_tensor: Tensor de l'image (1, 1, H, W)
        device: Device sur lequel effectuer le calcul
        
    Returns:
        L'embedding de l'image (vecteur numpy)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        embedding = model.forward_one(image_tensor)
        return embedding.cpu().numpy()[0]

def compare_embeddings(embedding1, embedding2):
    """
    Compare deux embeddings et retourne leur similarité
    
    Args:
        embedding1, embedding2: Vecteurs d'embedding numpy
        
    Returns:
        Score de similarité (1 = identique, 0 = complètement différent)
    """
    # Distance euclidienne
    distance = np.linalg.norm(embedding1 - embedding2)
    
    # Conversion de la distance en similarité (1 = identique, 0 = très différent)
    # On utilise une fonction exponentielle décroissante
    similarity = np.exp(-distance)
    
    return similarity

class SiameseDataset(Dataset):
    """
    Dataset pour l'entraînement du réseau siamois
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Répertoire contenant les sous-dossiers d'images
            transform: Transformations à appliquer aux images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        
        # Créer une liste de tous les chemins d'images avec leurs classes
        self.image_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append((os.path.join(class_dir, img_name), class_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Obtenir une image aléatoire
        img1_path, class1 = self.image_paths[idx]
        
        # 50% de chance de prendre une image de la même classe
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # Trouver une autre image de la même classe
            while True:
                idx2 = random.randint(0, len(self.image_paths) - 1)
                img2_path, class2 = self.image_paths[idx2]
                if class2 == class1 and img2_path != img1_path:
                    break
        else:
            # Trouver une image d'une classe différente
            while True:
                idx2 = random.randint(0, len(self.image_paths) - 1)
                img2_path, class2 = self.image_paths[idx2]
                if class2 != class1:
                    break
        
        # Charger les images
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
        
        # Appliquer les transformations si définies
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.FloatTensor([int(class1 == class2)])

def save_model(model, save_path):
    """
    Sauvegarde un modèle siamois dans un fichier
    
    Args:
        model: Modèle siamois à sauvegarder
        save_path: Chemin où sauvegarder le modèle
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}") 