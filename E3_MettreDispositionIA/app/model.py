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

def evaluate_embeddings_quality(model, test_loader, device='cpu'):
    """
    Évalue la qualité des embeddings en calculant les distances intra-classe et inter-classe
    
    Args:
        model: Modèle siamois entraîné
        test_loader: DataLoader pour les données de test
        device: Device sur lequel effectuer le calcul
        
    Returns:
        Un dictionnaire avec les métriques de qualité
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    class_embeddings = {}
    
    print("Calcul des embeddings pour l'évaluation...")
    with torch.no_grad():
        for batch in test_loader:
            # Support pour les deux types de datasets (standard et triplet)
            if len(batch) == 3 and not isinstance(batch[2], torch.Tensor):
                # Dataset triplet (anchor, positive, negative)
                imgs = batch[0]
                # Récupérer les labels à partir des chemins des fichiers
                labels = [os.path.basename(os.path.dirname(path)) 
                         for path, _ in test_loader.dataset.dataset.image_paths[:len(imgs)]]
            else:
                # Dataset standard (img1, img2, label)
                imgs = batch[0]
                # Récupérer les labels à partir des chemins des fichiers
                labels = [os.path.basename(os.path.dirname(path)) 
                         for path, _ in test_loader.dataset.dataset.image_paths[:len(imgs)]]
            
            # Calculer les embeddings
            imgs = imgs.to(device)
            embs = model.forward_one(imgs)
            embs = embs.cpu().numpy()
            
            # Stocker les embeddings et labels
            for i, (emb, label) in enumerate(zip(embs, labels)):
                all_embeddings.append(emb)
                all_labels.append(label)
                
                if label not in class_embeddings:
                    class_embeddings[label] = []
                class_embeddings[label].append(emb)
    
    # Convertir en arrays numpy
    all_embeddings = np.array(all_embeddings)
    
    # Calculer les distances moyennes intra-classe
    intra_class_distances = []
    for label, embeddings in class_embeddings.items():
        if len(embeddings) <= 1:
            continue
            
        embeddings = np.array(embeddings)
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                intra_class_distances.append(dist)
    
    # Calculer les distances moyennes inter-classe
    inter_class_distances = []
    labels = list(class_embeddings.keys())
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            label1, label2 = labels[i], labels[j]
            for emb1 in class_embeddings[label1]:
                for emb2 in class_embeddings[label2]:
                    dist = np.linalg.norm(emb1 - emb2)
                    inter_class_distances.append(dist)
    
    # Calculer les statistiques
    avg_intra_class = np.mean(intra_class_distances) if intra_class_distances else 0
    avg_inter_class = np.mean(inter_class_distances) if inter_class_distances else 0
    min_inter_class = np.min(inter_class_distances) if inter_class_distances else 0
    max_intra_class = np.max(intra_class_distances) if intra_class_distances else 0
    
    # Calculer le ratio (distance inter / distance intra)
    # Un ratio plus élevé indique de meilleurs embeddings
    ratio = avg_inter_class / avg_intra_class if avg_intra_class > 0 else 0
    
    # Retourner les résultats
    results = {
        "avg_intra_class_distance": avg_intra_class,
        "avg_inter_class_distance": avg_inter_class,
        "min_inter_class_distance": min_inter_class,
        "max_intra_class_distance": max_intra_class,
        "quality_ratio": ratio
    }
    
    return results

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