import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Constante pour la taille des images
IMAGE_SIZE = 224

class EfficientNetEmbedding(nn.Module):
    """
    Modèle d'embeddings basé sur EfficientNet avec une structure siamoise
    et optimisé pour la Triplet Loss
    """
    def __init__(self, embedding_dim=256, pretrained=True):
        """
        Args:
            embedding_dim: Dimension du vecteur d'embedding final
            pretrained: Utiliser les poids préentraînés sur ImageNet
        """
        super(EfficientNetEmbedding, self).__init__()
        
        # Charger EfficientNet-B0 préentraîné
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Récupérer la dimension de sortie du backbone
        last_channel = self.backbone.classifier[1].in_features
        
        # Remplacer la tête de classification par notre MLP d'embedding
        self.backbone.classifier = nn.Identity()
        
        # Créer la tête MLP pour l'embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
        
        # Conversion en grayscale pour l'entrée
        self.grayscale_conv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.grayscale_conv.weight)
    
    def forward_one(self, x):
        """
        Forward pass pour une seule image
        Args:
            x: Image en niveaux de gris (B, 1, H, W)
        Returns:
            Embedding normalisé
        """
        # Convertir les images grayscale en 3 canaux
        if x.size(1) == 1:
            x = self.grayscale_conv(x)
        
        # Extraction des features par le backbone
        features = self.backbone(x)
        
        # Si batch_size est 1, mettre BatchNorm en mode évaluation temporairement
        batch_size = x.size(0)
        if batch_size == 1:
            # Sauvegarder le mode d'entraînement actuel des modules BatchNorm
            training_status = {}
            for name, module in self.embedding_head.named_modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                    training_status[name] = module.training
                    module.eval()
        
        # Projection dans l'espace d'embedding
        embedding = self.embedding_head(features)
        
        # Restaurer le mode d'entraînement des modules BatchNorm
        if batch_size == 1:
            for name, module in self.embedding_head.named_modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) and name in training_status:
                    module.training = training_status[name]
        
        # Normalisation L2 de l'embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass pour un triplet d'images
        """
        anchor_embedding = self.forward_one(anchor)
        positive_embedding = self.forward_one(positive)
        negative_embedding = self.forward_one(negative)
        
        return anchor_embedding, positive_embedding, negative_embedding

class TripletLoss(nn.Module):
    """
    Triplet Margin Loss avec mining intelligent des triplets
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, anchor, positive, negative):
        """
        Calcule la Triplet Loss
        
        Args:
            anchor: Embeddings des images d'ancrage
            positive: Embeddings des images positives (même classe que anchor)
            negative: Embeddings des images négatives (classe différente)
        """
        # Distance entre anchor et positive
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        
        # Distance entre anchor et negative
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Créer une variable cible pour la fonction de ranking loss
        # 1 signifie que dist_neg devrait être plus grand que dist_pos
        target = torch.ones_like(dist_pos)
        
        # Calculer la loss
        loss = self.ranking_loss(dist_neg, dist_pos, target)
        
        return loss

class HardTripletLoss(nn.Module):
    """
    Triplet Loss avec mining de hard negatives et positives en ligne
    """
    def __init__(self, margin=0.3, mining_type='semi-hard'):
        """
        Args:
            margin: Marge entre les distances positives et négatives
            mining_type: Type de mining: 'semi-hard', 'hard', ou 'all'
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.mining_type = mining_type
    
    def forward(self, embeddings, labels):
        """
        Calcule la Triplet Loss avec mining
        
        Args:
            embeddings: Tous les embeddings du batch (B, embedding_dim)
            labels: Les étiquettes correspondantes (B,)
        """
        # Calculer la matrice de distance pairwise
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Pour chaque ancre, trouver tous les positifs et négatifs
        matches = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        # Mettre les diagonales à False (la distance d'un point avec lui-même)
        matches.fill_diagonal_(False)
        
        # Masque des non-matches (négatifs)
        non_matches = ~matches
        
        # Loss pour les triplets
        loss = 0.0
        num_triplets = 0
        
        batch_size = embeddings.size(0)
        
        if self.mining_type == 'hard':
            # Pour chaque ancre, trouver le positive le plus difficile (distance max)
            # et le negative le plus difficile (distance min)
            for i in range(batch_size):
                # Trouver les indices des positifs
                pos_indices = torch.where(matches[i])[0]
                
                if len(pos_indices) == 0:
                    continue
                
                # Trouver les indices des négatifs
                neg_indices = torch.where(non_matches[i])[0]
                
                if len(neg_indices) == 0:
                    continue
                
                # Distance aux positifs
                pos_dists = dist_matrix[i, pos_indices]
                
                # Distance aux négatifs
                neg_dists = dist_matrix[i, neg_indices]
                
                # Le positive le plus difficile (plus éloigné)
                hardest_pos_dist = torch.max(pos_dists)
                
                # Le negative le plus difficile (plus proche)
                hardest_neg_dist = torch.min(neg_dists)
                
                # Calculer la loss pour ce triplet
                triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
                
                if triplet_loss > 0:
                    loss += triplet_loss
                    num_triplets += 1
        
        elif self.mining_type == 'semi-hard':
            # Pour chaque ancre, calculer la loss pour tous les positifs et les semi-hard negatives
            for i in range(batch_size):
                # Trouver les indices des positifs
                pos_indices = torch.where(matches[i])[0]
                
                if len(pos_indices) == 0:
                    continue
                
                # Trouver les indices des négatifs
                neg_indices = torch.where(non_matches[i])[0]
                
                if len(neg_indices) == 0:
                    continue
                
                # Distance aux positifs
                pos_dists = dist_matrix[i, pos_indices]
                
                # Semi-hard negative mining (négatifs qui sont plus proches que le positif le plus éloigné)
                neg_dists = dist_matrix[i, neg_indices]
                
                # Pour chaque positif, trouver les semi-hard negatives
                for pos_idx, pos_dist in enumerate(pos_dists):
                    # Semi-hard: négatifs plus proches que le positif, mais pas trop faciles
                    # a < neg < pos + margin
                    semi_hard_negs = (neg_dists < pos_dist + self.margin) & (neg_dists > pos_dist)
                    
                    if torch.sum(semi_hard_negs) > 0:
                        # Utiliser les semi-hard negatives
                        semi_hard_neg_dists = neg_dists[semi_hard_negs]
                        triplet_loss = F.relu(pos_dist.repeat(len(semi_hard_neg_dists)) - semi_hard_neg_dists + self.margin)
                        loss += torch.sum(triplet_loss)
                        num_triplets += torch.sum(semi_hard_negs).item()
                    else:
                        # Pas de semi-hard, utiliser le negative le plus difficile
                        hardest_neg_dist = torch.min(neg_dists)
                        triplet_loss = F.relu(pos_dist - hardest_neg_dist + self.margin)
                        
                        if triplet_loss > 0:
                            loss += triplet_loss
                            num_triplets += 1
        else:  # 'all' - toutes les combinaisons de triplets valides
            for i in range(batch_size):
                # Trouver les indices des positifs
                pos_indices = torch.where(matches[i])[0]
                
                if len(pos_indices) == 0:
                    continue
                
                # Trouver les indices des négatifs
                neg_indices = torch.where(non_matches[i])[0]
                
                if len(neg_indices) == 0:
                    continue
                
                # Distance aux positifs
                pos_dists = dist_matrix[i, pos_indices]
                
                # Distance aux négatifs
                neg_dists = dist_matrix[i, neg_indices]
                
                # Calculer la loss pour toutes les combinaisons (pos, neg)
                for pos_dist in pos_dists:
                    for neg_dist in neg_dists:
                        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
                        if triplet_loss > 0:
                            loss += triplet_loss
                            num_triplets += 1
        
        # Retourner la loss moyenne des triplets
        if num_triplets > 0:
            return loss / num_triplets
        else:
            # Si aucun triplet valide, retourner 0
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

class TripletDataset(Dataset):
    """
    Dataset pour l'entraînement avec Triplet Loss
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Dossier contenant les sous-répertoires de classes
            transform: Transformations à appliquer aux images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Liste des classes (dossiers)
        self.classes = [d for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Liste des images avec leurs classes
        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Pour l'entraînement standard (non triplet), retourne une image et sa classe
        """
        img_path, label = self.samples[idx]
        
        try:
            # Essayer d'ouvrir l'image et la convertir en niveaux de gris
            image = Image.open(img_path).convert('L')
            
            # Appliquer les transformations
            if self.transform:
                image = self.transform(image)
            
            # Retourner l'image transformée et son label
            return image, label
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path}: {e}")
            # Retourner un tensor noir en cas d'erreur
            dummy_tensor = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE))
            return dummy_tensor, label

def extract_embedding(model, image_tensor, device='cpu'):
    """
    Extrait l'embedding d'une image
    
    Args:
        model: Modèle EfficientNet
        image_tensor: Tensor de l'image (1, 1, H, W)
        device: Device sur lequel effectuer le calcul
        
    Returns:
        L'embedding de l'image (vecteur numpy)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        embedding = model.forward_one(image_tensor)
        return embedding.cpu().numpy()[0]

def load_model(model_path, device='cpu', embedding_dim=256):
    """
    Charge un modèle EfficientNet à partir d'un fichier
    
    Args:
        model_path: Chemin vers le fichier du modèle
        device: Device sur lequel charger le modèle (cpu ou cuda)
        embedding_dim: Dimension de l'embedding
        
    Returns:
        Le modèle chargé
    """
    model = EfficientNetEmbedding(embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_model(model, save_path):
    """
    Sauvegarde le modèle
    
    Args:
        model: Modèle à sauvegarder
        save_path: Chemin où sauvegarder le modèle
    """
    # Créer le dossier parent si nécessaire
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def compare_embeddings(embedding1, embedding2):
    """
    Compare deux embeddings et retourne leur similarité cosinus
    
    Args:
        embedding1, embedding2: Vecteurs d'embedding numpy
        
    Returns:
        Score de similarité cosinus (1 = identique, 0 = orthogonal, -1 = opposé)
    """
    # Normaliser si ce n'est pas déjà fait
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Similarité cosinus
    similarity = np.dot(embedding1, embedding2)
    
    # Ramener à [0, 1] pour faciliter l'interprétation
    similarity = (similarity + 1) / 2
    
    return similarity 