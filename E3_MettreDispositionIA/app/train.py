import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import seaborn as sns

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer la configuration
from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    IMAGE_SIZE,
    EMBEDDING_DIM,
    BATCH_SIZE,
    NUM_EPOCHS
)

from .model import SiameseNetwork, ContrastiveLoss, TripletLoss, SiameseDataset, save_model

class GravureDataset(Dataset):
    """
    Dataset pour l'entraînement du réseau siamois
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Répertoire contenant les images
            transform (callable, optional): Transformation à appliquer aux images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # Code de la gravure
        
        # Parcourir les fichiers
        for filename in os.listdir(root_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(root_dir, filename)
                
                # Extraire le code de la gravure du nom de fichier (ex: varilux_1.67.jpg -> varilux)
                label = os.path.splitext(filename)[0].split('_')[0]
                
                self.image_paths.append(path)
                self.labels.append(label)
        
        # Dictionnaire pour associer chaque label unique à un indice
        self.unique_labels = list(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Pour l'entraînement siamois, on retourne une paire d'images et un label indiquant
        si elles sont de la même classe (1) ou de classes différentes (0)
        """
        # Première image
        img1_path = self.image_paths[idx]
        img1_label = self.labels[idx]
        img1 = Image.open(img1_path).convert('L')  # Convertir en niveaux de gris
        
        # Avec une probabilité de 0.5, on choisit une image de la même classe
        should_get_same_class = random.random() < 0.5
        
        # Trouver une deuxième image
        if should_get_same_class:
            # Même classe
            same_class_indices = [i for i, label in enumerate(self.labels) if label == img1_label and i != idx]
            if same_class_indices:
                # S'il y a d'autres images de la même classe
                img2_idx = random.choice(same_class_indices)
            else:
                # Sinon on utilise la même image avec une augmentation
                img2_idx = idx
            
            # Label 1 signifie "même classe"
            target = 1.0
        else:
            # Classe différente
            different_class_indices = [i for i, label in enumerate(self.labels) if label != img1_label]
            if different_class_indices:
                img2_idx = random.choice(different_class_indices)
                # Label 0 signifie "classes différentes"
                target = 0.0
            else:
                # Si pas d'autre classe, on utilise la même image (cas rare)
                img2_idx = idx
                target = 1.0
        
        img2_path = self.image_paths[img2_idx]
        img2 = Image.open(img2_path).convert('L')
        
        # Appliquer les transformations si spécifiées
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.FloatTensor([target])

class SiameseTripletDataset(Dataset):
    """
    Dataset pour l'entraînement du réseau siamois avec triplet loss
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.classes = []
        
        # Parcourir les sous-répertoires (classes)
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Trouver toutes les images pour cette classe
            class_images = []
            for filename in os.listdir(class_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, filename)
                    class_images.append(img_path)
                    
            # Ajouter les images et leurs classes
            for img_path in class_images:
                self.image_paths.append((img_path, class_name))
                if class_name not in self.classes:
                    self.classes.append(class_name)
        
        # Créer un dictionnaire où les clés sont les classes et les valeurs sont les indices des images
        self.class_to_indices = {cls: [] for cls in self.classes}
        for idx, (_, cls) in enumerate(self.image_paths):
            self.class_to_indices[cls].append(idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Pour l'entraînement avec triplet loss, on retourne un triplet:
        - anchor: image de référence
        - positive: image de la même classe que l'ancrage
        - negative: image d'une classe différente
        """
        # Image d'ancrage
        anchor_path, anchor_class = self.image_paths[idx]
        anchor_img = Image.open(anchor_path).convert('L')
        
        # Sélectionner une image positive (même classe, différente de l'ancrage)
        positive_indices = [i for i in self.class_to_indices[anchor_class] if i != idx]
        
        # Si pas d'autre image de cette classe, utiliser la même image
        if not positive_indices:
            positive_idx = idx
        else:
            # Stratégie pour "hard positive": choisir une image différente mais de même classe
            positive_idx = random.choice(positive_indices)
        
        positive_path, _ = self.image_paths[positive_idx]
        positive_img = Image.open(positive_path).convert('L')
        
        # Sélectionner une image négative (classe différente)
        negative_classes = [cls for cls in self.classes if cls != anchor_class]
        
        if not negative_classes:
            # Si pas d'autre classe, utiliser une image de la même classe
            negative_idx = random.choice(self.class_to_indices[anchor_class])
        else:
            # Stratégie pour "hard negative": choisir une classe différente mais qui peut être visuellement similaire
            negative_class = random.choice(negative_classes)
            negative_idx = random.choice(self.class_to_indices[negative_class])
        
        negative_path, _ = self.image_paths[negative_idx]
        negative_img = Image.open(negative_path).convert('L')
        
        # Appliquer les transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img

def moving_average(data, window_size=3):
    """
    Calcule une moyenne glissante sur une liste de données
    """
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def visualize_embeddings(model, val_loader, device, epoch):
    """
    Visualise les embeddings avec t-SNE
    """
    model.eval()
    embeddings = []
    labels = []
    classes = []
    
    print("\nGénération des embeddings pour t-SNE...")
    with torch.no_grad():
        for img1, img2, label in tqdm(val_loader):
            img1 = img1.to(device)
            emb1 = model.forward_one(img1)
            embeddings.append(emb1.cpu().numpy())
            
            # Extraire les noms de classes des chemins d'images
            batch_classes = []
            for path in val_loader.dataset.dataset.image_paths[:len(img1)]:
                if isinstance(path, tuple):  # Si path est un tuple
                    path_str = path[0]  # Premier élément du tuple est le chemin
                else:
                    path_str = path
                # Utiliser os.path pour extraire le répertoire parent (nom de classe)
                class_name = os.path.basename(os.path.dirname(path_str))
                batch_classes.append(class_name)
            classes.extend(batch_classes)
    
    # Convertir en array numpy
    embeddings = np.vstack(embeddings)
    
    # Appliquer t-SNE
    print("Application de t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Créer le plot
    plt.figure(figsize=(12, 8))
    
    # Créer un mapping des classes vers des couleurs
    unique_classes = list(set(classes))
    color_map = dict(zip(unique_classes, sns.color_palette("husl", len(unique_classes))))
    colors = [color_map[c] for c in classes]
    
    # Tracer les points
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
    
    # Ajouter la légende
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color_map[label], label=label, markersize=8)
                      for label in unique_classes]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Visualisation t-SNE des embeddings (époque {epoch})')
    plt.tight_layout()
    plt.savefig(f'embeddings_tsne_epoch_{epoch}.png', bbox_inches='tight')
    plt.close()

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, device='cpu', scheduler=None):
    """
    Entraîne le modèle et suit les métriques d'entraînement et de validation
    """
    print("Début de l'entraînement...")
    
    # Pour stocker les losses
    train_losses = []
    val_losses = []
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 3  # Nombre d'époques à attendre avant d'arrêter
    patience_counter = 0
    best_model_state = None
    
    # Créer une figure pour le graphique en temps réel
    plt.figure(figsize=(10, 6))
    plt.ion()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Entraînement
        for i, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        # Calculer la loss moyenne d'entraînement
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                val_loss = criterion(output1, output2, label)
                val_running_loss += val_loss.item()
        
        # Calculer la loss moyenne de validation
        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Sauvegarder le meilleur modèle
            save_model(model, "model/best_siamese_model.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping! Pas d'amélioration depuis {patience} époques.")
            # Restaurer le meilleur modèle
            model.load_state_dict(best_model_state)
            break
        
        # Mettre à jour le scheduler si présent
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {epoch_train_loss:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        print(f'Best Validation Loss: {best_val_loss:.4f}')
        print(f'Patience Counter: {patience_counter}/{patience}')
        
        # Visualiser les embeddings tous les 5 époques
        if (epoch + 1) % 5 == 0:
            visualize_embeddings(model, val_loader, device, epoch + 1)
        
        # Mettre à jour le graphique en temps réel
        plt.clf()
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.5)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.5)
        
        # Ajouter les moyennes glissantes
        if len(train_losses) > 3:
            smoothed_train = moving_average(train_losses, 3)
            smoothed_val = moving_average(val_losses, 3)
            plt.plot(range(len(smoothed_train)), smoothed_train,
                    label='Training Loss (3 époques)', color='blue', linewidth=2)
            plt.plot(range(len(smoothed_val)), smoothed_val,
                    label='Validation Loss (3 époques)', color='red', linewidth=2)
        
        plt.title('Loss au cours de l\'entraînement')
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.draw()
        plt.pause(0.1)
    
    # Sauvegarder le graphique final
    plt.savefig('training_loss.png')
    plt.close()
    
    return train_losses, val_losses

def train_with_triplet(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, device='cpu', scheduler=None):
    """
    Entraîne le modèle avec triplet loss
    """
    print("Début de l'entraînement avec triplet loss...")
    
    # Pour stocker les losses
    train_losses = []
    val_losses = []
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 5  # Augmenté car triplet loss peut prendre plus de temps à converger
    patience_counter = 0
    best_model_state = None
    
    # Créer une figure pour le graphique en temps réel
    plt.figure(figsize=(10, 6))
    plt.ion()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Entraînement
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_out = model.forward_one(anchor)
            positive_out = model.forward_one(positive)
            negative_out = model.forward_one(negative)
            
            # Calculer la loss
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        # Calculer la loss moyenne d'entraînement
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                # Forward pass
                anchor_out = model.forward_one(anchor)
                positive_out = model.forward_one(positive)
                negative_out = model.forward_one(negative)
                
                # Calculer la loss
                val_loss = criterion(anchor_out, positive_out, negative_out)
                val_running_loss += val_loss.item()
        
        # Calculer la loss moyenne de validation
        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Sauvegarder le meilleur modèle
            save_model(model, "model/best_siamese_model.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping! Pas d'amélioration depuis {patience} époques.")
            # Restaurer le meilleur modèle
            model.load_state_dict(best_model_state)
            break
        
        # Mettre à jour le scheduler si présent
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {epoch_train_loss:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        print(f'Best Validation Loss: {best_val_loss:.4f}')
        print(f'Patience Counter: {patience_counter}/{patience}')
        
        # Visualiser les embeddings tous les 5 époques
        if (epoch + 1) % 5 == 0:
            visualize_embeddings(model, val_loader, device, epoch + 1)
        
        # Mettre à jour le graphique en temps réel
        plt.clf()
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.5)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.5)
        
        # Ajouter les moyennes glissantes
        if len(train_losses) > 3:
            smoothed_train = moving_average(train_losses, 3)
            smoothed_val = moving_average(val_losses, 3)
            plt.plot(range(len(smoothed_train)), smoothed_train,
                    label='Training Loss (3 époques)', color='blue', linewidth=2)
            plt.plot(range(len(smoothed_val)), smoothed_val,
                    label='Validation Loss (3 époques)', color='red', linewidth=2)
        
        plt.title('Loss au cours de l\'entraînement (Triplet)')
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.draw()
        plt.pause(0.1)
    
    # Sauvegarder le graphique final
    plt.savefig('training_loss_triplet.png')
    plt.close()
    
    return train_losses, val_losses

def visualize_batch(dataloader, num_samples=5):
    """
    Visualise quelques paires d'images du dataset avec leurs labels
    """
    # Obtenir un batch
    img1_batch, img2_batch, labels = next(iter(dataloader))
    
    plt.figure(figsize=(15, 3*num_samples))
    for i in range(min(num_samples, len(img1_batch))):
        # Première image
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(img1_batch[i].squeeze(), cmap='gray')
        plt.title(f'Image 1 (Paire {i+1})')
        plt.axis('off')
        
        # Deuxième image
        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(img2_batch[i].squeeze(), cmap='gray')
        plt.title(f'Image 2 (Label: {labels[i].item():.0f})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_visualization.png')
    plt.close()

def main():
    # Configuration avec hyperparamètres ajustés
    PROCESSED_DATA_DIR = "data/augmented_gravures"
    MODEL_OUTPUT_DIR = "model"
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001  # Réduit de 0.001 à 0.0001 (×0.1)
    WEIGHT_DECAY = 0.01
    DROPOUT_RATE = 0.3
    USE_TRIPLET_LOSS = True  # Utiliser TripletLoss au lieu de ContrastiveLoss
    MARGIN = 1.0  # Tester avec différentes valeurs: 0.5, 1.0, 2.0
    
    # Vérifier si CUDA est disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de {device}")
    
    # Transformations augmentées pour les images avec plus de distorsions
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=30,  # Augmenté de 15 à 30 degrés
                translate=(0.2, 0.2),  # Augmenté de 0.1 à 0.2
                scale=(0.8, 1.2),  # Augmenté de (0.9, 1.1) à (0.8, 1.2)
                shear=15  # Augmenté de 10 à 15
            )
        ], p=0.8),  # Augmenté de 0.7 à 0.8
        transforms.RandomApply([
            transforms.GaussianBlur(5, sigma=(0.1, 0.8))  # Augmenté de 3 à 5 et sigma de (0.1, 0.5) à (0.1, 0.8)
        ], p=0.4),  # Augmenté de 0.3 à 0.4
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),  # Augmenté de 15 à 30
        transforms.RandomPerspective(distortion_scale=0.5, p=0.3),  # Augmenté de 0.2 à 0.3
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Ajout du contraste
        transforms.ToTensor(),
    ])
    
    print("\nChargement des données...")
    if USE_TRIPLET_LOSS:
        # Utiliser le dataset pour triplet loss
        dataset = SiameseTripletDataset(PROCESSED_DATA_DIR, transform=transform)
        
        print(f"\nNombre total d'images dans le dataset: {len(dataset)}")
        print(f"Nombre de classes: {len(dataset.classes)}")
        print("\nClasses disponibles:")
        for class_name in dataset.classes:
            count = len(dataset.class_to_indices[class_name])
            print(f"- {class_name}: {count} images")
    else:
        # Utiliser le dataset standard
        dataset = SiameseDataset(PROCESSED_DATA_DIR, transform=transform)
        
        print(f"\nNombre total d'images dans le dataset: {len(dataset)}")
        print(f"Nombre de classes: {len(dataset.classes)}")
        print("\nClasses disponibles:")
        for class_name in dataset.classes:
            count = sum(1 for path, cls in dataset.image_paths if cls == class_name)
            print(f"- {class_name}: {count} images")
    
    # Séparer en train et validation (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nTaille du dataset d'entraînement: {len(train_dataset)}")
    print(f"Taille du dataset de validation: {len(val_dataset)}")
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialiser le modèle avec une architecture plus large
    model = SiameseNetwork(
        embedding_dim=256,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Choisir la loss function
    if USE_TRIPLET_LOSS:
        criterion = TripletLoss(margin=MARGIN)
    else:
        criterion = ContrastiveLoss(margin=MARGIN)
    
    # Utiliser Adam avec un learning rate réduit
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Scheduler plus patient
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Moins agressif
        patience=3,  # Plus patient
        verbose=True,
        min_lr=1e-6
    )
    
    # Entraîner le modèle
    if USE_TRIPLET_LOSS:
        train_losses, val_losses = train_with_triplet(
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            NUM_EPOCHS,
            device,
            scheduler
        )
    else:
        train_losses, val_losses = train_model(
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            NUM_EPOCHS,
            device,
            scheduler
        )
    
    # Sauvegarder le modèle
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, "siamese_model.pt")
    save_model(model, model_path)
    
    # Tracer les courbes de loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.5)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.5)
    
    # Ajouter les moyennes glissantes
    if len(train_losses) > 3:
        smoothed_train = moving_average(train_losses, 3)
        smoothed_val = moving_average(val_losses, 3)
        plt.plot(range(len(smoothed_train)), smoothed_train,
                label='Training Loss (3 époques)', color='blue', linewidth=2)
        plt.plot(range(len(smoothed_val)), smoothed_val,
                label='Validation Loss (3 époques)', color='red', linewidth=2)
    
    plt.title('Loss au cours de l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss_final.png')
    plt.close()
    
    print("Entraînement terminé!")

if __name__ == "__main__":
    main() 