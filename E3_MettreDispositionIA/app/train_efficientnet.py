import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import seaborn as sns
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import shutil

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer notre modèle
from app.efficientnet_model import (
    EfficientNetEmbedding, 
    TripletLoss, 
    HardTripletLoss, 
    TripletDataset, 
    save_model
)

# Constantes par défaut
IMAGE_SIZE = 224  # EfficientNet attend des images 224x224
EMBEDDING_DIM = 256
BATCH_SIZE = 64   # Taille de batch plus grande pour une meilleure convergence
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001

class BatchSampler:
    """
    Sampler personnalisé pour créer des batchs avec plusieurs exemples par classe
    """
    def __init__(self, dataset, n_classes, n_samples):
        """
        Args:
            dataset: Le dataset à échantillonner
            n_classes: Nombre de classes par batch
            n_samples: Nombre d'échantillons par classe
        """
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Créer un dictionnaire class -> [indices]
        self.class_indices = {}
        
        # Vérifier si c'est un Subset (après random_split) ou le dataset original
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            # C'est un Subset
            original_dataset = dataset.dataset
            indices = dataset.indices
            
            for subset_idx, idx in enumerate(indices):
                _, label = original_dataset.samples[idx]
                if label not in self.class_indices:
                    self.class_indices[label] = []
                self.class_indices[label].append(subset_idx)
        else:
            # C'est le dataset original
            for idx, (_, label) in enumerate(dataset.samples):
                if label not in self.class_indices:
                    self.class_indices[label] = []
                self.class_indices[label].append(idx)
    
    def __iter__(self):
        """
        Retourne un itérateur sur les indices des échantillons
        """
        # Liste des classes disponibles
        available_classes = list(self.class_indices.keys())
        
        # Tant qu'il y a assez de classes disponibles
        while len(available_classes) >= self.n_classes:
            batch_indices = []
            
            # Choisir n_classes classes aléatoirement
            selected_classes = random.sample(available_classes, self.n_classes)
            
            # Pour chaque classe sélectionnée
            for cls in selected_classes:
                # Prendre n_samples échantillons de cette classe
                if len(self.class_indices[cls]) >= self.n_samples:
                    samples = random.sample(self.class_indices[cls], self.n_samples)
                    batch_indices.extend(samples)
                else:
                    # Si pas assez d'échantillons, prendre ce qu'il y a avec remplacement
                    samples = random.choices(self.class_indices[cls], k=self.n_samples)
                    batch_indices.extend(samples)
            
            # Mélanger les indices dans le batch
            random.shuffle(batch_indices)
            
            yield batch_indices
            
            # Mettre à jour les classes disponibles (pour varier les batchs)
            random.shuffle(available_classes)
    
    def __len__(self):
        """
        Nombre approximatif de batchs par époque
        """
        # Estimation basée sur le nombre de classes et d'échantillons
        num_classes = len(self.class_indices)
        return num_classes // self.n_classes

def visualize_embeddings(model, val_loader, device, save_path=None):
    """
    Visualise les embeddings du modèle sur un jeu de données de validation
    en utilisant t-SNE pour la réduction de dimension
    """
    model.eval()
    embeddings = []
    labels = []
    classes = []
    
    # Collecter les embeddings et les labels
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Génération des embeddings"):
            images = images.to(device)
            outputs = model.forward_one(images)
            embeddings.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    # Convertir en arrays numpy
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    # Réduction de dimension avec t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Visualiser avec Matplotlib
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
              c=labels, cmap='viridis', alpha=0.7)
    
    # Ajouter une légende
    unique_labels = np.unique(labels)
    handles, _ = scatter.legend_elements()
    
    # Récupérer les noms des classes
    try:
        class_names = val_loader.dataset.dataset.classes
        if len(class_names) < len(unique_labels):
            class_names = [f"Classe {i}" for i in unique_labels]
    except:
        class_names = [f"Classe {i}" for i in unique_labels]
    
    plt.legend(handles, [class_names[i] for i in unique_labels], 
              title="Classes", loc="upper right")
    
    plt.title("Visualisation t-SNE des embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def save_incorrect_predictions(model, val_dataloader, class_names, device, output_dir):
    """
    Sauvegarde les images incorrectement classifiées pour analyse
    
    Args:
        model: Le modèle entraîné
        val_dataloader: DataLoader du jeu de validation
        class_names: Liste des noms de classes
        device: Device pour le modèle (cuda/cpu)
        output_dir: Dossier de sortie pour les images
    """
    model.eval()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Créer un dossier pour chaque classe prédite
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    with torch.no_grad():
        incorrect_count = 0
        try:
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                images, labels = images.to(device), labels.to(device)
                
                # Pour utiliser forward_one au lieu de forward (qui attend 3 arguments)
                embeddings = model.forward_one(images)
                
                # Classifier les embeddings avec k-NN
                # (Simplification pour démonstration)
                knn = KNeighborsClassifier(n_neighbors=5)
                
                # Collecter des embeddings pour le training de k-NN
                all_embeddings = embeddings.cpu().numpy()
                all_labels = labels.cpu().numpy()
                
                # Si batch est trop petit, ignorer
                if len(all_labels) < 2:
                    continue
                    
                # Entraîner k-NN sur une partie des données
                train_size = max(1, len(all_labels) // 2)
                indices = np.random.permutation(len(all_labels))
                train_indices = indices[:train_size]
                test_indices = indices[train_size:]
                
                if len(np.unique(all_labels[train_indices])) < 2:
                    continue  # Pas assez de classes pour entraîner
                
                X_train = all_embeddings[train_indices]
                y_train = all_labels[train_indices]
                
                knn.fit(X_train, y_train)
                
                # Prédire sur les données restantes
                X_test = all_embeddings[test_indices]
                y_test = all_labels[test_indices]
                
                preds = knn.predict(X_test)
                
                # Identifier les prédictions incorrectes
                for i, (idx, true_label, pred_label) in enumerate(zip(test_indices, y_test, preds)):
                    if pred_label != true_label:
                        # Récupérer l'image d'origine
                        img = images[idx].cpu()
                        
                        # Convertir le tenseur en image PIL
                        img = img * 0.5 + 0.5  # Dénormaliser
                        img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
                        
                        # Vérifier si c'est une image en niveaux de gris (1 canal)
                        if img.shape[2] == 1:
                            img = img.squeeze(2)  # Supprimer le canal unique
                        
                        img = (img * 255).astype(np.uint8)
                        
                        # Sauvegarder l'image
                        true_class_name = class_names[true_label]
                        pred_class_name = class_names[pred_label]
                        
                        image_path = os.path.join(
                            output_dir, 
                            pred_class_name, 
                            f"true_{true_class_name}_pred_{pred_class_name}_{incorrect_count}.png"
                        )
                        
                        # Créer l'image PIL et la sauvegarder
                        if len(img.shape) == 2:
                            # Image en niveaux de gris
                            img_pil = Image.fromarray(img, mode='L')
                        else:
                            # Image RGB
                            img_pil = Image.fromarray(img)
                            
                        img_pil.save(image_path)
                        incorrect_count += 1
                        
        except Exception as e:
            print(f"Erreur pendant l'analyse des prédictions incorrectes: {e}")
    
    print(f"Sauvegarde de {incorrect_count} prédictions incorrectes dans {output_dir}")
    return incorrect_count

def train_with_triplet_miner(train_loader, val_loader, model, criterion, optimizer, 
                            num_epochs=10, device='cpu', scheduler=None, 
                            save_dir="model", save_prefix="efficientnet_triplet",
                            progressive_unfreeze=False, initial_freeze=0.8):
    """
    Entraîne le modèle EfficientNet avec hard triplet mining
    
    Args:
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        model: Le modèle à entraîner
        criterion: Fonction de perte (HardTripletLoss)
        optimizer: Optimiseur
        num_epochs: Nombre d'époques d'entraînement
        device: Device sur lequel effectuer l'entraînement
        scheduler: Scheduler pour l'ajustement du taux d'apprentissage
        save_dir: Dossier où sauvegarder le modèle
        save_prefix: Préfixe pour les fichiers de modèle sauvegardés
        progressive_unfreeze: Dégeler progressivement les couches du backbone
        initial_freeze: Proportion initiale des couches à geler
        
    Returns:
        Le modèle entraîné et les historiques de perte
    """
    # S'assurer que le dossier existe
    os.makedirs(save_dir, exist_ok=True)
    
    # Historiques
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_top3_accuracies = []
    val_top3_accuracies = []
    best_val_acc = 0.0
    
    # Si dégel progressif activé, geler toutes les couches au début
    if progressive_unfreeze:
        freeze_backbone(model, initial_freeze)
    
    # Barre de progression principale
    progress_bar = tqdm(range(num_epochs), desc="Entraînement")
    
    for epoch in range(num_epochs):
        # Dégeler progressivement les couches du backbone
        if progressive_unfreeze:
            freeze_ratio = max(0, initial_freeze * (1 - epoch / (num_epochs * 0.7)))
            freeze_backbone(model, freeze_ratio)
            
            # Mettre à jour l'optimiseur pour les paramètres qui nécessitent un gradient
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=optimizer.param_groups[0]['lr'],
                weight_decay=1e-4
            )
        
        # Mode entraînement
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Pour calculer la précision pendant l'entraînement
        all_embeddings = []
        all_labels = []
        
        # Boucle d'entraînement
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, targets in batch_progress:
            # Déplacer les données sur le bon device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Forward pass - extraire les embeddings
            embeddings = model.forward_one(inputs)
            
            # Calculer la loss avec hard triplet mining
            loss = criterion(embeddings, targets)
            
            # Backward pass et optimisation
            loss.backward()
            optimizer.step()
            
            # Statistiques
            train_loss += loss.item()
            batch_count += 1
            
            # Collecter les embeddings et labels pour calculer la précision
            all_embeddings.append(embeddings.detach().cpu().numpy())
            all_labels.append(targets.detach().cpu().numpy())
            
            # Mettre à jour la progression
            batch_progress.set_postfix({
                'loss': loss.item()
            })
        
        # Calculer la perte moyenne par batch (et non par échantillon)
        train_loss = train_loss / batch_count
        
        # Calculer la précision top-1 et top-3 sur l'ensemble d'entraînement
        train_acc, train_top3_acc = compute_accuracy(all_embeddings, all_labels)
        
        # Mode évaluation
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        # Pour calculer la précision sur la validation
        val_embeddings = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                embeddings = model.forward_one(inputs)
                loss = criterion(embeddings, targets)
                val_loss += loss.item()
                val_batch_count += 1
                
                # Collecter pour calculer la précision
                val_embeddings.append(embeddings.cpu().numpy())
                val_labels.append(targets.cpu().numpy())
        
        val_loss = val_loss / val_batch_count
        
        # Calculer la précision top-1 et top-3 sur l'ensemble de validation
        val_acc, val_top3_acc = compute_accuracy(val_embeddings, val_labels)
        
        # Ajuster le taux d'apprentissage si nécessaire
        if scheduler is not None:
            scheduler.step()
        
        # Enregistrer les statistiques
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_top3_accuracies.append(train_top3_acc)
        val_top3_accuracies.append(val_top3_acc)
        
        # Mettre à jour la barre de progression
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'val_acc': f'{val_acc:.2f}%',
            'val_top3': f'{val_top3_acc:.2f}%'
        })
        progress_bar.update(1)
        
        # Informations détaillées
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
              f"Acc: {train_acc:.2f}%/{val_acc:.2f}% - "
              f"Top3: {train_top3_acc:.2f}%/{val_top3_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle selon la précision
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f"best_{save_prefix}.pt")
            save_model(model, save_path)
            print(f"\nMeilleur modèle sauvegardé avec précision de validation: {val_acc:.2f}%")
        
        # Sauvegarder le modèle à chaque 10 époques
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"{save_prefix}_epoch{epoch+1}.pt")
            save_model(model, save_path)
    
    # Sauvegarder le modèle final
    save_path = os.path.join(save_dir, f"final_{save_prefix}.pt")
    save_model(model, save_path)
    
    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(15, 5))
    
    # Courbe de perte
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Entraînement')
    plt.plot(val_losses, label='Validation')
    plt.title('Perte pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    # Courbe de précision top-1
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Entraînement')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Précision Top-1 pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Précision (%)')
    plt.legend()
    
    # Courbe de précision top-3
    plt.subplot(1, 3, 3)
    plt.plot(train_top3_accuracies, label='Entraînement')
    plt.plot(val_top3_accuracies, label='Validation')
    plt.title('Précision Top-3 pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Précision (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_learning_curves.png"))
    
    # Sauvegarder les prédictions incorrectes pour analyse
    if hasattr(train_loader.dataset, 'dataset'):
        if hasattr(train_loader.dataset.dataset, 'classes'):
            class_names = train_loader.dataset.dataset.classes
            incorrect_dir = os.path.join(save_dir, 'incorrect_predictions')
            evaluation_loader = DataLoader(val_loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
            try:
                incorrect_count = save_incorrect_predictions(model, evaluation_loader, class_names, device, incorrect_dir)
                print(f"Analyse des erreurs terminée : {incorrect_count} prédictions incorrectes identifiées")
            except Exception as e:
                print(f"Impossible de sauvegarder les prédictions incorrectes: {e}")
        else:
            print("Impossible d'accéder aux classes du dataset")
    else:
        print("Impossible d'analyser les prédictions incorrectes: structure de dataset incompatible")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def compute_accuracy(embeddings_list, labels_list):
    """
    Calcule la précision top-1 et top-3 en utilisant une approche k-NN
    """
    # Concaténer tous les embeddings et labels
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    # Division train/test pour le k-NN
    num_samples = len(embeddings)
    if num_samples < 10:
        return 0.0, 0.0  # Pas assez d'échantillons
    
    # Utiliser 80% pour l'entraînement
    train_size = int(0.8 * num_samples)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, y_train = embeddings[train_indices], labels[train_indices]
    X_test, y_test = embeddings[test_indices], labels[test_indices]
    
    if len(test_indices) == 0 or len(np.unique(y_train)) < 2:
        return 0.0, 0.0  # Pas assez d'échantillons
    
    # KNN pour la précision top-1
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    
    # Précision top-1
    y_pred = knn.predict(X_test)
    top1_acc = accuracy_score(y_test, y_pred) * 100.0
    
    # Précision top-3 (si possible)
    try:
        # Obtenir les probabilités de toutes les classes
        proba = knn.predict_proba(X_test)
        
        # Top-3 accuracy
        top3_correct = 0
        for i, true_label in enumerate(y_test):
            # Indices des 3 classes les plus probables
            top3_classes = np.argsort(proba[i])[::-1][:3]
            # Convertir les indices en vraies classes
            top3_class_labels = knn.classes_[top3_classes]
            if true_label in top3_class_labels:
                top3_correct += 1
                
        top3_acc = (top3_correct / len(y_test)) * 100.0
    except:
        # Fallback si la prédiction de probabilité ne fonctionne pas
        top3_acc = top1_acc
    
    return top1_acc, top3_acc

def analyze_class_distribution(dataset):
    """
    Analyse la distribution des classes dans le dataset
    
    Args:
        dataset: Le dataset à analyser (TripletDataset)
        
    Returns:
        counts: Dictionnaire avec le nombre d'échantillons par classe
        stats: Statistiques sur la distribution (min, max, moyenne, médiane)
    """
    # Collecter les classes de tous les échantillons
    class_labels = [label for _, label in dataset.samples]
    
    # Compter les occurrences de chaque classe
    counts = Counter(class_labels)
    
    # Calculer les statistiques
    values = list(counts.values())
    stats = {
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
        'median': sorted(values)[len(values) // 2],
        'total': sum(values),
        'num_classes': len(counts)
    }
    
    return counts, stats

def create_weighted_sampler(dataset):
    """
    Crée un WeightedRandomSampler pour équilibrer les classes
    
    Args:
        dataset: Dataset avec les échantillons
        
    Returns:
        sampler: WeightedRandomSampler pour équilibrer les classes
    """
    # Gérer le cas d'un Subset (après random_split)
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        original_dataset = dataset.dataset
        indices = dataset.indices
        samples = [original_dataset.samples[i] for i in indices]
    else:
        samples = dataset.samples
    
    # Collecter les classes
    class_labels = [label for _, label in samples]
    
    # Compter les occurrences de chaque classe
    counts = Counter(class_labels)
    
    # Calculer les poids pour chaque échantillon
    weights = []
    for _, label in samples:
        # Poids = 1 / fréquence de la classe
        class_weight = 1.0 / counts[label]
        weights.append(class_weight)
    
    # Convertir en tensor
    weights = torch.DoubleTensor(weights)
    
    # Créer le sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, 
        num_samples=len(weights), 
        replacement=True
    )
    
    return sampler

def freeze_backbone(model, freeze_ratio=1.0):
    """
    Gèle une partie du backbone EfficientNet
    
    Args:
        model: Modèle EfficientNet
        freeze_ratio: Proportion des couches à geler (0.0 = aucune, 1.0 = toutes)
    """
    # Récupérer toutes les couches du modèle
    backbone_layers = list(model.backbone.named_children())
    num_layers = len(backbone_layers)
    
    # Calculer le nombre de couches à geler
    num_freeze = int(num_layers * freeze_ratio)
    
    # Geler les couches
    for i, (name, layer) in enumerate(backbone_layers):
        if i < num_freeze:
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Couche {name} gelée")
        else:
            # S'assurer que les autres couches sont bien dégelées
            for param in layer.parameters():
                param.requires_grad = True
    
    # Toujours dégeler la tête d'embedding
    for param in model.embedding_head.parameters():
        param.requires_grad = True
        
    # Compter les paramètres gelés vs dégelés
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Paramètres gelés: {frozen_params/1000:.1f}K / {total_params/1000:.1f}K ({frozen_params/total_params*100:.1f}%)")

def progressive_unfreeze_layers(model, epoch, total_epochs, initial_freeze=1.0):
    """
    Dégèle progressivement les couches du backbone au fil des époques
    
    Args:
        model: Modèle EfficientNet
        epoch: Époque actuelle
        total_epochs: Nombre total d'époques
        initial_freeze: Proportion initiale des couches gelées
    """
    # Calculer la proportion à geler en fonction de l'époque
    # Plus l'époque avance, moins on gèle
    freeze_ratio = max(0, initial_freeze * (1 - epoch / (total_epochs * 0.7)))
    
    # Appliquer le gel
    freeze_backbone(model, freeze_ratio)
    
    return freeze_ratio

def main():
    # Configuration avec hyperparamètres ajustés
    parser = argparse.ArgumentParser(description='Entraînement du modèle EfficientNet avec Triplet Loss')
    parser.add_argument('--data_dir', type=str, default='data/processed', 
                      help='Dossier contenant les données')
    parser.add_argument('--batch_size', type=int, default=16, 
                      help='Taille du batch')
    parser.add_argument('--epochs', type=int, default=30, 
                      help='Nombre d\'époques')
    parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, 
                      help='Dimension de l\'embedding')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, 
                      help='Taux d\'apprentissage')
    parser.add_argument('--save_dir', type=str, default='model', 
                      help='Dossier où sauvegarder le modèle')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Utiliser les poids préentraînés sur ImageNet')
    parser.add_argument('--balance_classes', action='store_true',
                      help='Équilibrer les classes avec un WeightedRandomSampler')
    parser.add_argument('--progressive_unfreeze', action='store_true',
                      help='Dégeler progressivement les couches du backbone')
    parser.add_argument('--initial_freeze', type=float, default=0.8,
                      help='Proportion initiale des couches à geler (0.0 = aucune, 1.0 = toutes)')
    parser.add_argument('--onecycle', action='store_true',
                      help='Utiliser le scheduler OneCycleLR')
    parser.add_argument('--mining_type', type=str, default='semi-hard', choices=['semi-hard', 'hard', 'all'],
                      help='Type de mining pour TripletLoss: semi-hard, hard, ou all')
    
    args = parser.parse_args()
    
    # Définir le device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Transformations de base avec quelques augmentations simples
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),  # Simple mais efficace
        transforms.RandomRotation(15),      # Rotation modérée
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Créer les datasets et dataloaders
    full_dataset = TripletDataset(args.data_dir, transform=train_transform)
    
    # Obtenir le nombre de classes
    num_classes = len(full_dataset.classes)
    print(f"Nombre de classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")
    
    # Analyser la distribution des classes
    class_counts, stats = analyze_class_distribution(full_dataset)
    print("\nStatistiques de distribution des classes:")
    print(f"- Nombre total d'images: {stats['total']}")
    print(f"- Nombre de classes: {stats['num_classes']}")
    print(f"- Min images/classe: {stats['min']}")
    print(f"- Max images/classe: {stats['max']}")
    print(f"- Moyenne images/classe: {stats['mean']:.1f}")
    print(f"- Médiane images/classe: {stats['median']}")

    # Afficher les classes avec peu d'images (< 10)
    low_count_classes = {cls: count for cls, count in class_counts.items() if count < 10}
    if low_count_classes:
        print("\nClasses avec moins de 10 images:")
        for cls, count in low_count_classes.items():
            class_name = full_dataset.classes[cls]
            print(f"- {class_name}: {count} images")
        print("\nConseil: Envisagez d'ajouter plus d'images pour ces classes ou d'utiliser l'oversampling.")

    # Sauvegarder les statistiques dans un fichier
    os.makedirs(args.save_dir, exist_ok=True)
    stats_path = os.path.join(args.save_dir, "class_distribution_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Statistiques de distribution des classes\n")
        f.write(f"Nombre total d'images: {stats['total']}\n")
        f.write(f"Nombre de classes: {stats['num_classes']}\n")
        f.write(f"Min images/classe: {stats['min']}\n")
        f.write(f"Max images/classe: {stats['max']}\n")
        f.write(f"Moyenne images/classe: {stats['mean']:.1f}\n")
        f.write(f"Médiane images/classe: {stats['median']}\n\n")
        
        f.write("Détail par classe:\n")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
            class_name = full_dataset.classes[cls]
            f.write(f"{class_name}: {count} images\n")

    # Créer un graphique de distribution
    plt.figure(figsize=(12, 6))
    class_names = [full_dataset.classes[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    bars = plt.bar(range(len(counts)), counts, tick_label=class_names)
    plt.xticks(rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes dans le dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "class_distribution.png"))
    print(f"\nStatistiques de classes sauvegardées dans {stats_path}")
    print(f"Graphique de distribution sauvegardé dans {os.path.join(args.save_dir, 'class_distribution.png')}")
    
    # Diviser en entraînement et validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Pour avoir des transformations différentes sur val_dataset
    val_dataset.dataset = TripletDataset(args.data_dir, transform=val_transform)
    
    # Créer les dataloaders
    if args.balance_classes:
        print("\nÉquilibrage des classes activé avec WeightedRandomSampler")
        try:
            sampler = create_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size,
                sampler=sampler,  # Utiliser le sampler pondéré au lieu de shuffle
                num_workers=0
            )
            print("Sampler créé avec succès")
        except Exception as e:
            print(f"Erreur lors de la création du sampler: {e}")
            print("Utilisation d'un DataLoader standard")
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"Nombre d'images d'entraînement: {len(train_dataset)}")
    print(f"Nombre d'images de validation: {len(val_dataset)}")
    
    # Créer le modèle
    model = EfficientNetEmbedding(
        embedding_dim=args.embedding_dim, 
        pretrained=args.pretrained
    )
    model.to(device)
    print(f"Modèle créé avec dimension d'embedding: {args.embedding_dim}")
    
    # Définir le loss et l'optimiseur
    criterion = HardTripletLoss(margin=0.3, mining_type=args.mining_type)
    
    print(f"Utilisation de HardTripletLoss avec mining_type: {args.mining_type}")
    
    # Optimiseur avec weight decay pour la régularisation
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4
    )
    
    # Définir le scheduler pour réduire le taux d'apprentissage
    if hasattr(args, 'onecycle') and args.onecycle:
        print("Utilisation du scheduler OneCycleLR")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,  # pic à 10x le taux initial
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.3  # 30% du temps pour la phase de warm-up
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=args.lr / 100
        )
    
    # Entraîner le modèle
    print(f"Début de l'entraînement pour {args.epochs} époques...")
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_with_triplet_miner(
        train_loader, val_loader, model, criterion, optimizer,
        num_epochs=args.epochs, device=device, scheduler=scheduler,
        save_dir=args.save_dir, save_prefix="efficientnet_triplet",
        progressive_unfreeze=args.progressive_unfreeze if hasattr(args, 'progressive_unfreeze') else False, 
        initial_freeze=args.initial_freeze if hasattr(args, 'initial_freeze') else 0.8
    )
    
    # Visualiser les embeddings
    visualize_embeddings(
        model, val_loader, device,
        save_path=os.path.join(args.save_dir, "efficientnet_embeddings_visualization.png")
    )
    
    print("Entraînement terminé!")

if __name__ == "__main__":
    main() 