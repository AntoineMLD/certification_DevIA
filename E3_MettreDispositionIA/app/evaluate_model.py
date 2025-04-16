import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modèles
from app.efficientnet_model import load_model as load_efficientnet_model, TripletDataset

def evaluate_embeddings_with_knn(model, val_loader, device, k=5):
    """
    Évalue le modèle en utilisant k-NN sur les embeddings générés
    
    Args:
        model: Le modèle à évaluer
        val_loader: DataLoader de validation
        device: Device utilisé (cpu/cuda)
        k: Nombre de voisins pour k-NN
        
    Returns:
        accuracy: Précision du modèle
        y_true: Vraies étiquettes
        y_pred: Prédictions
    """
    model.eval()
    embeddings = []
    labels = []
    
    # Extraire les embeddings
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="Extraction des embeddings")):
            inputs = inputs.to(device)
            
            # Extraire les embeddings selon le type de modèle
            if hasattr(model, 'get_embedding'):
                # Pour les modèles avec méthode get_embedding
                batch_embeddings = model.get_embedding(inputs).cpu().numpy()
            else:
                # Modèle EfficientNet ou Siamese
                batch_embeddings = model.forward_one(inputs).cpu().numpy()
                
            embeddings.append(batch_embeddings)
            labels.append(targets.numpy())
    
    # Concaténer les embeddings et les labels
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    # Diviser les données en entraînement et test pour k-NN
    train_size = int(0.6 * len(embeddings))
    indices = np.random.permutation(len(embeddings))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, y_train = embeddings[train_indices], labels[train_indices]
    X_test, y_test = embeddings[test_indices], labels[test_indices]
    
    # Appliquer k-NN
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X_train, y_train)
    
    # Prédire et évaluer
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_test, y_pred, knn, (X_train, y_train, X_test, y_test)

def plot_confusion_matrix(y_true, y_pred, class_names, output_path=None):
    """
    Trace la matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    # Normaliser la matrice de confusion
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Tracer avec seaborn
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de confusion normalisée')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Évaluation de modèle avec k-NN')
    parser.add_argument('--model_path', type=str, required=True, 
                      help='Chemin vers le modèle entraîné')
    parser.add_argument('--model_type', type=str, choices=['efficientnet', 'siamese'], 
                      default='efficientnet', help='Type de modèle à évaluer')
    parser.add_argument('--data_dir', type=str, default='data/processed', 
                      help='Dossier contenant les données')
    parser.add_argument('--embedding_dim', type=int, default=256, 
                      help='Dimension de l\'embedding')
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='Taille du batch')
    parser.add_argument('--k', type=int, default=5, 
                      help='Nombre de voisins pour k-NN')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                      help='Dossier pour sauvegarder les résultats')
    
    args = parser.parse_args()
    
    # Définir le device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet utilise 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Charger le dataset approprié
    dataset = TripletDataset(args.data_dir, transform=transform)
    
    # Obtenir les noms des classes
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Nombre de classes: {num_classes}")
    
    # Créer le DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Charger le modèle
    if args.model_type == 'efficientnet':
        model = load_efficientnet_model(args.model_path, device=device, embedding_dim=args.embedding_dim)
    else:  # Siamese
        from app.model import load_model
        model = load_model(args.model_path, device=device, embedding_dim=args.embedding_dim)
    
    # Évaluer le modèle
    print(f"Évaluation du modèle {args.model_type} avec k-NN (k={args.k})...")
    accuracy, y_true, y_pred, knn, data = evaluate_embeddings_with_knn(
        model, dataloader, device, k=args.k
    )
    
    # Afficher les résultats
    print(f"Précision du modèle: {accuracy * 100:.2f}%")
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Tracer la matrice de confusion
    confusion_matrix_path = os.path.join(args.output_dir, f"{args.model_type}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, confusion_matrix_path)
    print(f"Matrice de confusion sauvegardée dans {confusion_matrix_path}")
    
    # Sauvegarder le classifier k-NN
    import joblib
    knn_path = os.path.join(args.output_dir, f"{args.model_type}_knn_classifier.joblib")
    joblib.dump(knn, knn_path)
    print(f"Classifier k-NN sauvegardé dans {knn_path}")
    
    # Créer un rapport
    report_path = os.path.join(args.output_dir, f"{args.model_type}_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Évaluation du modèle {args.model_type}\n")
        f.write(f"Modèle: {args.model_path}\n")
        f.write(f"Dimension d'embedding: {args.embedding_dim}\n")
        f.write(f"Nombre de classes: {num_classes}\n")
        f.write(f"k-NN avec k={args.k}\n\n")
        f.write(f"Précision: {accuracy * 100:.2f}%\n\n")
        f.write("Rapport de classification:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"Rapport d'évaluation sauvegardé dans {report_path}")

if __name__ == "__main__":
    main() 