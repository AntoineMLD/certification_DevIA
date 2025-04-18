import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# Ajout du répertoire parent au chemin Python
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

# Imports modifiés pour fonctionner depuis le répertoire models
from efficientnet_triplet import EfficientNetEmbedding
from losses.triplet_losses import HardTripletLoss
from datasets.triplet_dataset import TripletDataset, default_transform

# configuration globale 
DATA_DIR = os.path.join(main_dir, "data", "oversampled_gravures")
SAVE_PATH = os.path.join(main_dir, "models", "efficientnet_triplet.pth")
EMBEDDING_DIM = 256
MARGIN = 0.3
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Chargement du dataset Triplet
dataset = TripletDataset(root_dir=DATA_DIR, transform=default_transform)


# Création du DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f" Dataset chargé : {len(dataset)} triplets disponibles")
    

# initialisation du modèle
model = EfficientNetEmbedding(embedding_dim=EMBEDDING_DIM, pretrained=True)
model = model.to(DEVICE)
print(f" Modèle EfficientNet prêt sur {DEVICE}")


#  Définition de la fonction de perte
criterion = HardTripletLoss(margin=MARGIN, mining_type="semi-hard")


# optimiser le modèle
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Boucle d'entraînement
model.train()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for anchor, positive, negative in progress_bar:
        anchor = anchor.to(DEVICE)
        positive = positive.to(DEVICE)
        negative = negative.to(DEVICE)

        # 1. Forward
        anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)

        # 2. Calcul de la loss
        loss = criterion(anchor_emb, pos_emb, neg_emb)

        # 3. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. Stat tracking
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} terminée - Loss moyenne : {epoch_loss / len(dataloader):.4f}")


# Sauvegarder le modèle
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Modèle sauvegardé dans : {SAVE_PATH}")


