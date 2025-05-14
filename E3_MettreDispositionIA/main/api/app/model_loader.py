import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Ajouter le répertoire parent au path pour pouvoir importer le module models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.efficientnet_triplet import EfficientNetEmbedding


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_model():
    # Nous utilisons pretrained=False car nous chargeons nos propres poids
    # Le warning ne devrait plus apparaître car nous avons modifié la classe pour utiliser weights=None
    model = EfficientNetEmbedding(embedding_dim=256, pretrained=False)
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../weights/efficientnet_triplet.pth"))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model 

def preprocess_image(img: Image.Image): 
    return transform(img).unsqueeze(0).to(DEVICE)

def get_embedding(model, img: Image.Image):
    tensor = preprocess_image(img)
    with torch.no_grad():
        emb = model.forward_one(tensor).cpu().numpy()
    return emb[0]