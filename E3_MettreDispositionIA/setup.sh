#!/bin/bash

# Script d'installation et configuration du projet E3_MettreDispositionIA
# Ce script configure l'environnement, prépare les données, 
# entraîne le modèle et lance l'application

set -e  # Arrêt en cas d'erreur

echo "========== Configuration du projet E3_MettreDispositionIA =========="

# Vérifier si Python est installé
if ! command -v python &> /dev/null; then
    echo "Python n'est pas installé. Veuillez installer Python 3.8 ou supérieur."
    exit 1
fi

# Vérifier si pip est installé
if ! command -v pip &> /dev/null; then
    echo "pip n'est pas installé. Veuillez installer pip."
    exit 1
fi

# Créer un environnement virtuel si nécessaire
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python -m venv venv
    echo "Environnement virtuel créé avec succès."
fi

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Installer les dépendances
echo "Installation des dépendances..."
pip install -r requirements.txt

# Créer les dossiers nécessaires
echo "Création des dossiers du projet..."
mkdir -p data/processed
mkdir -p model
mkdir -p embeddings

# Vérifier si des images brutes existent
echo "Vérification des données..."
RAW_DATA_DIR="data/raw_gravures"
PROCESSED_DATA_DIR="data/processed"

# Prétraiter les données si le dossier raw_gravures existe et contient des images
PROCESS_DATA=false

if [ -d "$RAW_DATA_DIR" ]; then
    # Compte les fichiers d'images dans les sous-dossiers
    IMAGE_COUNT=$(find "$RAW_DATA_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" \) | wc -l)
    
    if [ $IMAGE_COUNT -gt 0 ]; then
        echo "Détection de $IMAGE_COUNT images à prétraiter."
        PROCESS_DATA=true
    fi
fi

if $PROCESS_DATA; then
    echo "Prétraitement des images..."
    python -m app.prepare_data --data_dir "$RAW_DATA_DIR" --output_dir "$PROCESSED_DATA_DIR"
    echo "Prétraitement terminé."
else
    echo "Aucune image brute détectée dans $RAW_DATA_DIR."
    echo "Pour prétraiter des images, veuillez placer vos images dans des sous-dossiers de $RAW_DATA_DIR."
    echo "Les sous-dossiers doivent représenter les classes (par exemple: cercle, triangle, losange)."
fi

# Vérifier si des données prétraitées existent
PROCESSED_COUNT=$(find "$PROCESSED_DATA_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" \) | wc -l)

if [ $PROCESSED_COUNT -gt 0 ]; then
    echo "Détection de $PROCESSED_COUNT images prétraitées."
    
    # Vérifier si un modèle existe déjà
    if [ ! -f "model/siamese_model.pt" ] && [ ! -f "model/best_siamese_model.pt" ]; then
        echo "Entraînement du modèle..."
        python -m app.train --data_dir "$PROCESSED_DATA_DIR" --output_dir "model" --num_epochs 10
        echo "Entraînement terminé."
    else
        echo "Un modèle existe déjà. Pour réentraîner, supprimez les fichiers du dossier 'model'."
    fi
    
    # Vérifier si les embeddings existent déjà
    if [ ! -f "embeddings/gravures_embeddings.pkl" ]; then
        echo "Génération des embeddings..."
        MODEL_PATH="model/best_siamese_model.pt"
        if [ ! -f "$MODEL_PATH" ]; then
            MODEL_PATH="model/siamese_model.pt"
        fi
        
        python -m admin_utils regenerate --model best
        echo "Génération des embeddings terminée."
    else
        echo "Les embeddings existent déjà. Pour les régénérer, supprimez le fichier 'embeddings/gravures_embeddings.pkl'."
    fi
else
    echo "Aucune image prétraitée détectée dans $PROCESSED_DATA_DIR."
    echo "Veuillez préparer des images avant de continuer."
fi

# Demander à l'utilisateur s'il souhaite lancer l'application
read -p "Voulez-vous lancer l'application ? (O/n) " -n 1 -r REPLY
echo
REPLY=${REPLY:-O}  # Par défaut "O" si aucune réponse

if [[ $REPLY =~ ^[Oo]$ ]]; then
    echo "Lancement de l'application..."
    echo "Interface Tkinter: lancer avec 'python tkinter_draw_app.py'"
    echo "API FastAPI: lancer avec 'uvicorn app.main:app --reload --host 0.0.0.0 --port 8000'"
    
    # Demander quelle interface lancer
    echo "Quelle interface souhaitez-vous lancer ?"
    echo "1. Interface Tkinter (application de dessin)"
    echo "2. API FastAPI (service web)"
    echo "3. Les deux (en parallèle)"
    read -p "Votre choix (1/2/3) : " -n 1 -r INTERFACE_CHOICE
    echo
    
    case $INTERFACE_CHOICE in
        1)
            python tkinter_draw_app.py
            ;;
        2)
            uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
            ;;
        3)
            # Lancer FastAPI en arrière-plan
            uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
            # Attendre que l'API démarre
            sleep 3
            # Lancer Tkinter
            python tkinter_draw_app.py
            # Arrêter l'API quand Tkinter se ferme
            kill $!
            ;;
        *)
            echo "Choix invalide. Aucune interface lancée."
            ;;
    esac
else
    echo "Installation terminée. Pour lancer l'application:"
    echo "- Interface Tkinter: 'python tkinter_draw_app.py'"
    echo "- API FastAPI: 'uvicorn app.main:app --reload --host 0.0.0.0 --port 8000'"
fi

echo "========== Configuration terminée ==========" 