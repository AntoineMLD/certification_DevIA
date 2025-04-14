#!/bin/bash

echo "Copie des dessins crees avec Streamlit vers le projet principal..."

if [ ! -d "drawings" ]; then
    echo "Aucun dessin trouve. Veuillez d'abord creer des dessins avec l'application Streamlit."
    read -p "Appuyez sur Entrée pour continuer..."
    exit 1
fi

# Créer le répertoire de destination s'il n'existe pas
if [ ! -d "../E3_MettreDispositionIA/data/raw_gravures" ]; then
    mkdir -p "../E3_MettreDispositionIA/data/raw_gravures"
fi

# Copier les fichiers de symboles
if [ -d "drawings/symbole" ]; then
    if [ ! -d "../E3_MettreDispositionIA/data/raw_gravures/symbole" ]; then
        mkdir -p "../E3_MettreDispositionIA/data/raw_gravures/symbole"
    fi
    cp -f drawings/symbole/*.png "../E3_MettreDispositionIA/data/raw_gravures/symbole/" 2>/dev/null || true
    echo "Dessins de symboles copies."
fi

echo ""
echo "Les dessins ont ete copies vers le projet principal."
echo ""
echo "Etapes suivantes:"
echo "1. Pretraiter les images: python -m app.prepare_data --input_dir data/raw_gravures --output_dir data/processed"
echo "2. Entrainer le modele: python -m app.train --data_dir data/processed --output_dir model --num_epochs 10 --batch_size 16"
echo "3. Generer les embeddings: python -m app.generate_embeddings --images_dir data/processed --output_path embeddings/gravures_embeddings.pkl --model_path model/siamese_model.pt"
echo ""

read -p "Appuyez sur Entrée pour continuer..." 