#!/bin/bash

# Activer l'environnement virtuel si nécessaire
if [ -d "../venv" ]; then
    source ../venv/Scripts/activate
fi

# Exécuter le script Python d'import des tags
python import_tags.py

# Désactiver l'environnement virtuel si nécessaire
if [ -d "../venv" ]; then
    deactivate
fi 