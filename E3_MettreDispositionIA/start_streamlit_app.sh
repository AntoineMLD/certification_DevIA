#!/bin/bash

# Script pour lancer l'application Streamlit
# Ce script est compatible avec Linux, macOS et Windows (avec Git Bash ou WSL)

echo "Démarrage de l'application Streamlit..."

# Chemin de ce script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activer l'environnement virtuel si disponible
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activation de l'environnement virtuel..."
    source "$SCRIPT_DIR/venv/bin/activate" 2>/dev/null || source "$SCRIPT_DIR/venv/Scripts/activate" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Impossible d'activer l'environnement virtuel. Utilisation de Python système."
    else
        echo "Environnement virtuel activé."
    fi
fi

# Lancer l'application Streamlit
cd "$SCRIPT_DIR"
echo "Lancement de l'application depuis $SCRIPT_DIR..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 