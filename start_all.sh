#!/bin/bash

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages avec timestamp
log_message() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"
}

# Fonction pour afficher les erreurs
error_message() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERREUR: $1${NC}"
}

# Fonction pour arrêter les services existants
stop_services() {
    log_message "Arrêt des services existants..."
    taskkill //F //IM python.exe //T > /dev/null 2>&1 || true
    sleep 2
}

# Fonction pour convertir les chemins en format Windows
convert_path() {
    echo "$1" | sed 's/\//\\/g'
}

# Vérification des dossiers
if [ ! -d "E1_GestionDonnees" ] || [ ! -d "E3_MettreDispositionIA" ]; then
    error_message "Les dossiers E1_GestionDonnees et E3_MettreDispositionIA doivent exister!"
    exit 1
fi

# Fonction pour démarrer un service
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local working_dir=$4
    local extra_pythonpath=$5
    
    log_message "Démarrage de $name sur le port $port..."
    
    # Convertir le chemin en format Windows
    local win_dir=$(convert_path "$working_dir")
    
    # Vérifier si le répertoire existe
    if [ ! -d "$working_dir" ]; then
        error_message "Le répertoire $working_dir n'existe pas!"
        return 1
    fi
    
    # Changement de répertoire et exécution
    cd "$working_dir" || {
        error_message "Impossible d'accéder au répertoire $working_dir"
        return 1
    }
    
    # Ajout du PYTHONPATH si nécessaire
    if [ ! -z "$extra_pythonpath" ]; then
        local win_pythonpath=$(convert_path "$extra_pythonpath")
        export PYTHONPATH="$win_pythonpath;$PYTHONPATH"
    fi
    
    # Exécution de la commande
    $command &
    
    # Retour au répertoire précédent
    cd - > /dev/null
    
    sleep 3  # Attente pour laisser le temps au service de démarrer
}

# Arrêt des services existants
stop_services

# Création des répertoires de logs si nécessaires
mkdir -p logs

# Obtenir le chemin absolu du répertoire de travail
WORKSPACE_DIR=$(pwd)

# Démarrage de l'API Base de données (E1)
start_service "API Base de données" "python -m uvicorn api.app.main:app --reload --port 8001" 8001 "E1_GestionDonnees"

# Démarrage de l'API IA (E3)
E3_DIR="$WORKSPACE_DIR/E3_MettreDispositionIA/main"
start_service "API IA" "python -m uvicorn api.app.main:app --reload --port 8000" 8000 "$E3_DIR" "$E3_DIR"

# Démarrage de l'interface Streamlit principale (E3)
start_service "Interface Streamlit" "python -m streamlit run app/app.py --server.port 8501" 8501 "$E3_DIR"

# Retour au répertoire initial
cd "$WORKSPACE_DIR"

# Affichage des URLs des services
echo -e "\n${BLUE}Services disponibles :${NC}"
echo -e "${BLUE}API Base de données : ${NC}http://localhost:8001"
echo -e "${BLUE}API IA : ${NC}http://localhost:8000"
echo -e "${BLUE}Interface principale : ${NC}http://localhost:8501"

# Attente de l'arrêt de tous les processus
wait

# Message de fin
log_message "Tous les services ont été arrêtés." 