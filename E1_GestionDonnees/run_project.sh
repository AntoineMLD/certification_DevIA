#!/bin/bash

# Obtenir le chemin absolu du répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"

# Affichage des messages avec des couleurs simples
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration du logging
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$LOG_DIR/run_${timestamp}.log"
mkdir -p "$LOG_DIR"

# Fonction pour logger les messages
log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$log_file"
}

# Fonction pour logger les erreurs
log_error() {
    local message="$1"
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $message${NC}" | tee -a "$log_file"
}

# Fonction pour logger les succès
log_success() {
    local message="$1"
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $message${NC}" | tee -a "$log_file"
}

log_message "=== Démarrage du projet ==="

# 1. Vérification de Python
if ! command -v python &> /dev/null; then
    log_error "Python n'est pas installé"
    exit 1
fi

# 2. Installation des dépendances
log_message "Installation des dépendances..."
pip install -r requirements.txt >> "$log_file" 2>&1
if [ $? -ne 0 ]; then
    log_error "Installation des dépendances échouée"
    exit 1
fi

# 3. Création des dossiers nécessaires
log_message "Création des dossiers..."
mkdir -p "$LOG_DIR"
mkdir -p "$SCRIPT_DIR/backups"

# 4. Exécution des scripts dans l'ordre
log_message "=== Lancement des scripts ==="

# Sauvegarder le répertoire initial
INITIAL_DIR="$PWD"

# Scraping des données
log_message "1. Scraping des données..."
cd "$SCRIPT_DIR/france_optique"
python run_spiders.py >> "$log_file" 2>&1
if [ $? -ne 0 ]; then
    log_error "Scraping échoué"
    cd "$INITIAL_DIR"
    exit 1
else
    log_success "Scraping terminé avec succès"
fi

# Nettoyage des données
log_message "2. Nettoyage des données..."
cd "$SCRIPT_DIR/Base_de_donnees"
python data_cleaning.py >> "$log_file" 2>&1
if [ $? -ne 0 ]; then
    log_error "Nettoyage échoué"
    cd "$INITIAL_DIR"
    exit 1
else
    log_success "Nettoyage terminé avec succès"
fi

# Migration de la base de données
log_message "3. Migration de la base de données..."
bash migrate_db.sh >> "$log_file" 2>&1
if [ $? -ne 0 ]; then
    log_error "Migration échouée"
    cd "$INITIAL_DIR"
    exit 1
else
    log_success "Migration terminée avec succès"
fi

# Téléchargement des images
log_message "4. Téléchargement des images..."
python download_images.py >> "$log_file" 2>&1
if [ $? -ne 0 ]; then
    log_error "Téléchargement des images échoué"
    cd "$INITIAL_DIR"
    exit 1
else
    log_success "Téléchargement des images terminé avec succès"
fi

# Retour au répertoire initial
cd "$INITIAL_DIR"

# 5. Sauvegarde des résultats
log_message "Sauvegarde des résultats..."
backup_dir="$SCRIPT_DIR/backups/$timestamp"
mkdir -p "$backup_dir"

# Copie des fichiers
cp "$SCRIPT_DIR/Base_de_donnees/france_optique.db" "$backup_dir/" 2>> "$log_file"
if [ -d "$SCRIPT_DIR/Base_de_donnees/images" ]; then
    cp -r "$SCRIPT_DIR/Base_de_donnees/images" "$backup_dir/" 2>> "$log_file"
fi

# Copie du fichier de log dans la sauvegarde
cp "$log_file" "$backup_dir/"

log_success "=== Projet terminé avec succès ! ==="
log_message "Les logs sont disponibles dans: $log_file" 