#!/bin/bash

# Définition des couleurs pour les logs
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Fonction de logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Vérification de l'existence du fichier SQL
if [ ! -f "create_tables.sql" ]; then
    error "Le fichier create_tables.sql n'existe pas"
    exit 1
fi

# Vérification de l'existence de la base de données
if [ ! -f "france_optique.db" ]; then
    error "La base de données france_optique.db n'existe pas"
    exit 1
fi

# Création d'une sauvegarde avant migration
log "Création d'une sauvegarde de la base de données..."
cp france_optique.db france_optique.db.backup
if [ $? -eq 0 ]; then
    success "Sauvegarde créée avec succès"
else
    error "Échec de la création de la sauvegarde"
    exit 1
fi

# Exécution du script SQL
log "Début de la migration de la base de données..."
OUTPUT=$(sqlite3 france_optique.db < create_tables.sql 2>&1)

# Vérification du résultat
if [ $? -eq 0 ]; then
    success "Migration terminée avec succès"
    
    # Affichage des statistiques
    log "Récupération des statistiques..."
    echo -e "\nStatistiques des tables :"
    echo "------------------------"
    
    for table in "traitements" "fournisseurs" "materiaux" "gammes" "series" "verres" "verres_traitements"
    do
        COUNT=$(sqlite3 france_optique.db "SELECT COUNT(*) FROM $table;")
        echo -e "${BLUE}$table:${NC} $COUNT enregistrements"
    done
else
    error "Erreur lors de la migration"
    error "Message d'erreur : $OUTPUT"
    
    log "Restauration de la sauvegarde..."
    cp france_optique.db.backup france_optique.db
    if [ $? -eq 0 ]; then
        success "Base de données restaurée avec succès"
    else
        error "Échec de la restauration de la base de données"
    fi
    exit 1
fi 