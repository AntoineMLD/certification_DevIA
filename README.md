# Système de Gestion des Gravures de Verres Optiques

## 📋 Description du Projet
Ce projet a été développé dans le cadre d'une certification de Développeur en Intelligence Artificielle. Il répond à un besoin concret d'un opticien qui souhaite moderniser son processus de recherche de gravures de verres.

### 🎯 Objectif
Remplacer la méthode manuelle de recherche de gravures de verres par une solution numérique automatisée, permettant ainsi :
- Un gain de temps significatif
- Une réduction des erreurs
- Une meilleure traçabilité
- Un accès rapide aux informations des verres

## 🔄 Flux de Traitement des Données

```mermaid
graph TD
    A[Scraping des Sites Web] -->|Données Brutes| B[Table Staging]
    B -->|Nettoyage & Enrichissement| C[Table Enhanced]
    C -->|Migration| D[Tables Normalisées]
    C -->|URLs d'Images| E[Téléchargement Images]
    E -->|Amélioration Images| F[Stockage Local]
    F -->|Mise à jour Chemins| C
```

## 📊 Structure de la Base de Données

### Table Staging (Données Brutes)
```mermaid
erDiagram
    STAGING {
        int id PK
        text source_url
        text glass_name
        text nasal_engraving
        text glass_index
        text material
        text glass_supplier_name
        text image_engraving
    }
```

### Table Enhanced (Données Nettoyées)
```mermaid
erDiagram
    ENHANCED {
        int id PK
        text nom_du_verre
        text gamme
        text serie
        text variante
        int hauteur_min
        int hauteur_max
        text traitement_protection
        text traitement_photochromique
        text materiau
        float indice
        text fournisseur
        text gravure
        text url_source
        timestamp created_at
    }
```

### Tables Normalisées
```mermaid
erDiagram
    VERRES ||--o{ VERRES_TRAITEMENTS : possède
    VERRES {
        int id PK
        text nom
        text variante
        int hauteur_min
        int hauteur_max
        float indice
        text gravure
        text url_source
        int fournisseur_id FK
        int materiau_id FK
        int gamme_id FK
        int serie_id FK
    }
    TRAITEMENTS ||--o{ VERRES_TRAITEMENTS : appliqué
    TRAITEMENTS {
        int id PK
        text nom
        text type
    }
    FOURNISSEURS ||--o{ VERRES : fournit
    FOURNISSEURS {
        int id PK
        text nom
    }
    MATERIAUX ||--o{ VERRES : compose
    MATERIAUX {
        int id PK
        text nom
    }
    GAMMES ||--o{ VERRES : catégorise
    GAMMES {
        int id PK
        text nom
    }
    SERIES ||--o{ VERRES : appartient
    SERIES {
        int id PK
        text nom
    }
```

## 🛠️ Fonctionnalités Principales

1. **Scraping des Données**
   - Collecte automatisée des informations sur les verres
   - Extraction des gravures et images associées

2. **Nettoyage des Données**
   - Standardisation des noms et valeurs
   - Enrichissement des informations
   - Validation des données

3. **Gestion des Images**
   - Téléchargement automatique
   - Amélioration de la qualité
   - Organisation structurée du stockage

4. **Base de Données Optimisée**
   - Structure normalisée
   - Indexation performante
   - Traçabilité des modifications

## 📦 Structure du Projet
```
E1_GestionDonnees/
├── Base_de_donnees/
│   ├── data_cleaning.py    # Nettoyage et enrichissement
│   ├── download_images.py  # Gestion des images
│   ├── migrate_db.sh      # Migration vers structure finale
│   └── images/            # Stockage des images
├── france_optique/
│   └── run_spiders.py     # Scripts de scraping
├── logs/                  # Journaux d'exécution
└── backups/              # Sauvegardes
```

## 🚀 Installation et Utilisation

1. Cloner le repository
```bash
git clone [url_du_repo]
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Exécuter le script principal
```bash
./run_project.sh
```

## 📝 Notes
- Les images et données sensibles sont exclues du versionnement
- Les logs sont générés pour chaque exécution
- Des sauvegardes sont créées automatiquement 