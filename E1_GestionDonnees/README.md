# E1 - Gestion des Données pour le Système de Gravures Optiques

## 📋 Contexte du Projet

### Présentation
Ce module (E1) constitue la première étape d'un projet de certification Développeur IA. Il établit les fondations d'un système de gestion des gravures de verres optiques, qui sera enrichi par des fonctionnalités d'IA dans les phases ultérieures.

### Acteurs
- **Utilisateurs finaux** : Opticiens cherchant à optimiser leur processus de recherche de gravures
- **Développeur** : Développeur IA en charge de la conception et réalisation
- **Client test** : Opticien partenaire pour les revues de sprint

## 🎯 Objectifs

### Objectifs Fonctionnels
- Collecter automatiquement les données des verres depuis les sites fournisseurs
- Nettoyer et standardiser les données pour une utilisation future
- Stocker les données de manière structurée et optimisée
- Préparer l'infrastructure pour les futures fonctionnalités IA

### Objectifs Techniques
- Temps de recherche cible : < 30 secondes
- Base de données normalisée et optimisée
- API REST avec FastAPI
- Solution containerisée (Docker)
- Compatible multi-plateforme

### Contraintes Techniques
- Respect du RGPD pour la gestion des tokens API
- Connexion internet standard requise
- Pas de contraintes matérielles spécifiques

## 🔧 Environnement Technique

### Architecture
Le module E1 est principalement constitué d'une API RESTful développée avec FastAPI. Cette API sert de backend pour la gestion des données des verres optiques.

- **Base de données** : SQLite (par défaut pour le développement local, fichier `Base_de_donnees/france_optique.db`). Le système est conçu pour être potentiellement compatible avec PostgreSQL pour la production (voir `requirements.txt`).
- **API** : Développée avec FastAPI, fournissant des endpoints CRUD pour les entités de la base de données (Verres, Fournisseurs, Matériaux, etc.).
- **Authentification** : Sécurisée par tokens JWT pour protéger les endpoints.
- **Conteneurisation** : Un `Dockerfile` et un `docker-compose.yml` sont fournis pour faciliter le déploiement et l'exécution dans un environnement conteneurisé.
- **Scripts Python** : Des scripts peuvent exister pour le scraping initial des données et le peuplement de la base (non gérés directement par l'API en fonctionnement normal).

La structure principale de l'API se trouve dans le dossier `api/`:
```
E1_GestionDonnees/
├── api/
│   ├── app/
│   │   ├── main.py         # Point d'entrée de l'application FastAPI, définit les routes
│   │   │   ├── auth/
│   │   │   │   └── jwt_auth.py # Logique d'authentification JWT
│   │   │   ├── models/
│   │   │   │   └── database.py # Modèles SQLAlchemy et initialisation de la DB
│   │   │   ├── schemas/
│   │   │   │   └── schemas.py  # Modèles Pydantic pour la validation des données et les réponses API
│   │   │   ├── config.py       # Configuration de l'application (chargement depuis .env)
│   │   │   └── dependencies.py # Dépendances FastAPI (si utilisées, ex: get_db)
│   │   ├── tests/              # Tests unitaires et d'intégration pour l'API
│   │   │   └── test_main.py
│   │   ├── .env.example        # Exemple de fichier d'environnement
│   │   └── requirements.txt    # Dépendances Python pour l'API
├── Base_de_donnees/
│   └── france_optique.db   # Fichier de base de données SQLite (si utilisé)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Compatibilité
- Multi-plateforme (Windows, Linux, MacOS)
- Navigateurs web modernes
- Connexion internet standard

## 📅 Organisation et Planification

### Méthodologie
- **Approche** : Scrum
- **Durée** : 3 mois
- **Sprint** : 2 semaines
- **Review** : Fin de chaque sprint avec l'opticien partenaire

### User Stories Principales
1. **Collecte des Données**
   ```
   En tant que développeur,
   Je veux scraper les sites des fournisseurs
   Afin de collecter les données des verres
   ```

2. **Nettoyage des Données**
   ```
   En tant que développeur,
   Je veux nettoyer et standardiser les données
   Afin d'assurer leur qualité et leur cohérence
   ```

3. **Stockage Optimisé**
   ```
   En tant que développeur,
   Je veux stocker les données dans une base normalisée
   Afin de faciliter les futures requêtes
   ```

4. **Gestion des Images**
   ```
   En tant que développeur,
   Je veux gérer le téléchargement et le stockage des images
   Afin de conserver les gravures de manière organisée
   ```

5. **Structure Base de Données**
   ```
   En tant que développeur,
   Je veux mettre en place une structure de base optimisée
   Afin de supporter les futures fonctionnalités
   ```

### Planning des Sprints

#### Sprint 1-2 : Infrastructure
- Mise en place de l'environnement
- Création de la structure de base de données
- Configuration du scraping

#### Sprint 3-4 : Collecte et Traitement
- Développement des scrapers
- Implémentation du nettoyage des données
- Tests de collecte

#### Sprint 5-6 : Optimisation et Documentation
- Optimisation des performances
- Documentation technique
- Tests d'intégration

## 📊 Mesures de Succès
- Base de données structurée et normalisée
- Données nettoyées et standardisées
- Tests unitaires et d'intégration passants
- Documentation technique complète
- Infrastructure prête pour les phases suivantes

## 🔄 Workflow des Données
```mermaid
graph TD
    A[Sites Fournisseurs] -->|Scraping| B[Données Brutes]
    B -->|Nettoyage| C[Données Standardisées]
    C -->|Validation| D[Base de Données]
    B -->|Extraction| E[Images Gravures]
    E -->|Traitement| F[Stockage Local]
```

## 📝 Notes
- Ce module E1 constitue la fondation pour les futures fonctionnalités IA
- L'accent est mis sur la qualité et la structure des données
- La documentation est considérée comme un délivrable critique

## CI/CD

Le projet utilise GitHub Actions pour l'intégration et le déploiement continus.

### Processus automatisé

1. **Tests automatiques** :
   - Vérification du code (flake8)
   - Exécution des tests unitaires (pytest)
   - Vérification des dépendances (snyk)

2. **Quand** :
   - À chaque push sur la branche main
   - À chaque pull request vers main

### Exécuter les tests localement

```bash
# Installation des outils de test
pip install flake8 pytest

# Vérification du code
flake8 .

# Exécution des tests
pytest
```

### Bonnes pratiques

1. **Avant chaque commit** :
   - Exécuter les tests localement
   - Vérifier le style du code avec flake8
   - Résoudre les conflits éventuels

2. **Pour les pull requests** :
   - Créer une branche pour chaque fonctionnalité
   - Attendre la validation des tests
   - Faire relire le code (si possible)

3. **Maintenance** :
   - Mettre à jour régulièrement les dépendances
   - Vérifier les rapports de sécurité
   - Maintenir une couverture de tests suffisante

### Configuration de l'API (FastAPI)

La configuration de l'API est gérée via le fichier `api/app/config.py`, qui charge les variables d'un fichier `.env` situé dans `api/.env`.
Un fichier d'exemple `api/.env.example` est fourni. Copiez-le en `api/.env` et ajustez les valeurs :

```dotenv
# api/.env
SECRET_KEY="votre_cle_secrete_tres_longue_et_aleatoire_ici"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Configuration de la base de données (SQLite par défaut)
# Assurez-vous que le chemin est correct par rapport à l'emplacement d'exécution de l'API.
# Si exécuté depuis E1_GestionDonnees/api/, le chemin relatif pour la DB à la racine de E1_GestionDonnees est ../Base_de_donnees/france_optique.db
DATABASE_URL="sqlite:///../Base_de_donnees/france_optique.db"

# Identifiants pour l'utilisateur admin par défaut (utilisé par jwt_auth.check_user)
# Ces identifiants sont utilisés pour obtenir un token via l'endpoint /token
ADMIN_EMAIL="admin@example.com"
ADMIN_PASSWORD="admin123"

# API Info
API_VERSION="1.0"
API_TITLE="API de Gestion de Données Optiques"
API_DESCRIPTION="Fournit un accès CRUD aux données des verres optiques."
```

**Variables clés** :
- `SECRET_KEY` : Une chaîne aléatoire longue et complexe pour la signature des tokens JWT. **À CHANGER EN PRODUCTION.**
- `ALGORITHM` : Algorithme de signature JWT (HS256 par défaut).
- `ACCESS_TOKEN_EXPIRE_MINUTES` : Durée de validité des tokens.
- `DATABASE_URL` : URL de connexion à la base de données. Par défaut, configurée pour SQLite. Pour PostgreSQL, le format serait `postgresql://user:password@host:port/dbname`.
- `ADMIN_EMAIL` / `ADMIN_PASSWORD` : Identifiants pour l'utilisateur par défaut permettant de générer un token initial pour tester l'API. En production, un système de gestion d'utilisateurs plus robuste serait nécessaire.

### Lancement de l'API localement

1.  **Prérequis** :
    *   Python 3.8+
    *   Avoir installé les dépendances de `api/requirements.txt` dans un environnement virtuel.
    *   Avoir configuré le fichier `api/.env`.
    *   La base de données doit exister et être initialisée (si elle n'est pas créée automatiquement).

2.  **Depuis le dossier `E1_GestionDonnees/api/`** :
    ```bash
    # Activer votre environnement virtuel si ce n'est pas déjà fait
    # source ../venv/bin/activate  # ou le chemin vers votre venv

    uvicorn app.main:app --reload --port 8001
    ```
    L'API sera alors accessible à `http://localhost:8001`.

3.  **Documentation de l'API** :
    Une fois l'API lancée, vous pouvez accéder à la documentation interactive (Swagger UI) générée automatiquement par FastAPI à l'adresse :
    `http://localhost:8001/docs`

    Et à la documentation alternative (ReDoc) :
    `http://localhost:8001/redoc`

### Lancement avec Docker (Optionnel)

Si vous souhaitez utiliser Docker :

1.  Assurez-vous que Docker et Docker Compose sont installés.
2.  Depuis la racine du dossier `E1_GestionDonnees/` :
    ```bash
    docker-compose up --build
    ```
    L'API devrait être accessible selon la configuration dans `docker-compose.yml` (souvent sur le port 8001 également, mais mappé depuis le conteneur).

### Exécuter les tests localement

```bash
# Installation des outils de test
pip install flake8 pytest

# Vérification du code
flake8 .

# Exécution des tests
pytest
``` 