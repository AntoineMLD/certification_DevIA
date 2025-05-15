# API REST pour Modèle d'IA de Classification d'Images

Cette API expose les fonctionnalités d'un modèle d'intelligence artificielle pour la classification d'images de verres et la recherche par caractéristiques.

## Fonctionnalités

- Classification d'images de verres
- Recherche de verres par tags
- Calcul d'embeddings vectoriels
- Validation des prédictions et collecte de métriques
- Monitoring des performances du modèle

## Prérequis

- Python 3.8+
- Base de données SQLite
- Modèle préentraîné d'IA

## Installation

1. Cloner le dépôt
```bash
git clone <url-du-depot>
cd E3_MettreDispositionIA/main/api
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement
```bash
# Copier le fichier d'exemple
cp .env.example .env
# Éditer le fichier .env avec vos valeurs
```

## Configuration

Les variables d'environnement principales à configurer sont :

- `SECRET_KEY` : Clé secrète pour la signature des tokens JWT
- `ADMIN_EMAIL` et `ADMIN_PASSWORD` : Identifiants pour l'authentification
- `DB_PATH` : Chemin vers la base de données

## Lancement de l'API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Documentation de l'API

Une fois l'API lancée, vous pouvez accéder à la documentation interactive :

- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

## Points d'API

### Authentification

```
POST /token
```
Permet d'obtenir un token JWT pour accéder aux endpoints protégés.

### Classification d'image

```
POST /match
```
Analyse une image et retourne les classes les plus similaires.

### Recherche de verres

```
POST /search_tags
```
Recherche des verres par tags.

### Détails d'un verre

```
GET /verre/{verre_id}
```
Récupère les détails complets d'un verre par son ID.

### Calcul d'embedding

```
POST /embedding
```
Calcule l'embedding vectoriel d'une image.

### Validation de prédiction

```
POST /validate_prediction
```
Valide une prédiction et l'ajoute aux métriques de monitoring.

## Tests

Pour exécuter les tests unitaires :

```bash
pytest
```

## Structure du projet

- `app/` : Code source de l'API
  - `main.py` : Point d'entrée principal et définition des routes
  - `model_loader.py` : Chargement et utilisation du modèle
  - `security.py` : Gestion de l'authentification et de la sécurité
  - `database.py` : Accès à la base de données
  - `similarity_search.py` : Recherche par similarité
  - `config.py` : Configuration centralisée
  - `monitoring/` : Monitoring et métriques du modèle
- `tests/` : Tests unitaires
- `weights/` : Poids du modèle préentraîné
- `data/` : Données de référence 