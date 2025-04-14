# E3 - Reconnaissance de Gravures Optiques

Ce module implémente un système de reconnaissance de gravures optiques pour verres de lunettes utilisant un réseau de neurones siamois pour comparer les croquis dessinés à la main avec une base de gravures connues.

## Caractéristiques

- API REST sécurisée par OAuth2/JWT
- Interface utilisateur Gradio pour dessiner et reconnaître les gravures
- Réseau siamois pour la génération d'embeddings
- Algorithme de recherche par similarité
- Pipeline de prétraitement d'image
- Conteneurisation Docker pour un déploiement facile

## Structure du projet

```
E3_MettreDispositionIA/
├── app/                      # Code source de l'application
│   ├── __init__.py           # Initialisation de l'application
│   ├── embeddings_manager.py # Gestion des embeddings
│   ├── generate_embeddings.py # Script pour générer les embeddings
│   ├── main.py               # Point d'entrée FastAPI
│   ├── model.py              # Modèle siamois
│   ├── train.py              # Script d'entraînement du modèle
│   └── ui.py                 # Interface Gradio
├── data/                     # Données
│   └── processed/            # Images traitées
├── embeddings/               # Embeddings précalculés
├── model/                    # Modèle entraîné
├── tests/                    # Tests unitaires
├── Dockerfile                # Configuration Docker
├── README.md                 # Documentation
└── requirements.txt          # Dépendances Python
```

## Installation

### Prérequis

- Python 3.9+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Démarrer l'application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Ensuite, accédez à:
- Interface Gradio: http://localhost:8000/gradio
- Documentation API: http://localhost:8000/docs

### Entraîner le modèle

```bash
python -m app.train --data_dir path/to/images --output_dir model/ --num_epochs 20
```

### Générer les embeddings

```bash
python -m app.generate_embeddings --images_dir path/to/images --output_path embeddings/gravures_embeddings.pkl
```

## API REST

### Authentification

```
POST /token
```
Corps de la requête (form-data):
- username: utilisateur
- password: password123

### Reconnaissance de gravure

```
POST /recognize
```
Headers:
- Authorization: Bearer {token}

Corps de la requête (form-data):
- file: image de la gravure

### Liste des gravures

```
GET /gravures
```
Headers:
- Authorization: Bearer {token}

## Déploiement avec Docker

### Construire l'image

```bash
docker build -t gravure-recognition:latest .
```

### Exécuter le conteneur

```bash
docker run -p 8000:8000 gravure-recognition:latest
```

## Crédits

Application développée dans le cadre du projet de certification Simplon. 