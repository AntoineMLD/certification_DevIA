# 🔍 Application de Recherche de Gravures Similaires

![Logo du projet](https://img.shields.io/badge/IA-Visuelle-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-purple)

## 📋 Table des matières
- [Introduction](#introduction)
- [Architecture du projet](#architecture-du-projet)
  - [Application Frontend Streamlit](#application-frontend-streamlit)
  - [API Backend FastAPI](#api-backend-fastapi)
  - [Interaction Frontend-Backend](#interaction-frontend-backend)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
  - [Prérequis](#prérequis)
  - [Installation des dépendances](#installation-des-dépendances)
- [Utilisation](#utilisation)
  - [Lancement de l'API Backend (FastAPI)](#lancement-de-lapi-backend-fastapi)
  - [Lancement de l'Application Frontend (Streamlit)](#lancement-de-lapplication-frontend-streamlit)
  - [Utilisation de l'interface](#utilisation-de-linterface)
- [Déploiement](#déploiement)
  - [API Backend FastAPI](#api-backend-fastapi-1)
  - [Application Frontend Streamlit](#application-frontend-streamlit-1)
- [Structure du code](#structure-du-code)
- [Pipeline de données](#pipeline-de-données)
- [Modèle d'IA](#modèle-dia)
- [Interface utilisateur](#interface-utilisateur)

## Introduction

Cette application permet aux utilisateurs de dessiner des gravures à main levée et de trouver automatiquement les gravures les plus similaires dans une base de données. Elle se compose d'une interface utilisateur développée avec Streamlit et d'une API backend construite avec FastAPI qui sert le modèle d'IA. Le modèle utilise un apprentissage profond basé sur EfficientNet et la technique de Triplet Loss pour apprendre des représentations vectorielles (embeddings) des images.

## Architecture du projet

Le projet `E3_MettreDispositionIA` est structuré en deux composants principaux : une application frontend Streamlit pour l'interaction utilisateur et une API backend FastAPI pour la logique métier et le service du modèle d'IA.

```
E3_MettreDispositionIA/
├── main/
│   ├── app/                      # Application Frontend Streamlit
│   │   ├── app.py                # Point d'entrée de l'application Streamlit
│   │   ├── auth.py               # Gestion de l'authentification Streamlit
│   │   └── api_client.py         # Client pour communiquer avec l'API FastAPI
│   ├── api/                      # API Backend FastAPI
│   │   ├── app/
│   │   │   ├── main.py           # Point d'entrée de l'API FastAPI
│   │   │   ├── security.py       # Gestion de la sécurité, tokens JWT
│   │   │   ├── config.py         # Configuration de l'API (variables d'env)
│   │   │   └── database.py       # Interactions avec la base de données (si utilisées par l'API)
│   │   ├── models/               # Modèles Pydantic (schemas) pour l'API (si séparés)
│   │   └── tests/                # Tests pour l'API FastAPI
│   │   └── .env.example          # Exemple de fichier d'environnement pour l'API
│   │   └── requirements.txt      # Dépendances spécifiques à l'API
│   ├── models/                   # Modèles d'IA
│   │   ├── efficientnet_triplet.py # Modèle EfficientNet
│   │   ├── efficientnet_triplet.pth # Modèle entraîné
│   │   ├── train.py              # Script d'entraînement
│   │   └── losses/
│   │       └── triplet_losses.py # Implémentation de la Triplet Loss
│   ├── datasets/
│   │   └── triplet_dataset.py    # Dataset pour l'entraînement
│   ├── data/
│   │   ├── raw_gravures/         # Données brutes
│   │   ├── augmented_gravures/   # Données augmentées
│   │   └── oversampled_gravures/ # Données équilibrées
│   ├── monitoring/               # Scripts et outils de monitoring
│   │   ├── metrics_collector.py
│   │   └── dashboard.py
│   ├── augment_gravures.py       # Script d'augmentation
│   └── oversample_classes.py     # Script d'équilibrage
├── requirements.txt              # Dépendances communes / du projet global E3
└── README.md
```

### Application Frontend Streamlit
Située dans `main/app/`, elle fournit l'interface graphique où les utilisateurs peuvent dessiner des gravures et voir les résultats de la recherche. Elle communique avec l'API backend pour toutes les opérations liées au modèle et potentiellement à l'authentification.

### API Backend FastAPI
Située dans `main/api/`, elle expose des endpoints pour :
- L'authentification des utilisateurs (génération de tokens JWT).
- Le chargement du modèle d'IA.
- La génération d'embeddings pour les images dessinées.
- La recherche de similarités avec les embeddings de référence.
- La récupération des détails des verres (potentiellement en communiquant avec l'API E1).

L'API assure que l'accès aux fonctionnalités du modèle est sécurisé et contrôlé. Elle est conçue pour être stateless autant que possible.

### Interaction Frontend-Backend
L'application Streamlit (frontend) utilise le module `main/app/api_client.py` pour effectuer des requêtes HTTP vers l'API FastAPI (backend). Les tokens d'authentification obtenus via l'API sont stockés et utilisés par le client pour les requêtes sécurisées.

## Fonctionnalités

- 🎨 Interface de dessin à main levée (via Streamlit)
- 🔑 Authentification des utilisateurs pour accéder aux fonctionnalités
- 🔍 Recherche de gravures similaires en temps réel (via API FastAPI)
- 📊 Affichage des 10 résultats les plus pertinents
- 🧠 Modèle d'IA entraîné sur des gravures historiques
- 📱 Interface utilisateur intuitive

## Installation

### Prérequis
- Python 3.8+
- `pip` pour l'installation des paquets
- Accès à une instance de l'API E1 (Gestion des Données) si l'API E3 en dépend pour certaines informations.

### Installation des dépendances
1. Clonez le dépôt (si ce n''est pas déjà fait) :
```bash
git clone https://github.com/votre-utilisateur/E3_MettreDispositionIA.git # Adaptez l'URL
cd E3_MettreDispositionIA
```

2. Il est recommandé d'utiliser un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/macOS
# venv\Scripts\activate    # Sur Windows
```

3. Installez les dépendances pour l'ensemble du projet E3 (incluant Streamlit, FastAPI et le modèle) :
   Assurez-vous que les `requirements.txt` sont bien structurés. Il pourrait y avoir un `requirements.txt` à la racine de `E3_MettreDispositionIA` pour les dépendances communes, et des `requirements.txt` spécifiques dans `main/api/` pour l'API.
   Pour cet exemple, nous supposons un `requirements.txt` principal et un pour l'API.
```bash
pip install -r requirements.txt                 # Dépendances principales (Streamlit, modèle)
pip install -r main/api/requirements.txt        # Dépendances de l'API FastAPI
```

## Utilisation

Pour utiliser l'application, l'API backend FastAPI et l'application frontend Streamlit doivent être lancées.

### Lancement de l'API Backend (FastAPI)
1. Configurez l'environnement de l'API :
   Créez un fichier `.env` dans le dossier `E3_MettreDispositionIA/main/api/` en vous basant sur `E3_MettreDispositionIA/main/api/.env.example`.
   Remplissez les variables nécessaires comme `SECRET_KEY`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `DATABASE_URL` (si l'API E3 utilise sa propre base ou pour la configuration de la connexion à l'API E1).

2. Lancez le serveur Uvicorn depuis le dossier `E3_MettreDispositionIA/main/api/` :
```bash
cd E3_MettreDispositionIA/main/api 
uvicorn app.main:app --reload --port 8000 
```
L'API sera généralement accessible à `http://localhost:8000`. Vous pouvez consulter sa documentation OpenAPI interactive sur `http://localhost:8000/docs`.

### Lancement de l'Application Frontend (Streamlit)
1. Assurez-vous que l'API FastAPI est en cours d'exécution.
2. Lancez l'application Streamlit depuis le dossier `E3_MettreDispositionIA/main/` :
```bash
cd E3_MettreDispositionIA/main 
streamlit run app/app.py
```
L'application Streamlit sera généralement accessible à `http://localhost:8501`.

### Utilisation de l'interface
1. Ouvrez l'application Streamlit dans votre navigateur.
2. Connectez-vous si un système d'authentification est actif.
3. Dessinez une gravure dans la zone de dessin.
4. Cliquez sur le bouton "🔍 Rechercher les gravures similaires".
5. Consultez les résultats affichés avec leur score de similarité.

## Déploiement

Le déploiement implique de rendre accessibles l'API FastAPI et l'application Streamlit.

### API Backend FastAPI
L'API FastAPI peut être conteneurisée avec Docker et déployée sur diverses plateformes :
- Services d'hébergement de conteneurs (AWS ECS, Google Cloud Run, Azure Container Instances)
- Serveurs virtuels avec Uvicorn derrière un reverse proxy comme Nginx.
- Plateformes PaaS supportant Python/FastAPI.

Consultez la documentation de FastAPI pour les meilleures pratiques de déploiement.

### Application Frontend Streamlit
Pour déployer l'application Streamlit et la rendre accessible :

#### Option 1 : Streamlit Cloud (recommandé)
1. Créez un compte sur [Streamlit Cloud](https://streamlit.io/cloud).
2. Connectez votre dépôt GitHub.
3. Sélectionnez le fichier `main/app/app.py` comme point d'entrée.
4. Assurez-vous que l'application Streamlit peut atteindre l'API FastAPI déployée (configurez l'URL de l'API dans Streamlit, par exemple via les secrets Streamlit).
5. Déployez l'application.

#### Option 2 : Heroku
1. Créez un fichier `Procfile` à la racine du projet `E3_MettreDispositionIA` (ou ajustez les chemins) :
```Procfile
web: streamlit run main/app/app.py
```
2. Déployez sur Heroku. L'API FastAPI devra être déployée séparément et son URL configurée dans l'application Streamlit.

#### Option 3 : Serveur personnel
1. Installez les dépendances sur votre serveur.
2. Lancez l'application Streamlit.
3. Configurez un reverse proxy.

Il est crucial que l'application Streamlit déployée puisse communiquer avec l'API FastAPI déployée.

## Structure du code

### Application Streamlit (`main/app/app.py`)
- Interface utilisateur avec zone de dessin.
- Communication avec l'API FastAPI via `api_client.py` pour l'authentification et la recherche.
- Affichage des résultats.

### API FastAPI (`main/api/app/main.py`)
- Endpoints pour `/token`, `/search_tags`, `/match`, `/embedding`, `/verre/{verre_id}`.
- Logique de chargement du modèle d'IA.
- Sécurité des endpoints via tokens JWT.

### Modèle EfficientNet (`main/models/efficientnet_triplet.py`)
- Architecture basée sur EfficientNet-B0.
- Adaptation pour les images en niveaux de gris.
- Tête d'embedding pour générer des vecteurs de 256 dimensions.

### Triplet Loss (`main/models/losses/triplet_losses.py`)
- Implémentation de la Triplet Loss standard.
- Version avec "hard mining" pour sélectionner les triplets difficiles.
- Optimisation pour l'apprentissage de représentations discriminatives.

## Pipeline de données

Le projet utilise un pipeline de données complet pour préparer les données d'entraînement :

1. **Données brutes** : Collection initiale de gravures
2. **Augmentation** : Génération de variations pour enrichir le dataset
   - Rotations, translations, changements d'échelle
   - Modifications de luminosité et contraste
   - Transformations élastiques et perspectives
3. **Équilibrage** : Oversampling des classes minoritaires
   - Duplication des images pour atteindre un minimum de 80 images par classe
   - Distribution équilibrée pour un entraînement optimal

## Modèle d'IA

Le modèle utilise une architecture d'apprentissage par transfert avec EfficientNet-B0 :

1. **Backbone** : EfficientNet-B0 pré-entraîné sur ImageNet
2. **Adaptation** : Conversion des images en niveaux de gris vers 3 canaux
3. **Tête d'embedding** : MLP pour projeter les features en vecteurs de 256 dimensions
4. **Entraînement** : Triplet Loss avec "semi-hard mining" pour optimiser les représentations

## Interface utilisateur

L'interface utilisateur est conçue pour être intuitive et réactive :

- Zone de dessin avec pinceau personnalisable
- Boutons pour effacer le dessin et lancer la recherche
- Affichage en grille des résultats avec scores de similarité
- Design épuré et moderne

---

Développé avec ❤️ par [Votre Nom] 