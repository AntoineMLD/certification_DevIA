# 🔍 Application de Recherche de Gravures Similaires

![Logo du projet](https://img.shields.io/badge/IA-Visuelle-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

## 📋 Table des matières
- [Introduction](#introduction)
- [Architecture du projet](#architecture-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Déploiement](#déploiement)
- [Structure du code](#structure-du-code)
- [Pipeline de données](#pipeline-de-données)
- [Modèle d'IA](#modèle-dia)
- [Interface utilisateur](#interface-utilisateur)

## Introduction

Cette application permet aux utilisateurs de dessiner des gravures à main levée et de trouver automatiquement les gravures les plus similaires dans une base de données. Elle utilise un modèle d'apprentissage profond basé sur EfficientNet et la technique de Triplet Loss pour apprendre des représentations vectorielles (embeddings) des images.

## Architecture du projet

```
E3_MettreDispositionIA/
├── main/
│   ├── app/
│   │   └── app.py                 # Application Streamlit
│   ├── models/
│   │   ├── efficientnet_triplet.py # Modèle EfficientNet
│   │   ├── efficientnet_triplet.pth # Modèle entraîné
│   │   ├── train.py               # Script d'entraînement
│   │   └── losses/
│   │       └── triplet_losses.py   # Implémentation de la Triplet Loss
│   ├── datasets/
│   │   └── triplet_dataset.py      # Dataset pour l'entraînement
│   ├── data/
│   │   ├── raw_gravures/           # Données brutes
│   │   ├── augmented_gravures/     # Données augmentées
│   │   └── oversampled_gravures/   # Données équilibrées
│   ├── augment_gravures.py         # Script d'augmentation
│   └── oversample_classes.py       # Script d'équilibrage
└── requirements.txt                 # Dépendances
```

## Fonctionnalités

- 🎨 Interface de dessin à main levée
- 🔍 Recherche de gravures similaires en temps réel
- 📊 Affichage des 10 résultats les plus pertinents
- 🧠 Modèle d'IA entraîné sur des gravures historiques
- 📱 Interface utilisateur intuitive

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-utilisateur/E3_MettreDispositionIA.git
cd E3_MettreDispositionIA
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancez l'application :
```bash
cd main
streamlit run app/app.py
```

## Utilisation

1. Ouvrez l'application dans votre navigateur (généralement à l'adresse http://localhost:8501)
2. Dessinez une gravure dans la zone de dessin
3. Cliquez sur le bouton "🔍 Rechercher les gravures similaires"
4. Consultez les résultats affichés avec leur score de similarité

## Déploiement

Pour déployer l'application et la rendre accessible à des utilisateurs externes, vous avez plusieurs options :

### Option 1 : Streamlit Cloud (recommandé)
1. Créez un compte sur [Streamlit Cloud](https://streamlit.io/cloud)
2. Connectez votre dépôt GitHub
3. Sélectionnez le fichier `app/app.py` comme point d'entrée
4. Déployez l'application

### Option 2 : Heroku
1. Créez un fichier `Procfile` à la racine du projet :
```
web: cd main && streamlit run app/app.py
```
2. Déployez sur Heroku :
```bash
heroku create votre-app-name
git push heroku main
```

### Option 3 : Serveur personnel
1. Installez les dépendances sur votre serveur
2. Lancez l'application avec :
```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```
3. Configurez un reverse proxy (Nginx, Apache) pour exposer l'application

## Structure du code

### Application Streamlit (`app.py`)
- Interface utilisateur avec zone de dessin
- Chargement du modèle et des embeddings de référence
- Calcul des similarités et affichage des résultats

### Modèle EfficientNet (`efficientnet_triplet.py`)
- Architecture basée sur EfficientNet-B0
- Adaptation pour les images en niveaux de gris
- Tête d'embedding pour générer des vecteurs de 256 dimensions

### Triplet Loss (`triplet_losses.py`)
- Implémentation de la Triplet Loss standard
- Version avec "hard mining" pour sélectionner les triplets difficiles
- Optimisation pour l'apprentissage de représentations discriminatives

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