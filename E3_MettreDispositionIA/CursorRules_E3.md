# CursorRules – Implémentation de la partie E3

Ce document décrit étape par étape l’implémentation d’une application de reconnaissance de gravures optiques basée sur un **modèle siamois**, en **Python**, avec **FastAPI**, **Gradio**, **OAuth2/JWT**, **tests pytest**, intégration à la **CI/CD**, et **déploiement Docker**. Le but est de respecter les exigences du volet E3 en offrant un service d’IA complet et documenté.

---

## Table des matières
1. [Structure et objectifs](#structure-et-objectifs)
2. [Organisation du projet](#organisation-du-projet)
3. [Environnement et dépendances](#environnement-et-dependances)
4. [Entraînement du modèle siamois](#entrainement-du-modele-siamois)
5. [Sauvegarde des embeddings](#sauvegarde-des-embeddings)
6. [API REST (FastAPI) et Authentification](#api-rest-fastapi-et-authentification)
7. [Interface Gradio pour croquis](#interface-gradio-pour-croquis)
8. [Tests unitaires et d’intégration](#tests-unitaires-et-dintegration)
9. [Intégration CI/CD](#integration-cicd)
10. [Déploiement Docker (Dockerfile)](#deploiement-docker-dockerfile)
11. [Points clés de la documentation](#points-cles-de-la-documentation)
12. [Références et ressources](#references-et-ressources)

---

## 1. Structure et objectifs

### Objectif
- Entraînement d’un modèle IA (siamois) pour reconnaître une gravure optique à partir d’un **croquis** (dessin) et **matching** avec une base de gravures existantes.
- Exposition d’une **API REST** sécurisée (OAuth2 + JWT) avec **FastAPI**.
- **Interface Gradio** permettant à l’utilisateur de dessiner la gravure en temps réel et d’obtenir une correspondance.
- **Sauvegarde des embeddings** dans un fichier **pickle** pour accélérer la recherche.
- **Tests** (pytest) et intégration à la **CI/CD** existante.
- **Déploiement Docker** de l’ensemble.

### Contexte
- Les gravures optiques (microsymboles ou caractères gravés sur les verres) sont difficiles à photographier.
- Nous n’utilisons pas d’images d’entrée depuis un appareil photo, mais **exclusivement** des **croquis** dessinés par l’utilisateur, via un composant Gradio.
- Les gravures de référence sont déjà récupérées (scrapées) et organisées sous forme d’images ou de fichiers.

### Objectif pédagogique
- Un *workflow complet* (IA + API + UI).
- Une *documentation* technique claire.
- Une *intégration* dans l’existant (tests, CI/CD, Docker).

---

## 2. Organisation du projet

Proposition de structure de répertoires :

```
E3/
├── app/
│   ├── __init__.py
│   ├── main.py  # Point d'entrée FastAPI
│   ├── auth.py  # Gestion OAuth2, JWT
│   ├── model.py  # Architecture du réseau siamois
│   ├── train.py  # Script ou module d'entraînement
│   ├── inference.py  # Fonctions de prédiction / embedding / matching
│   ├── schemas.py  # Schémas Pydantic pour les endpoints
│   └── ...
├── data/
│   ├── raw_gravures/  # Images scrapées
│   ├── processed/  # Images prétraitées (optionnel)
│   └── ...
├── embeddings/  # Fichier pickle des embeddings calculés
│   └── embeddings.pkl
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── ...
├── Dockerfile
├── requirements.txt
└── CursorRules.md  # Le présent fichier
```

Les modules Python peuvent être adaptés selon l’organisation souhaitée, l’important étant de séparer clairement l’entraînement du modèle, la logique d’inférence (embedding + matching), et l’API.

---

## 3. Environnement et dépendances

- **Python 3.9+**
- Bibliothèques majeures :
  - `fastapi`, `uvicorn` : API REST.
  - `gradio` : Interface utilisateur pour croquis.
  - `torch` ou `tensorflow` : Modèle siamois.
  - `PyJWT` : Authentification JWT.
  - `pytest` : Tests unitaires.
  - `requests` : Appels API.
  - `pickle` ou `joblib` : Serialization des embeddings.

Exemple `requirements.txt` :
```
fastapi==0.95.0
uvicorn==0.22.0
gradio==3.50.0
torch==2.0.0
PyJWT==2.6.0
pytest==7.2.1
```

---

## 4. Entraînement du modèle siamois

### 4.1. Préparation des données
- Placer les images de gravures scrapées dans `data/raw_gravures/`.
- Générer des **paires positives** (même gravure) et **paires négatives** (gravures différentes).
- Stocker ces paires dans un CSV.

### 4.2. Architecture du réseau
```python
import torch
import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3)
    def forward_once(self, x):
        return self.conv(x)
    def forward(self, input1, input2):
        return torch.abs(self.forward_once(input1) - self.forward_once(input2))
```
