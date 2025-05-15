# Documentation de l'API E3_MettreDispositionIA

Cette documentation couvre l'architecture, les points de terminaison et les règles d'authentification de l'API E3_MettreDispositionIA.

## Table des matières

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Authentification](#authentification)
4. [Points de terminaison](#points-de-terminaison)
5. [Exemples d'utilisation](#exemples-dutilisation)
6. [Recommandations de sécurité](#recommandations-de-sécurité)
7. [Accessibilité](#accessibilité)

## Introduction

L'API E3_MettreDispositionIA permet de rechercher des gravures similaires à partir d'une image dessinée par l'utilisateur. Elle utilise un modèle d'IA basé sur EfficientNet pour générer des embeddings vectoriels des images et calculer la similarité entre elles.

## Architecture

L'API est construite avec FastAPI et suit une architecture modulaire avec les composants suivants :

```
E3_MettreDispositionIA/
├── main/
│   ├── api/
│   │   ├── app/
│   │   │   ├── main.py           # Point d'entrée de l'API
│   │   │   ├── security.py       # Gestion de la sécurité (tokens JWT)
│   │   │   ├── model_loader.py   # Chargement et utilisation du modèle
│   │   │   ├── similarity_search.py # Recherche de similarité
│   │   │   ├── database.py       # Interactions avec la base de données
│   │   │   ├── config.py         # Configuration de l'API
│   │   │   └── monitoring/       # Module de monitoring
│   │   ├── tests/                # Tests unitaires et d'intégration
│   │   └── docs/                 # Documentation (ce dossier)
│   ├── models/                   # Modèles d'IA
│   │   └── efficientnet_triplet.py # Implémentation du modèle
```

### Flux de données

1. L'utilisateur soumet une image via l'API
2. L'image est prétraitée et validée
3. Le modèle génère un embedding vectoriel de l'image
4. L'API recherche les embeddings les plus similaires dans sa base de références
5. Les résultats triés par similarité sont renvoyés à l'utilisateur

### Technologies utilisées

- **FastAPI**: Framework web pour la création de l'API
- **PyTorch**: Bibliothèque pour le modèle d'IA
- **JWT**: Authentification basée sur des tokens
- **Scikit-learn**: Calcul de similarité cosinus
- **Slowapi**: Rate limiting

## Authentification

L'API utilise l'authentification OAuth2 avec des tokens JWT. Pour accéder aux endpoints protégés :

1. Obtenez un token en envoyant vos identifiants à l'endpoint `/token`
2. Incluez ce token dans l'en-tête `Authorization` de toutes vos requêtes sous la forme `Bearer {token}`

Les tokens ont une durée de validité limitée et sont automatiquement renouvelés lorsqu'ils approchent de leur expiration.

### Gestion des tokens

- **Émission** : Les tokens sont émis avec une durée de validité définie
- **Validation** : Chaque requête vérifie la validité du token
- **Rotation** : Les tokens proches de l'expiration sont automatiquement renouvelés
- **Révocation** : Les versions de token permettent la révocation immédiate en cas de compromission

## Points de terminaison

L'API expose les endpoints suivants :

### `/token` (POST)

Obtenir un token d'authentification.

### `/embedding` (POST)

Calcule et renvoie l'embedding vectoriel d'une image.

### `/match` (POST)

Analyse une image et renvoie les classes les plus similaires.

### `/validate_prediction` (POST)

Permet de valider une prédiction et de l'ajouter aux métriques.

### `/search_tags` (POST)

Recherche les verres correspondant à une liste de tags.

### `/verre/{verre_id}` (GET)

Récupère les détails complets d'un verre par son ID.

Pour des informations détaillées sur chaque endpoint, y compris les paramètres, les corps de requête et les réponses, consultez la documentation OpenAPI dans le fichier [openapi.yaml](./openapi.yaml) ou via l'interface Swagger de l'API à l'adresse `/docs` lorsque l'API est en cours d'exécution.

## Exemples d'utilisation

### Obtenir un token d'authentification

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@example.com&password=admin_password"
```

### Rechercher des images similaires

```bash
curl -X POST "http://localhost:8000/match" \
  -H "Authorization: Bearer votre_token_jwt" \
  -F "file=@chemin/vers/votre/image.png"
```

## Recommandations de sécurité

- Stockez les tokens JWT de manière sécurisée côté client
- Utilisez HTTPS en production pour protéger les communications
- Ne partagez jamais vos identifiants d'API
- Respectez les limites de taux pour éviter les restrictions d'accès

## Accessibilité

Cette documentation a été conçue pour être accessible à tous les utilisateurs, conformément aux recommandations d'accessibilité suivantes :

### Structure du document

- Utilisation cohérente des niveaux de titre (h1, h2, h3) pour créer une hiérarchie claire
- Table des matières avec liens d'ancrage pour faciliter la navigation
- Texte alternatif pour toutes les images et diagrammes
- Utilisation de listes pour présenter l'information de manière structurée

### Lisibilité

- Police sans-serif pour une meilleure lisibilité à l'écran
- Contraste élevé entre le texte et l'arrière-plan
- Taille de police ajustable
- Texte aligné à gauche pour faciliter la lecture
- Phrases et paragraphes courts et clairs

### Compatibilité avec les lecteurs d'écran

- Utilisation appropriée des balises sémantiques HTML
- Descriptions textuelles pour les éléments non textuels
- Indication de la langue principale du document
- Liens explicites avec texte descriptif

### Accessibilité des exemples de code

- Utilisation d'indentation cohérente
- Commentaires explicatifs
- Indication du langage pour la coloration syntaxique
- Exemples simples et compréhensibles

Pour en savoir plus sur les recommandations d'accessibilité, consultez :
- [Recommandations de l'association Valentin Haüy](https://www.avh.asso.fr/fr/favoriser-laccessibilite/accessibilite-numerique)
- [Directives d'accessibilité Microsoft](https://www.microsoft.com/fr-fr/accessibility) 