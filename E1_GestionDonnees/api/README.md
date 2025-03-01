# API France Optique - Documentation Technique

## Table des matières
- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Installation](#installation)
- [Authentification](#authentification)
- [Points de terminaison (Endpoints)](#points-de-terminaison)
- [Modèles de données](#modèles-de-données)
- [Gestion des erreurs](#gestion-des-erreurs)
- [Tests](#tests)

## Vue d'ensemble

L'API France Optique est une API REST sécurisée permettant de gérer et d'accéder aux données des verres optiques. Elle utilise FastAPI pour offrir une interface moderne et performante.

```mermaid
graph TD
    A[Client] -->|Requête HTTP| B[API FastAPI]
    B -->|JWT Auth| C[Middleware Authentification]
    C -->|Validé| D[Endpoints]
    D -->|SQLAlchemy| E[Base de données SQLite]
    D -->|Réponse JSON| A
```

## Architecture

Structure du projet :
```
api/
├── app/
│   ├── auth/
│   │   ├── __init__.py
│   │   └── jwt_auth.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── Base_de_donnees/
│   └── france_optique.db
├── .env
├── generate_secret.py
├── README.md
├── requirements.txt
└── run.py
```

## Installation

1. **Prérequis**
   - Python 3.8+
   - pip

2. **Configuration de l'environnement**
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

3. **Configuration**
```bash
# Générer une nouvelle clé secrète et le fichier .env
python generate_secret.py
```

4. **Lancement**
```bash
python run.py
```

## Authentification

L'API utilise l'authentification JWT (JSON Web Token) avec le schéma Bearer.

> ⚠️ **Sécurité** : Les identifiants d'administration doivent être gardés secrets et changés régulièrement. Les exemples ci-dessous utilisent des identifiants fictifs à des fins de démonstration uniquement.

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant D as Database
    
    C->>A: POST /token (credentials)
    A->>D: Vérifier credentials
    D-->>A: Validation OK
    A-->>C: JWT Token
    
    C->>A: GET /verres (Bearer Token)
    A->>A: Vérifier JWT
    A->>D: Requête données
    D-->>A: Données
    A-->>C: Réponse JSON
```

### Obtention du token

```bash
curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=votre_email" \
     -d "password=votre_mot_de_passe"
```

Réponse :
```json
{
    "access_token": "eyJhbGciOiJIUzI1...",
    "token_type": "bearer"
}
```

### Bonnes pratiques de sécurité

1. **Gestion des identifiants**
   - Ne jamais partager ou exposer les identifiants d'administration
   - Utiliser des mots de passe forts
   - Changer régulièrement les mots de passe
   - Stocker les identifiants de manière sécurisée (variables d'environnement)

2. **Gestion des tokens**
   - Ne jamais stocker les tokens JWT en clair
   - Respecter leur durée de validité
   - Renouveler les tokens avant expiration
   - Ne pas transmettre les tokens via des canaux non sécurisés

## Points de terminaison

### Authentification

| Méthode | Endpoint | Description | Auth requise |
|---------|----------|-------------|--------------|
| POST | `/token` | Obtenir un token JWT | Non |

### Verres optiques

| Méthode | Endpoint | Description | Auth requise |
|---------|----------|-------------|--------------|
| GET | `/verres` | Liste des verres | Oui |
| GET | `/verres/{id}` | Détails d'un verre | Oui |
| GET | `/verres/search` | Recherche de verres | Oui |

```mermaid
graph LR
    A[Client] --> B[/token]
    A --> C[/verres]
    A --> D[/verres/{id}]
    A --> E[/verres/search]
    
    B --> F[JWT Token]
    C & D & E --> G[Auth Required]
```

## Modèles de données

### Verre
```json
{
    "id": 1,
    "nom": "string",
    "variante": "string",
    "hauteur_min": 0,
    "hauteur_max": 100,
    "indice": 1.5,
    "gravure": "string",
    "url_source": "string",
    "fournisseur": {
        "id": 1,
        "nom": "string"
    },
    "materiau": {
        "id": 1,
        "nom": "string"
    },
    "gamme": {
        "id": 1,
        "nom": "string"
    },
    "serie": {
        "id": 1,
        "nom": "string"
    },
    "traitements": [
        {
            "id": 1,
            "nom": "string",
            "type": "string"
        }
    ]
}
```

## Gestion des erreurs

| Code | Description |
|------|-------------|
| 200 | Succès |
| 401 | Non authentifié |
| 403 | Non autorisé |
| 404 | Ressource non trouvée |
| 422 | Erreur de validation |
| 500 | Erreur serveur |

```mermaid
graph TD
    A[Requête] --> B{Auth OK?}
    B -->|Non| C[401 Unauthorized]
    B -->|Oui| D{Ressource existe?}
    D -->|Non| E[404 Not Found]
    D -->|Oui| F{Validation OK?}
    F -->|Non| G[422 Unprocessable]
    F -->|Oui| H[200 Success]
```

## Tests

### Test avec curl

1. **Obtenir un token**
```bash
curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=votre_email" \
     -d "password=votre_mot_de_passe"
```

2. **Liste des verres**
```bash
curl -X GET "http://localhost:8000/verres" \
     -H "Authorization: Bearer votre_token"
```

3. **Recherche de verres**
```bash
curl -X GET "http://localhost:8000/verres/search?query=photochromique" \
     -H "Authorization: Bearer votre_token"
```

### Test avec Python

```python
import requests

# Configuration
BASE_URL = "http://localhost:8000"
credentials = {
    "username": "votre_email",
    "password": "votre_mot_de_passe"
}

# Obtenir le token
response = requests.post(f"{BASE_URL}/token", data=credentials)
token = response.json()["access_token"]

# Headers avec le token
headers = {"Authorization": f"Bearer {token}"}

# Test des endpoints
verres = requests.get(f"{BASE_URL}/verres", headers=headers)
print(verres.json())

recherche = requests.get(
    f"{BASE_URL}/verres/search",
    headers=headers,
    params={"query": "photochromique"}
)
print(recherche.json())
```

## Documentation OpenAPI

La documentation interactive OpenAPI (Swagger) est disponible à :
- http://localhost:8000/docs
- http://localhost:8000/redoc (format ReDoc)

> ⚠️ **Note de sécurité** : En production, assurez-vous de :
> - Désactiver la documentation Swagger si elle n'est pas nécessaire
> - Limiter l'accès à la documentation aux adresses IP autorisées
> - Ne jamais exposer les identifiants d'administration dans la documentation
> - Utiliser HTTPS pour toutes les communications

Cette documentation est générée automatiquement et inclut :
- Tous les endpoints avec leurs paramètres
- Les schémas de requête et réponse
- Les exemples de requêtes
- L'interface de test interactive 