# Guide de Test de l'API France Optique

## Prérequis
1. Python installé
2. Environnement virtuel activé
3. Dépendances installées
4. API en cours d'exécution

## 1. Démarrage de l'API

```bash
# Activer l'environnement virtuel
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Générer la clé secrète
python generate_secret.py

# Lancer l'API
python run.py
```

## 2. Tests avec l'interface Swagger
1. Ouvrir dans le navigateur : http://localhost:8000/docs
2. Vous verrez l'interface interactive Swagger

## 3. Tests Étape par Étape

### 3.1 Obtenir un Token
1. Dans Swagger, cliquer sur `/token`
2. Cliquer sur "Try it out"
3. Entrer les identifiants :
   ```
   username: admin@france-optique.com
   password: admin123!@#
   ```
4. Cliquer sur "Execute"
5. Copier le token reçu

### 3.2 Tester la Liste des Verres
1. Cliquer sur `/verres`
2. Cliquer sur "Try it out"
3. Dans "Authorize", coller "Bearer votre_token"
4. Tester avec différentes valeurs de skip et limit
5. Cliquer sur "Execute"

### 3.3 Rechercher un Verre Spécifique
1. Cliquer sur `/verres/search`
2. Entrer un terme de recherche (ex: "photochromique")
3. Cliquer sur "Execute"

## 4. Tests avec cURL

### Obtenir un Token
```bash
curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin@france-optique.com&password=admin123!@#"
```

### Liste des Verres
```bash
curl -X GET "http://localhost:8000/verres?skip=0&limit=10" \
     -H "Authorization: Bearer votre_token"
```

### Recherche de Verres
```bash
curl -X GET "http://localhost:8000/verres/search?query=photochromique" \
     -H "Authorization: Bearer votre_token"
```

## 5. Tests avec Python

```python
import requests

# Configuration
BASE_URL = "http://localhost:8000"
admin_credentials = {
    "username": "admin@france-optique.com",
    "password": "admin123!@#"
}

# Obtenir le token
response = requests.post(
    f"{BASE_URL}/token",
    data=admin_credentials
)
token = response.json()["access_token"]

# Configuration des headers avec le token
headers = {
    "Authorization": f"Bearer {token}"
}

# Test : Liste des verres
verres = requests.get(
    f"{BASE_URL}/verres",
    headers=headers,
    params={"skip": 0, "limit": 10}
)
print("Liste des verres :", verres.json())

# Test : Recherche de verres
recherche = requests.get(
    f"{BASE_URL}/verres/search",
    headers=headers,
    params={"query": "photochromique"}
)
print("Résultats de recherche :", recherche.json())
```

## 6. Vérification des Résultats

Pour chaque test, vérifier :
1. Le code de statut (200 pour succès)
2. La structure des données reçues
3. La présence des champs attendus
4. La validité du token (durée de 30 minutes)

## 7. Tests d'Erreurs Courants

1. Token invalide :
```bash
curl -X GET "http://localhost:8000/verres" \
     -H "Authorization: Bearer token_invalide"
```

2. Token expiré (après 30 minutes)
3. Recherche sans paramètre
4. ID de verre inexistant 