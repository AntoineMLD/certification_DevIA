# Guide d'utilisation de la documentation API

Ce document explique comment utiliser et naviguer dans la documentation de l'API E3_MettreDispositionIA.

## Structure de la documentation

La documentation complète de l'API se compose des éléments suivants :

1. **README.md** - Vue d'ensemble de l'API, son architecture et ses fonctionnalités
2. **openapi.yaml** - Spécification OpenAPI/Swagger détaillant tous les endpoints
3. **USAGE.md** (ce fichier) - Guide d'utilisation de la documentation
4. **CHANGELOG.md** - Historique des modifications de l'API

## Consulter la documentation

### Option 1 : Interface Swagger (recommandée)

Lorsque l'API est en cours d'exécution, vous pouvez accéder à une interface interactive à l'adresse :

```
http://localhost:8000/docs
```

Cette interface vous permet de :
- Explorer tous les endpoints disponibles
- Voir les modèles de données et les schémas
- Tester les endpoints directement depuis le navigateur
- Visualiser les exemples de requêtes et de réponses

### Option 2 : Interface ReDoc

Une version alternative de la documentation est disponible à l'adresse :

```
http://localhost:8000/redoc
```

Cette interface est souvent plus lisible mais moins interactive que Swagger.

### Option 3 : Fichiers Markdown

Vous pouvez également consulter les fichiers Markdown directement :

- `README.md` pour une vue d'ensemble
- `USAGE.md` (ce fichier) pour l'utilisation
- `CHANGELOG.md` pour l'historique des versions

### Option 4 : Fichier OpenAPI

Le fichier `openapi.yaml` peut être importé dans divers outils :

- [Swagger Editor](https://editor.swagger.io/) - Copier/coller le contenu du fichier
- [Postman](https://www.postman.com/) - Importer le fichier pour tester l'API
- Générateurs de clients API - Pour créer des clients dans différents langages

## Utilisation de l'API

### Authentification

Pour utiliser l'API, vous devez d'abord obtenir un token JWT :

1. Appelez l'endpoint `/token` avec vos identifiants
2. Enregistrez le token renvoyé
3. Utilisez ce token dans l'en-tête `Authorization` de vos requêtes

Exemple avec cURL :

```bash
# Obtenir un token
TOKEN=$(curl -s -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@example.com&password=admin_password" | jq -r '.access_token')

# Utiliser le token
curl -X POST "http://localhost:8000/match" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@image.png"
```

### Exemples de scénarios courants

1. **Analyser une image et trouver des correspondances**
   - Endpoint: `/match`
   - Méthode: POST
   - Corps: Fichier image
   - Authentification: Requise

2. **Rechercher des verres par tags**
   - Endpoint: `/search_tags`
   - Méthode: POST
   - Corps: Liste de tags
   - Authentification: Requise

3. **Consulter les détails d'un verre**
   - Endpoint: `/verre/{verre_id}`
   - Méthode: GET
   - Paramètre de chemin: ID du verre
   - Authentification: Requise

## Compatibilité avec les lecteurs d'écran

La documentation a été conçue pour être compatible avec les lecteurs d'écran :

- Utilisation de balises d'en-tête appropriées
- Structure hiérarchique claire
- Descriptions textuelles pour tous les éléments
- Tables des matières navigables

Si vous rencontrez des problèmes d'accessibilité avec cette documentation, veuillez nous contacter à l'adresse support@example.com.

## Résolution des problèmes courants

- **401 Unauthorized** : Votre token est manquant, expiré ou invalide
- **400 Bad Request** : Format de requête incorrect
- **429 Too Many Requests** : Vous avez dépassé le nombre de requêtes autorisées
- **500 Internal Server Error** : Erreur côté serveur, contactez l'administrateur

## Besoin d'aide supplémentaire ?

Si vous avez besoin d'aide supplémentaire, vous pouvez :

1. Consulter les exemples de code dans les fichiers de documentation
2. Examiner les modèles de données dans la spécification OpenAPI
3. Contacter l'équipe de support à support@example.com 