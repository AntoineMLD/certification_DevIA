# Tests pour l'API E3_MettreDispositionIA

Ce répertoire contient des tests automatisés pour l'API E3_MettreDispositionIA. Les tests couvrent tous les points de terminaison et les fonctionnalités de sécurité.

## Fichiers de tests

- `test_api.py` : Tests des points de terminaison de l'API
- `test_security.py` : Tests des fonctionnalités de sécurité (authentification, tokens JWT)
- `test_model.py` : Tests des fonctionnalités du modèle d'IA (embeddings, similarité)
- `conftest.py` : Fixtures partagées pour tous les tests

## Structure des tests

Les tests sont organisés par fonctionnalité :

1. **Tests des points de terminaison** : Vérifient que chaque endpoint retourne les réponses attendues
2. **Tests de sécurité** : Vérifient l'authentification, la validation des tokens et la protection des endpoints
3. **Tests du modèle** : Vérifient le chargement du modèle, la génération d'embeddings et la recherche de similarités

## Comment exécuter les tests

### Prérequis

- Python 3.8+
- pytest
- Toutes les dépendances listées dans `requirements.txt`

### Installation des dépendances

```bash
pip install -r requirements.txt
pip install pytest pytest-mock
```

### Exécution des tests

Depuis le répertoire racine de l'API (`E3_MettreDispositionIA/main/api`), exécutez :

```bash
# Exécuter tous les tests
pytest tests/

# Exécuter un fichier de tests spécifique
pytest tests/test_api.py

# Exécuter un test spécifique
pytest tests/test_api.py::test_token_endpoint_success

# Exécuter avec des détails
pytest tests/ -v

# Exécuter avec un rapport de couverture
pytest tests/ --cov=api
```

## Couverture des tests

Les tests couvrent :

### Points de terminaison API

- `/token` : Authentification et obtention de token
- `/embedding` : Calcul d'embedding pour une image
- `/match` : Recherche de correspondances pour une image
- `/validate_prediction` : Validation d'une prédiction
- `/search_tags` : Recherche de verres par tags
- `/verre/{verre_id}` : Récupération des détails d'un verre

### Sécurité

- Création et vérification de tokens JWT
- Validation des fichiers images
- Journalisation des événements de sécurité
- Limitation du taux d'appels (rate limiting)
- Rotation des tokens

### Modèle d'IA

- Chargement du modèle
- Prétraitement des images
- Génération d'embeddings
- Recherche de similarités

## Stratégie de mocking

Les tests utilisent des mocks pour isoler les composants et éviter les dépendances externes :

- Les appels au modèle d'IA sont mockés pour éviter de charger un modèle réel
- Les embeddings de référence sont générés de manière aléatoire
- Les fonctions de sécurité sont mockées avec des clés de test

## Ajout de nouveaux tests

Pour ajouter de nouveaux tests :

1. Identifiez la catégorie appropriée (API, sécurité, modèle)
2. Ajoutez une fonction de test dans le fichier correspondant
3. Utilisez les fixtures existantes ou créez-en de nouvelles si nécessaire
4. Suivez le modèle "Arrange-Act-Assert" :
   - Arrangez les données et les mocks
   - Exécutez l'action à tester
   - Vérifiez les résultats 