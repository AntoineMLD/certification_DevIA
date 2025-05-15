# Historique des modifications de l'API

Ce document liste les modifications importantes apportées à l'API E3_MettreDispositionIA au fil des versions.

## 1.0.0 (Version actuelle)

Date de sortie : 2024-05-15

### Ajouts
- Endpoint `/token` pour l'authentification OAuth2 avec tokens JWT
- Endpoint `/embedding` pour générer des embeddings d'images
- Endpoint `/match` pour trouver les gravures similaires
- Endpoint `/validate_prediction` pour valider et améliorer les prédictions
- Endpoint `/search_tags` pour rechercher des verres par tags
- Endpoint `/verre/{verre_id}` pour obtenir les détails d'un verre
- Documentation OpenAPI complète
- Schéma de sécurité JWT avec rotation automatique des tokens
- Système de rate limiting pour prévenir les abus
- Journalisation des événements de sécurité
- Validation des fichiers images

### Changements techniques
- Utilisation de FastAPI pour l'implémentation de l'API
- Intégration du modèle EfficientNet pour la génération d'embeddings
- Calcul de similarité cosinus pour la recherche de correspondances
- Tests unitaires couvrant l'ensemble des endpoints

## Prochaines versions (planifiées)

### 1.1.0
- Support pour les formats d'image supplémentaires (SVG, TIFF)
- Amélioration des performances du modèle d'IA
- Ajout d'informations de contexte historique sur les gravures
- Documentation multilingue (Français, Anglais)

### 1.2.0
- API GraphQL alternative
- Système de cache pour les recherches fréquentes
- Support pour l'authentification par clé API
- Version mobile optimisée 