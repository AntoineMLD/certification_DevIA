# Proof of Concept - Recherche d'image par description vocale

Ce POC (Proof of Concept) permet de rechercher une image de verre à partir d'une description textuelle ou vocale.

## Fonctionnement

1. L'application récupère toutes les descriptions de verres depuis Supabase
2. Elle utilise un modèle de traitement du langage (Sentence Transformers) pour convertir les descriptions en vecteurs
3. Lorsqu'un utilisateur entre une nouvelle description, celle-ci est également convertie en vecteur
4. L'application calcule la similarité entre la nouvelle description et toutes les descriptions existantes
5. Elle affiche les images correspondant aux descriptions les plus similaires

## Installation

1. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```

2. Configurer les variables d'environnement :
   - Créer un fichier `.streamlit/secrets.toml` avec les informations de connexion à Supabase :
     ```toml
     supabase_url = "https://rgmumgolnowpilenhdvb.supabase.co"
     supabase_key = "votre_clé_supabase"
     ```

## Utilisation

1. Lancer l'application :
   ```
   streamlit run recherche_image.py
   ```

2. Dans l'interface web :
   - Entrer une description du verre recherché dans la zone de texte
   - Ou utiliser la fonction de reconnaissance vocale (à implémenter)
   - Cliquer sur "Rechercher"

3. L'application affichera les images les plus similaires à la description fournie

## Améliorations futures

- Implémenter la reconnaissance vocale avec JavaScript
- Améliorer la précision en utilisant un modèle de langage plus performant
- Ajouter une option pour filtrer par type de verre
- Intégrer cette fonctionnalité à l'application principale 