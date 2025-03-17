# Outil de Collaboration pour la Description de Verres

## Objectif

Cet outil a été développé dans le cadre du projet de certification DevIA pour répondre à un besoin spécifique : collecter des descriptions textuelles de verres gravés afin de constituer une base de données d'apprentissage pour un modèle d'IA.

## Pourquoi cet outil ?

La création de cet outil répond à plusieurs besoins :

1. **Collecte de données** : Pour entraîner un modèle d'IA à reconnaître des verres à partir de descriptions, nous avons besoin d'un grand nombre de descriptions variées pour chaque verre.

2. **Approche collaborative** : En permettant à plusieurs personnes de contribuer, nous obtenons une diversité de descriptions qui enrichit notre jeu de données.

3. **Priorisation intelligente** : L'application présente en priorité les verres ayant le moins de descriptions, assurant ainsi une répartition équilibrée des données.

4. **Stockage sécurisé** : Les descriptions sont stockées dans une base de données Supabase avec des mécanismes de vérification d'intégrité.

## Fonctionnalités

- Interface utilisateur simple et intuitive
- Affichage des images de verres
- Formulaire pour ajouter des descriptions
- Affichage des descriptions existantes
- Navigation entre les différentes images
- Priorisation des verres les moins décrits
- Randomisation pour éviter la répétition
- Validation des descriptions (longueur minimale/maximale)
- Stockage sécurisé avec vérification d'intégrité

## Architecture technique

- **Frontend** : Application Streamlit pour l'interface utilisateur
- **Backend** : API Supabase pour le stockage des données
- **Stockage d'images** : Images stockées localement dans le dossier `../images/`
- **Sécurité** : Hachage SHA-256 pour vérifier l'intégrité des données

## Utilisation

1. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```

2. Configurer les secrets Supabase :
   - Créer un fichier `.streamlit/secrets.toml` avec les informations de connexion

3. Lancer l'application :
   ```
   streamlit run app_streamlit.py
   ```

## Perspectives

Cet outil de collecte de données est la première étape d'un projet plus large. Les descriptions collectées serviront ensuite à entraîner un modèle d'IA capable de reconnaître un verre à partir d'une description textuelle ou vocale, comme démontré dans le dossier `../poc_ia/`. 