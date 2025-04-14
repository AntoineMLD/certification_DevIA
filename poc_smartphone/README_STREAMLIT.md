# Application Streamlit pour dessin de gravures

Cette application permet de dessiner des symboles (cercle, triangle, losange) et de les enregistrer au format PNG 64x64 pixels pour entrainer le modèle de reconnaissance.

## Avantages

- Accessible depuis n'importe quel appareil avec un navigateur web (PC, smartphone, tablette)
- Interface intuitive pour dessiner des symboles
- Sauvegarde automatique des dessins au format 64x64 pixels
- Statistiques sur les symboles dessinés

## Installation

1. Assurez-vous d'avoir Python installé (version 3.7 ou supérieure).
2. Installez les dépendances:
   ```
   pip install -r requirements_streamlit.txt
   ```
   Ou simplement exécutez le script `start_streamlit_app.bat` qui installera les dépendances automatiquement.

## Utilisation

1. Lancez l'application:
   ```
   streamlit run streamlit_draw_app.py
   ```
   Ou simplement exécutez le script `start_streamlit_app.bat`.

2. Une fois l'application en cours d'exécution, vous pouvez y accéder:
   - Sur l'ordinateur local à l'adresse: http://localhost:8501
   - Depuis un smartphone ou une tablette: http://[IP_DE_VOTRE_PC]:8501
     (où [IP_DE_VOTRE_PC] est l'adresse IP de votre ordinateur sur le réseau local)

3. Utilisez l'interface pour:
   - Sélectionner le type de symbole (cercle, triangle, losange)
   - Régler la taille du pinceau
   - Dessiner le symbole sur le canvas
   - Enregistrer le dessin
   - Effacer le canvas pour dessiner un nouveau symbole

## Structure des données

Les dessins sont enregistrés dans le dossier `drawings/` organisé par type de symbole:
- `drawings/cercle/`
- `drawings/triangle/`
- `drawings/losange/`

Chaque fichier est au format PNG 64x64 pixels, nommé selon le format `{type}_{timestamp}.png`.

## Transfert des dessins vers le projet principal

Après avoir dessiné suffisamment de symboles, utilisez le script `copy_to_project_streamlit.bat` pour copier les dessins vers le projet principal E3_MettreDispositionIA.

## Entraînement du modèle

Une fois les dessins transférés vers le projet principal, suivez ces étapes:

1. Prétraiter les images:
   ```
   cd E3_MettreDispositionIA
   python -m app.prepare_data --input_dir data/raw_gravures --output_dir data/processed
   ```

2. Entraîner le modèle:
   ```
   python -m app.train --data_dir data/processed --output_dir model --num_epochs 10 --batch_size 16
   ```

3. Générer les embeddings:
   ```
   python -m app.generate_embeddings --images_dir data/processed --output_path embeddings/gravures_embeddings.pkl --model_path model/siamese_model.pt
   ```

## Accès à l'application depuis un smartphone

Pour accéder à l'application depuis votre smartphone:

1. Assurez-vous que votre PC et votre smartphone sont sur le même réseau WiFi.
2. Lancez l'application sur votre PC avec `start_streamlit_app.bat`.
3. Trouvez l'adresse IP de votre PC (exécutez `ipconfig` dans une invite de commande).
4. Sur votre smartphone, ouvrez un navigateur et accédez à http://[IP_DE_VOTRE_PC]:8501. 