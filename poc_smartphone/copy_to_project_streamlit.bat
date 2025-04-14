@echo off
echo Copie des dessins crees avec Streamlit vers le projet principal...

if not exist drawings (
    echo Aucun dessin trouve. Veuillez d'abord creer des dessins avec l'application Streamlit.
    goto :end
)

if not exist ..\E3_MettreDispositionIA\data\raw_gravures (
    mkdir ..\E3_MettreDispositionIA\data\raw_gravures
)

if exist drawings\cercle (
    if not exist ..\E3_MettreDispositionIA\data\raw_gravures\cercle (
        mkdir ..\E3_MettreDispositionIA\data\raw_gravures\cercle
    )
    xcopy /Y /Q drawings\cercle\*.png ..\E3_MettreDispositionIA\data\raw_gravures\cercle\
    echo Dessins de cercles copies.
)

if exist drawings\triangle (
    if not exist ..\E3_MettreDispositionIA\data\raw_gravures\triangle (
        mkdir ..\E3_MettreDispositionIA\data\raw_gravures\triangle
    )
    xcopy /Y /Q drawings\triangle\*.png ..\E3_MettreDispositionIA\data\raw_gravures\triangle\
    echo Dessins de triangles copies.
)

if exist drawings\losange (
    if not exist ..\E3_MettreDispositionIA\data\raw_gravures\losange (
        mkdir ..\E3_MettreDispositionIA\data\raw_gravures\losange
    )
    xcopy /Y /Q drawings\losange\*.png ..\E3_MettreDispositionIA\data\raw_gravures\losange\
    echo Dessins de losanges copies.
)

echo.
echo Les dessins ont ete copies vers le projet principal.
echo.
echo Etapes suivantes:
echo 1. Pretraiter les images: python -m app.prepare_data --input_dir data/raw_gravures --output_dir data/processed
echo 2. Entrainer le modele: python -m app.train --data_dir data/processed --output_dir model --num_epochs 10 --batch_size 16
echo 3. Generer les embeddings: python -m app.generate_embeddings --images_dir data/processed --output_path embeddings/gravures_embeddings.pkl --model_path model/siamese_model.pt

:end
pause 