import os
import logging
import requests
from PIL import Image
from io import BytesIO
import sqlite3
from pathlib import Path
from datetime import datetime
import re
import hashlib

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "france_optique.db"
# Les images seront stockées dans le même dossier que la base de données
MEDIA_PATH = Path("images")

def setup_media_folder():
    """Crée le dossier des images s'il n'existe pas."""
    # Crée une structure organisée par année/mois
    current_date = datetime.now()
    image_folder = MEDIA_PATH / str(current_date.year) / f"{current_date.month:02d}"
    image_folder.mkdir(parents=True, exist_ok=True)
    return image_folder

def get_images_to_download():
    """Récupère les URLs des images à télécharger depuis la base de données."""
    query = """
    SELECT 
        id,
        glass_name,
        glass_index,
        nasal_engraving as engraving_url
    FROM staging 
    WHERE nasal_engraving != ''
    AND nasal_engraving IS NOT NULL
    """
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        return cursor.execute(query).fetchall()

def generate_image_name(row):
    """Crée un nom de fichier unique pour l'image.
    
    Format: ID_NomVerre_Indice.png
    Exemple: 42_varilux_150.png
    """
    # 1. Nettoie le nom du verre (enlève les caractères spéciaux)
    nom_verre = ''.join(c.lower() for c in row['glass_name'] if c.isalnum())
    
    # 2. Formate l'indice (exemple: 1.50 -> 150)
    indice = str(row['glass_index']).replace('.', '')
    
    # 3. Crée le nom final
    return f"{row['id']}_{nom_verre}_{indice}.png"

def download_and_enhance_image(url, image_path):
    """Télécharge et améliore une image."""
    try:
        # Téléchargement de l'image
        response = requests.get(url)
        response.raise_for_status()
        
        # Ouverture de l'image
        image = Image.open(BytesIO(response.content))
        
        # Conversion en PNG avec fond transparent
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Amélioration de la qualité
        enhanced = image.resize(
            (image.width * 2, image.height * 2),
            Image.Resampling.LANCZOS
        )
        
        # Sauvegarde de l'image
        enhanced.save(image_path, 'PNG', quality=95)
        logger.info(f"✅ Image sauvegardée: {image_path.name}")
        return True
        
    except Exception as error:
        logger.error(f"❌ Erreur téléchargement {url}: {error}")
        return False

def update_gravure(image_id, gravure_path):
    """Met à jour le chemin de l'image dans la base de données."""
    query = """
    UPDATE enhanced
    SET gravure = ?
    WHERE id = ?
    """
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(gravure_path), image_id))
            conn.commit()
    except Exception as error:
        logger.error(f"❌ Erreur lors de la mise à jour de la gravure: {error}")
        return False
    return True

def main():
    """Point d'entrée du script."""
    try:
        # Création du dossier des images
        image_folder = Path("images")
        image_folder.mkdir(exist_ok=True)
        
        # Connexion à la base de données
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
            cursor = conn.cursor()
            
            # Récupération des URLs d'images et informations nécessaires
            cursor.execute("""
                SELECT e.id, e.gravure, e.nom_du_verre, e.indice
                FROM enhanced e
                WHERE e.gravure LIKE 'http%'
            """)
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("Aucune image à télécharger")
                return
                
            logger.info(f"🔍 {len(rows)} images à télécharger")
            
            # Traitement de chaque URL
            for row in rows:
                # Génération du nom de fichier avec l'ancien format
                nom_verre = ''.join(c.lower() for c in row['nom_du_verre'] if c.isalnum())
                indice = str(row['indice']).replace('.', '')
                image_name = f"{row['id']}_{nom_verre}_{indice}.png"
                image_path = image_folder / image_name
                
                # Vérification si l'image existe déjà
                if image_path.exists():
                    logger.info(f"⏩ Image déjà existante: {image_name}")
                    continue
                
                # Téléchargement et amélioration de l'image
                if download_and_enhance_image(row['gravure'], image_path):
                    # Mise à jour du chemin dans la base de données
                    relative_path = str(image_path)
                    update_gravure(row['id'], relative_path)
                    
            logger.info("✨ Traitement terminé")
            
    except Exception as error:
        logger.error(f"❌ Erreur principale: {error}")
        raise

if __name__ == "__main__":
    main() 