import os
import sqlite3
import requests
from PIL import Image
from io import BytesIO
import hashlib
from bs4 import BeautifulSoup
import time
from pathlib import Path

class OpticalPipeline:
    """
    Pipeline pour traiter et sauvegarder les données des verres optiques.
    
    Cette classe gère le stockage des données dans SQLite.
    Les images seront téléchargées séparément par le script download_images.py
    """
    
    # Configuration de la base de données
    TABLE_NAME = "staging"
    DB_FOLDER = Path("..") / ".." / ".." / "Base_de_donnees"
    DB_NAME = "france_optique.db"
    
    def __init__(self):
        """Initialise le pipeline avec les configurations de base."""
        # S'assure que le dossier de la base de données existe
        self.DB_FOLDER.mkdir(parents=True, exist_ok=True)
        self.DB_PATH = str(self.DB_FOLDER / self.DB_NAME)
        self.create_database_table()

    def create_database_table(self):
        """Crée la table staging si elle n'existe pas."""
        query = """
        CREATE TABLE IF NOT EXISTS staging (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            glass_name TEXT,             -- Nom du verre
            range TEXT,                  -- Gamme
            series TEXT,                 -- Série
            variant TEXT,                -- Variante
            height TEXT,                 -- Hauteur
            protection_treatment TEXT,    -- Traitement de protection
            photochromic_treatment TEXT, -- Traitement photochromique
            material TEXT,               -- Matériau
            glass_index TEXT,            -- Indice de réfraction
            supplier TEXT,               -- Fournisseur
            engraving_url TEXT,          -- URL de la gravure
            source_url TEXT,             -- URL source
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            with sqlite3.connect(self.DB_PATH) as connection:
                cursor = connection.cursor()
                cursor.execute(query)
                connection.commit()
                
        except sqlite3.Error as error:
            print(f"❌ Erreur création table: {error}")
            raise

    def get_image_url(self, html_content):
        """
        Extrait l'URL de l'image depuis le HTML.
        
        Args:
            html_content (str): Code HTML contenant l'image
            
        Returns:
            str ou None: URL de l'image si trouvée
        """
        if not html_content or not isinstance(html_content, str):
            return None

        try:
            # Cas 1: C'est déjà une URL
            if html_content.startswith(('http://', 'https://')):
                return html_content

            # Cas 2: C'est du HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            img_tag = soup.find('img')
            
            if not img_tag or not img_tag.get('src'):
                return None
                
            # Récupère et nettoie l'URL
            url = img_tag['src']
            
            # Ajoute le protocole si nécessaire
            if url.startswith('//'):
                return 'https:' + url
            elif not url.startswith(('http://', 'https://')):
                return 'https://' + url.lstrip('/')
                
            return url

        except Exception:
            return None

    def save_to_database(self, item, spider):
        """Sauvegarde les données dans la base."""
        try:
            # Connexion à la base de données
            with sqlite3.connect(self.DB_PATH) as connection:
                cursor = connection.cursor()
                
                # Préparation de la requête d'insertion
                # On utilise glass_index au lieu de index (mot réservé SQL)
                query = f"""
                    INSERT INTO {self.TABLE_NAME} (
                        glass_name, material, glass_index,
                        supplier, engraving_url, source_url
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """
                
                # Valeurs à insérer
                values = (
                    item.get('glass_name'),
                    item.get('material'),
                    item.get('glass_index'),
                    item.get('supplier'),
                    item.get('engraving_url'),
                    item.get('source_url')
                )
                
                # Exécution de la requête
                cursor.execute(query, values)
                connection.commit()
                
            spider.logger.info(f"✅ Données sauvegardées: {item['source_url']}")
            return True
            
        except sqlite3.Error as error:
            spider.logger.error(f"❌ Erreur base de données: {error}")
            return False

    def process_item(self, item, spider):
        """
        Traite un nouvel item du spider.
        
        Cette méthode est appelée automatiquement par Scrapy pour chaque item trouvé.
        Elle extrait l'URL de l'image si présente et sauvegarde les données.
        
        Args:
            item (dict): Item à traiter
            spider: Spider Scrapy qui a trouvé l'item
            
        Returns:
            dict: Item traité
        """
        try:
            # 1. Extrait l'URL de l'image si présente
            if item.get('engraving_url'):
                item['engraving_url'] = self.get_image_url(item['engraving_url'])

            # 2. Sauvegarde en base de données
            self.save_to_database(item, spider)
            return item

        except Exception as error:
            spider.logger.error(f"❌ Erreur traitement item: {error}")
            return item

    def open_spider(self, spider):
        """Appelé quand le spider démarre."""
        spider.logger.info("🚀 Démarrage du pipeline")

    def close_spider(self, spider):
        """Appelé quand le spider termine."""
        spider.logger.info("✅ Pipeline terminé")
