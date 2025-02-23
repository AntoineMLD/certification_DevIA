import logging
import re
from pathlib import Path
from typing import Any, Dict
import hashlib

import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text


class OpticalDataCleaner:
    """Classe pour nettoyer et enrichir les données de verres optiques."""

    def __init__(self, db_path: str):
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Connexion à la base de données
        self.engine = create_engine(f'sqlite:///{db_path}')

        # Valeurs par défaut
        self.DEFAULT_VALUES = {
            "glass_name": "INCONNU",
            "range": "STANDARD",
            "series": "INCONNU",
            "variant": "STANDARD",
            "min_height": 14,
            "max_height": 14,
            "protection_treatment": "NON",
            "photochromic_treatment": "NON",
            "material": "ORGANIQUE",
            "glass_index": 1.5,
            "supplier": "INCONNU",
            "engraving_url": "",
            "source_url": ""
        }

        # Mappings pour la standardisation
        self.MATERIAL_MAPPING = {
            "ORG": "ORGANIQUE",
            "Orma": "ORGANIQUE_ORMA",
            "Ormix": "ORGANIQUE_ORMIX",
            "Stylis": "ORGANIQUE_STYLIS",
            "Airwear": "POLYCARBONATE_AIRWEAR",
            "Lineis": "ORGANIQUE_LINEIS",
        }

        self.RANGE_MAPPING = {
            "Varilux": "PROGRESSIF_PREMIUM",
            "Eyezen": "UNIFOCAL_DIGITAL",
            "Essilor": "STANDARD",
        }

        self.SERIES_INFO = {
            "Comfort": {"type": "CONFORT", "niveau": "STANDARD"},
            "Physio": {"type": "PRECISION", "niveau": "PREMIUM"},
            "Liberty": {"type": "ECONOMIQUE", "niveau": "BASIQUE"},
            "XR": {"type": "INNOVATION", "niveau": "PREMIUM_PLUS"},
            "Digitime": {"type": "DIGITAL", "niveau": "SPECIALISE"},
        }

    def clean_html_content(self, html_content: str) -> str:
        """
        Nettoie le contenu HTML.
        
        Args:
            html_content (str): Contenu HTML à nettoyer
            
        Returns:
            str: Contenu nettoyé
        """
        if pd.isna(html_content):
            return ""
            
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text().strip()
        except Exception as error:
            self.logger.warning(f"Erreur nettoyage HTML: {error}")
            return ""

    def extract_image_url(self, html_str: str) -> str:
        """
        Extrait l'URL de l'image depuis une balise HTML.
        
        Args:
            html_str (str): Balise HTML contenant l'image
            
        Returns:
            str: URL de l'image ou valeur par défaut
        """
        if pd.isna(html_str):
            return self.DEFAULT_VALUES["engraving_url"]

        try:
            if "<img" in html_str:
                soup = BeautifulSoup(html_str, "html.parser")
                img_tag = soup.find("img")
                return img_tag["src"] if img_tag else self.DEFAULT_VALUES["engraving_url"]
        except Exception as error:
            self.logger.warning(f"Erreur extraction URL image: {error}")
            
        return self.DEFAULT_VALUES["engraving_url"]

    def clean_material(self, material: str) -> str:
        """
        Nettoie et standardise le matériau.
        
        Args:
            material (str): Matériau brut
            
        Returns:
            str: Matériau standardisé
        """
        if pd.isna(material):
            return self.DEFAULT_VALUES["material"]

        try:
            clean_material = self.clean_html_content(material)
            return self.MATERIAL_MAPPING.get(clean_material, self.DEFAULT_VALUES["material"])
        except Exception as error:
            self.logger.warning(f"Erreur nettoyage matériau: {error}")
            return self.DEFAULT_VALUES["material"]

    def analyze_glass_name(self, name: str) -> dict:
        """
        Analyse le nom du verre pour en extraire les caractéristiques.
        
        Args:
            name (str): Nom du verre
            
        Returns:
            dict: Caractéristiques extraites
        """
        if pd.isna(name):
            return {k: v for k, v in self.DEFAULT_VALUES.items() 
                   if k in ["glass_name", "range", "series", "variant", 
                           "min_height", "max_height", "protection_treatment", 
                           "photochromic_treatment"]}

        try:
            # Nettoyage du nom avec BeautifulSoup si c'est du HTML
            if "<" in name:
                name = BeautifulSoup(name, "html.parser").get_text().strip()

            # Extraction du nom de base
            composants = name.split(" ")
            nom_du_verre = composants[0]
            gamme = self.RANGE_MAPPING.get(nom_du_verre, self.DEFAULT_VALUES["range"])

            # Détection de la série
            serie = self.DEFAULT_VALUES["series"]
            for serie_connue in self.SERIES_INFO.keys():
                if serie_connue in name:
                    serie = serie_connue
                    break

            # Détection des traitements
            traitement_protection = "OUI" if any(t in name for t in ["Eye Protect System", "Blue Natural", "UVBlue"]) else "NON"
            traitement_photochromique = "OUI" if any(t in name for t in ["Transitions", "Trans®"]) else "NON"

            # Détection des variantes
            variantes = []
            if "Short" in name:
                variantes.append("COURT")
            if "Fit" in name or "FIT" in name:
                variantes.append("ADAPTATIF")

            variante = "|".join(variantes) if variantes else self.DEFAULT_VALUES["variant"]

            # Configuration des hauteurs
            hauteur_min = hauteur_max = 11 if "COURT" in variante else 14

            return {
                "glass_name": nom_du_verre,
                "range": gamme,
                "series": serie,
                "variant": variante,
                "min_height": hauteur_min,
                "max_height": hauteur_max,
                "protection_treatment": traitement_protection,
                "photochromic_treatment": traitement_photochromique,
            }

        except Exception as error:
            self.logger.warning(f"Erreur analyse nom: {error}")
            return {k: v for k, v in self.DEFAULT_VALUES.items() 
                   if k in ["glass_name", "range", "series", "variant", 
                           "min_height", "max_height", "protection_treatment", 
                           "photochromic_treatment"]}

    def clean_index(self, index_val: Any) -> list:
        """Nettoie et valide les valeurs d'indice.

        Args:
            index_val: Valeur d'indice brute

        Returns:
            list: Liste des indices nettoyés
        """
        if pd.isna(index_val):
            return [self.DEFAULT_VALUES["glass_index"]]

        if isinstance(index_val, (int, float)):
            return [float(index_val)]

        indices = []
        try:
            parts = re.split(r"[/\s-]", str(index_val))
            for part in parts:
                clean_part = part.strip().replace(",", ".")
                if not clean_part:
                    continue
                index = float(clean_part)
                if 1.4 <= index <= 1.9:  # Validation de l'indice
                    indices.append(index)

            return indices if indices else [self.DEFAULT_VALUES["glass_index"]]

        except (ValueError, TypeError):
            logging.warning(f"❌ Valeur d'indice invalide: {index_val}")
            return [self.DEFAULT_VALUES["glass_index"]]

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et enrichit le DataFrame.
        
        Args:
            df (pd.DataFrame): Données brutes
            
        Returns:
            pd.DataFrame: Données nettoyées
        """
        try:
            # 1. Copie du DataFrame original
            clean_df = pd.DataFrame()
            
            # 2. Traitement des noms de verre
            noms_traites = df["glass_name"].apply(self.analyze_glass_name)
            
            # 3. Attribution des colonnes analysées
            clean_df["glass_name"] = noms_traites.apply(lambda x: x["glass_name"])
            clean_df["range"] = noms_traites.apply(lambda x: x["range"])
            clean_df["series"] = noms_traites.apply(lambda x: x["series"])
            clean_df["variant"] = noms_traites.apply(lambda x: x["variant"])
            clean_df["min_height"] = noms_traites.apply(lambda x: x["min_height"])
            clean_df["max_height"] = noms_traites.apply(lambda x: x["max_height"])
            clean_df["protection_treatment"] = noms_traites.apply(lambda x: x["protection_treatment"])
            clean_df["photochromic_treatment"] = noms_traites.apply(lambda x: x["photochromic_treatment"])
            
            # 4. Nettoyage du matériau
            clean_df["material"] = df["material"].apply(self.clean_material)
            
            # 5. Nettoyage de l'indice
            clean_df["glass_index"] = df["glass_index"].apply(self.clean_index)
            # Duplication des lignes pour chaque indice
            clean_df = clean_df.explode("glass_index")
            clean_df = clean_df.rename(columns={"glass_index": "indice"})
            
            # 6. Nettoyage du fournisseur
            clean_df["supplier"] = df["glass_supplier_name"].fillna(self.DEFAULT_VALUES["supplier"])
            
            # 7. Nettoyage des URLs et gestion des images
            clean_df["gravure"] = df["nasal_engraving"].apply(
                lambda x: self.extract_image_url(x) or self.clean_html_content(x)
            )
            
            # 8. URL source
            clean_df["source_url"] = df["source_url"].fillna(self.DEFAULT_VALUES["source_url"])
            
            # 9. Validation des données
            clean_df.loc[~clean_df["indice"].between(1.4, 1.9), "indice"] = self.DEFAULT_VALUES["glass_index"]
            
            # 10. Renommage des colonnes en français
            colonnes_francaises = {
                "glass_name": "nom_du_verre",
                "range": "gamme",
                "series": "serie",
                "variant": "variante",
                "min_height": "hauteur_min",
                "max_height": "hauteur_max",
                "protection_treatment": "traitement_protection",
                "photochromic_treatment": "traitement_photochromique",
                "material": "materiau",
                "supplier": "fournisseur",
                "source_url": "url_source"
            }
            
            clean_df = clean_df.rename(columns=colonnes_francaises)
            
            # 11. Sélection des colonnes finales dans le bon ordre
            colonnes_finales = [
                "nom_du_verre", "gamme", "serie", "variante",
                "hauteur_min", "hauteur_max", "traitement_protection",
                "traitement_photochromique", "materiau", "indice",
                "fournisseur", "gravure", "url_source"
            ]
            
            return clean_df[colonnes_finales]

        except Exception as error:
            self.logger.error(f"❌ Erreur nettoyage DataFrame: {error}")
            raise

    def process_data(self, batch_size: int = 1000):
        """
        Traite les données de la table staging vers la table enhanced.
        
        Cette fonction:
        1. Crée la table enhanced
        2. Lit les données de staging
        3. Nettoie les données
        4. Sauvegarde dans enhanced
        """
        try:
            # 1. Création de la table enhanced
            self._create_enhanced_table()
            
            # 2. Lecture des données de staging
            query = """
            SELECT 
                source_url,              -- URL source
                glass_name,              -- Nom du verre
                nasal_engraving,         -- Gravure nasale
                glass_index,             -- Indice
                material,                -- Matériau
                glass_supplier_name,     -- Fournisseur
                image_engraving         -- Image de gravure
            FROM staging
            """
            
            # Charge toutes les données dans un DataFrame
            self.logger.info("📥 Chargement des données...")
            df = pd.read_sql_query(query, self.engine)
            
            if df.empty:
                self.logger.warning("⚠️ Aucune donnée trouvée dans staging")
                return
                
            total = len(df)
            self.logger.info(f"🔍 {total} enregistrements trouvés")
            
            # 3. Nettoyage des données
            self.logger.info("🧹 Nettoyage des données...")
            clean_df = self.clean_dataframe(df)
            
            # 4. Sauvegarde dans la table enhanced
            self.logger.info("💾 Sauvegarde des données...")
            clean_df.to_sql(
                name='enhanced',
                con=self.engine,
                if_exists='append',
                index=False,
                chunksize=batch_size
            )
            
            self.logger.info("✅ Traitement terminé avec succès")

        except Exception as error:
            self.logger.error(f"❌ Erreur pendant le traitement: {error}")
            raise

    def _create_enhanced_table(self):
        """Crée la table enhanced pour stocker les données nettoyées."""
        # On supprime d'abord la table si elle existe
        drop_query = """
        DROP TABLE IF EXISTS enhanced;
        """
        
        create_query = """
        CREATE TABLE IF NOT EXISTS enhanced (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom_du_verre TEXT,              -- Nom du verre
            gamme TEXT,                     -- Gamme du verre
            serie TEXT,                     -- Série du verre
            variante TEXT,                  -- Variante du verre
            hauteur_min INTEGER,            -- Hauteur minimale
            hauteur_max INTEGER,            -- Hauteur maximale
            traitement_protection TEXT,     -- Protection
            traitement_photochromique TEXT, -- Photochromique
            materiau TEXT,                  -- Matériau
            indice REAL,                    -- Indice de réfraction du verre
            fournisseur TEXT,               -- Fournisseur
            gravure TEXT,                   -- Gravure
            url_source TEXT,                -- URL source
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Exécute la suppression puis la création de la table
        with self.engine.connect() as conn:
            conn.execute(text(drop_query))
            conn.execute(text(create_query))
            conn.commit()  # Sauvegarde les changements

def main():
    """Point d'entrée du script."""
    try:
        # Configuration
        current_dir = Path(__file__).parent  # Obtient le dossier Base_de_donnees
        db_path = current_dir / "france_optique.db"  # Crée le chemin complet vers la base de données
        
        # Création et exécution du nettoyeur
        cleaner = OpticalDataCleaner(str(db_path))
        cleaner.process_data()
        
    except Exception as error:
        logging.error(f"❌ Erreur principale: {error}")
        raise

if __name__ == "__main__":
    main()
