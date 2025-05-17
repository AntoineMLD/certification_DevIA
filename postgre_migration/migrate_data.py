import sqlite3
import psycopg2
from psycopg2.extras import execute_values
import logging
import os
import re
from dotenv import load_dotenv
import json

# Configuration du logging plus détaillé
logging.basicConfig(
    level=logging.DEBUG,  # Changé en DEBUG pour plus de détails
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Chargemment des variables d'environnement
logging.info("Chargement du fichier .env...")
load_dotenv()
logging.info("Variables d'environnement chargées")

# Connexion à SQLite
def connect_sqlite():
    try:
        # Récupération de la connexion depuis le .env
        logging.info("Tentative de récupération de sqlite_conn depuis .env")
        sqlite_conn_str = os.getenv('sqlite_conn')
        logging.debug(f"Valeur de sqlite_conn: {sqlite_conn_str}")
        
        if not sqlite_conn_str:
            raise Exception("La variable sqlite_conn n'est pas définie dans le .env")
        
        # Ajustement du chemin relatif
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'E1_GestionDonnees', 'Base_de_donnees', 'france_optique.db')
        logging.info(f"Chemin ajusté de la base de données: {db_path}")
        
        # Création de la connexion avec le bon chemin
        sqlite_conn = sqlite3.connect(db_path)
        sqlite_conn.row_factory = sqlite3.Row
        logging.info("Connexion à SQLite réussie")
        return sqlite_conn
    except Exception as e:
        logging.error(f"Erreur de connexion à SQLite: {str(e)}")
        logging.error(f"Type d'erreur: {type(e)}")
        raise

# Connexion à PostgreSQL
def connect_postgres():
    try:
        pg_conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB', 'glass_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        logging.info("Connexion à PostgreSQL réussie")
        return pg_conn
    except psycopg2.Error as e:
        logging.error(f"Erreur de connexion à PostgreSQL: {e}")
        raise

def migrate_reference_table(sqlite_cur, pg_cur, table_name):
    """Migre une table de référence (fournisseurs, materiaux, etc.)"""
    try:
        # Récupération des données SQLite
        sqlite_cur.execute(f"SELECT * FROM {table_name}")
        rows = sqlite_cur.fetchall()
        
        if not rows:
            logging.warning(f"Aucune donnée trouvée dans la table {table_name}")
            return
        
        # Insertion dans PostgreSQL
        columns = [key for key in dict(rows[0]).keys()]
        values = [[row[col] for col in columns] for row in rows]
        
        # Construction de la requête d'insertion
        columns_str = ', '.join(columns)
        placeholders = ','.join(['(' + ','.join(['%s'] * len(columns)) + ')'] * len(values))
        
        insert_query = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES {placeholders}
        ON CONFLICT DO NOTHING
        """
        
        # Aplatir la liste de valeurs
        flat_values = [val for row in values for val in row]
        
        logging.info(f"Tentative d'insertion de {len(rows)} lignes dans {table_name}")
        logging.debug(f"Requête: {insert_query}")
        logging.debug(f"Nombre de colonnes: {len(columns)}")
        logging.debug(f"Nombre de valeurs: {len(flat_values)}")
        
        pg_cur.execute(insert_query, flat_values)
        logging.info(f"Migration de {len(rows)} lignes vers {table_name} réussie")
        
    except (sqlite3.Error, psycopg2.Error) as e:
        logging.error(f"Erreur lors de la migration de {table_name}: {str(e)}")
        logging.error(f"Type d'erreur: {type(e)}")
        raise

def migrate_verres(sqlite_cur, pg_cur):
    """Migre la table verres avec gestion des clés étrangères"""
    try:
        # Vérifions d'abord la structure de la table
        sqlite_cur.execute("PRAGMA table_info(verres)")
        columns = sqlite_cur.fetchall()
        logging.debug("Structure de la table verres dans SQLite:")
        for col in columns:
            logging.debug(f"Colonne: {col['name']}, Type: {col['type']}")

        # Requête simplifiée car les IDs sont déjà présents
        sqlite_cur.execute("""
            SELECT id, nom, variante, hauteur_min, hauteur_max, 
                   indice, gravure, url_source,
                   fournisseur_id, materiau_id, gamme_id, serie_id
            FROM verres
        """)
        rows = sqlite_cur.fetchall()
        
        if not rows:
            logging.warning("Aucun verre trouvé dans la table source")
            return
            
        logging.info(f"Nombre de verres trouvés: {len(rows)}")
        
        # Insertion par lots de 100 verres
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            values = []
            for row in batch:
                values.extend([
                    row['nom'], row['variante'], row['hauteur_min'], row['hauteur_max'],
                    row['indice'], row['gravure'], row['url_source'],
                    row['fournisseur_id'], row['materiau_id'], 
                    row['gamme_id'], row['serie_id']
                ])
            
            placeholders = ','.join(['(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'] * len(batch))
            
            insert_query = f"""
                INSERT INTO verres (
                    nom, variante, hauteur_min, hauteur_max, 
                    indice, gravure, url_source,
                    fournisseur_id, materiau_id, gamme_id, serie_id
                ) VALUES {placeholders}
                ON CONFLICT DO NOTHING
            """
            
            pg_cur.execute(insert_query, values)
            logging.info(f"Migration du lot de {len(batch)} verres réussie")
        
        logging.info(f"Migration totale de {len(rows)} verres réussie")
        
    except (sqlite3.Error, psycopg2.Error) as e:
        logging.error(f"Erreur lors de la migration des verres: {str(e)}")
        logging.error(f"Type d'erreur: {type(e)}")
        raise

def migrate_staging(sqlite_cur, pg_cur):
    """Migre les données vers la table staging"""
    try:
        sqlite_cur.execute("""
            SELECT DISTINCT  -- Pour éviter les doublons
                   v.url_source as source_url,
                   v.nom as glass_name,
                   v.gravure as nasal_engraving,
                   CAST(v.indice as TEXT) as glass_index,
                   m.nom as material,
                   f.nom as glass_supplier_name,
                   v.gravure as image_engraving
            FROM verres v
            LEFT JOIN materiaux m ON v.materiau_id = m.id
            LEFT JOIN fournisseurs f ON v.fournisseur_id = f.id
            WHERE v.url_source IS NOT NULL  -- Pour éviter les lignes sans source
        """)
        rows = sqlite_cur.fetchall()
        
        if not rows:
            logging.warning("Aucune donnée trouvée pour la table staging")
            return
            
        logging.info(f"Nombre de lignes trouvées pour staging: {len(rows)}")
        
        # Insertion par lots
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            values = []
            for row in batch:
                values.extend([
                    row['source_url'], row['glass_name'], row['nasal_engraving'],
                    row['glass_index'], row['material'], row['glass_supplier_name'],
                    row['image_engraving']
                ])
            
            placeholders = ','.join(['(%s, %s, %s, %s, %s, %s, %s)'] * len(batch))
            
            insert_query = f"""
                INSERT INTO staging (
                    source_url, glass_name, nasal_engraving,
                    glass_index, material, glass_supplier_name,
                    image_engraving
                ) VALUES {placeholders}
            """
            
            pg_cur.execute(insert_query, values)
            logging.info(f"Migration du lot de {len(batch)} lignes vers staging réussie")
        
        logging.info(f"Migration totale de {len(rows)} lignes vers staging réussie")
        
    except (sqlite3.Error, psycopg2.Error) as e:
        logging.error(f"Erreur lors de la migration vers staging: {str(e)}")
        logging.error(f"Type d'erreur: {type(e)}")
        raise

def migrate_enhanced(sqlite_cur, pg_cur):
    """Migre les données vers la table enhanced"""
    try:
        sqlite_cur.execute("""
            SELECT DISTINCT  -- Pour éviter les doublons
                   v.nom as nom_du_verre,
                   g.nom as gamme,
                   s.nom as serie,
                   v.variante,
                   CAST(COALESCE(v.hauteur_min, 0) as INTEGER) as hauteur_min,
                   CAST(COALESCE(v.hauteur_max, 0) as INTEGER) as hauteur_max,
                   '' as traitement_protection,
                   '' as traitement_photochromique,
                   m.nom as materiau,
                   CAST(COALESCE(v.indice, 0.0) as REAL) as indice,
                   f.nom as fournisseur,
                   v.gravure,
                   v.url_source
            FROM verres v
            LEFT JOIN materiaux m ON v.materiau_id = m.id
            LEFT JOIN fournisseurs f ON v.fournisseur_id = f.id
            LEFT JOIN gammes g ON v.gamme_id = g.id
            LEFT JOIN series s ON v.serie_id = s.id
            WHERE v.nom IS NOT NULL  -- Pour éviter les lignes sans nom
        """)
        rows = sqlite_cur.fetchall()
        
        if not rows:
            logging.warning("Aucune donnée trouvée pour la table enhanced")
            return
            
        logging.info(f"Nombre de lignes trouvées pour enhanced: {len(rows)}")
        
        # Insertion par lots
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            values = []
            for row in batch:
                values.extend([
                    row['nom_du_verre'], row['gamme'], row['serie'],
                    row['variante'], row['hauteur_min'], row['hauteur_max'],
                    row['traitement_protection'], row['traitement_photochromique'],
                    row['materiau'], row['indice'], row['fournisseur'],
                    row['gravure'], row['url_source']
                ])
            
            placeholders = ','.join(['(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'] * len(batch))
            
            insert_query = f"""
                INSERT INTO enhanced (
                    nom_du_verre, gamme, serie, variante,
                    hauteur_min, hauteur_max,
                    traitement_protection, traitement_photochromique,
                    materiau, indice, fournisseur,
                    gravure, url_source
                ) VALUES {placeholders}
            """
            
            pg_cur.execute(insert_query, values)
            logging.info(f"Migration du lot de {len(batch)} lignes vers enhanced réussie")
        
        logging.info(f"Migration totale de {len(rows)} lignes vers enhanced réussie")
        
    except (sqlite3.Error, psycopg2.Error) as e:
        logging.error(f"Erreur lors de la migration vers enhanced: {str(e)}")
        logging.error(f"Type d'erreur: {type(e)}")
        raise

def migrate_all():
    """Fonction principale de migration"""
    sqlite_conn = None
    pg_conn = None
    
    try:
        sqlite_conn = connect_sqlite()
        pg_conn = connect_postgres()
        
        sqlite_cur = sqlite_conn.cursor()
        pg_cur = pg_conn.cursor()
        
        # 1. Migration des données de base vers staging et enhanced
        logging.info("Début de la migration vers staging...")
        migrate_staging(sqlite_cur, pg_cur)
        pg_conn.commit()
        
        logging.info("Début de la migration vers enhanced...")
        migrate_enhanced(sqlite_cur, pg_cur)
        pg_conn.commit()
        
        # 2. Migration des tables de référence
        logging.info("Début de la migration des tables de référence...")
        
        # Extraction des valeurs uniques pour les tables de référence
        sqlite_cur.execute("""
            SELECT DISTINCT f.nom 
            FROM verres v
            JOIN fournisseurs f ON v.fournisseur_id = f.id
            WHERE f.nom IS NOT NULL
        """)
        fournisseurs = [row[0] for row in sqlite_cur.fetchall()]
        
        sqlite_cur.execute("""
            SELECT DISTINCT m.nom 
            FROM verres v
            JOIN materiaux m ON v.materiau_id = m.id
            WHERE m.nom IS NOT NULL
        """)
        materiaux = [row[0] for row in sqlite_cur.fetchall()]
        
        sqlite_cur.execute("""
            SELECT DISTINCT g.nom 
            FROM verres v
            JOIN gammes g ON v.gamme_id = g.id
            WHERE g.nom IS NOT NULL
        """)
        gammes = [row[0] for row in sqlite_cur.fetchall()]
        
        sqlite_cur.execute("""
            SELECT DISTINCT s.nom 
            FROM verres v
            JOIN series s ON v.serie_id = s.id
            WHERE s.nom IS NOT NULL
        """)
        series = [row[0] for row in sqlite_cur.fetchall()]
        
        logging.info(f"Nombre de fournisseurs trouvés : {len(fournisseurs)}")
        logging.info(f"Nombre de matériaux trouvés : {len(materiaux)}")
        logging.info(f"Nombre de gammes trouvées : {len(gammes)}")
        logging.info(f"Nombre de séries trouvées : {len(series)}")
        
        # Insertion dans les tables de référence
        for fournisseur in fournisseurs:
            pg_cur.execute("INSERT INTO fournisseurs (nom) VALUES (%s) ON CONFLICT (nom) DO NOTHING", (fournisseur,))
        
        for materiau in materiaux:
            pg_cur.execute("INSERT INTO materiaux (nom) VALUES (%s) ON CONFLICT (nom) DO NOTHING", (materiau,))
        
        for gamme in gammes:
            pg_cur.execute("INSERT INTO gammes (nom) VALUES (%s) ON CONFLICT (nom) DO NOTHING", (gamme,))
        
        for serie in series:
            pg_cur.execute("INSERT INTO series (nom) VALUES (%s) ON CONFLICT (nom) DO NOTHING", (serie,))
        
        # Création des traitements de base
        traitements_base = [
            ('Anti-reflet', 'protection'),
            ('Anti-UV', 'protection'),
            ('Photochromique', 'photochromique')
        ]
        for nom, type_traitement in traitements_base:
            pg_cur.execute("""
                INSERT INTO traitements (nom, type)
                VALUES (%s, %s)
                ON CONFLICT ON CONSTRAINT traitements_nom_key DO NOTHING
            """, (nom, type_traitement))
        
        pg_conn.commit()
        
        # 3. Migration de la table principale verres avec les clés étrangères
        logging.info("Début de la migration de la table verres...")
        sqlite_cur.execute("""
            SELECT v.id, v.nom, v.variante, v.hauteur_min, v.hauteur_max,
                   v.indice, v.gravure, v.url_source,
                   v.fournisseur_id, v.materiau_id, v.gamme_id, v.serie_id
            FROM verres v
            WHERE v.nom IS NOT NULL
            AND v.fournisseur_id IS NOT NULL
            AND v.materiau_id IS NOT NULL
            AND v.gamme_id IS NOT NULL
            AND v.serie_id IS NOT NULL
        """)
        verres = sqlite_cur.fetchall()
        
        if not verres:
            logging.error("Aucun verre trouvé dans la table source avec toutes les références requises")
            return
            
        logging.info(f"Nombre de verres trouvés avec toutes les références: {len(verres)}")
        
        for verre in verres:
            try:
                # Insertion dans la table verres
                pg_cur.execute("""
                    INSERT INTO verres (
                        nom, variante, hauteur_min, hauteur_max,
                        indice, gravure, url_source,
                        fournisseur_id, materiau_id, gamme_id, serie_id,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                        CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                        CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                    RETURNING id
                """, (
                    verre['nom'], verre['variante'],
                    verre['hauteur_min'], verre['hauteur_max'],
                    verre['indice'], verre['gravure'], verre['url_source'],
                    verre['fournisseur_id'], verre['materiau_id'], 
                    verre['gamme_id'], verre['serie_id']
                ))
                
                # Récupération de l'ID du verre inséré
                result = pg_cur.fetchone()
                if not result:
                    logging.error(f"Échec de l'insertion du verre {verre['nom']}")
                    continue
                    
                verre_id = result[0]
                
                # Création des tags
                try:
                    # Vérification et récupération des données de référence
                    pg_cur.execute("SELECT nom FROM fournisseurs WHERE id = %s", (verre['fournisseur_id'],))
                    f = pg_cur.fetchone()
                    if not f:
                        logging.warning(f"Fournisseur non trouvé pour l'ID {verre['fournisseur_id']}")
                        continue

                    pg_cur.execute("SELECT nom FROM materiaux WHERE id = %s", (verre['materiau_id'],))
                    m = pg_cur.fetchone()
                    if not m:
                        logging.warning(f"Matériau non trouvé pour l'ID {verre['materiau_id']}")
                        continue

                    pg_cur.execute("SELECT nom FROM gammes WHERE id = %s", (verre['gamme_id'],))
                    g = pg_cur.fetchone()
                    if not g:
                        logging.warning(f"Gamme non trouvée pour l'ID {verre['gamme_id']}")
                        continue

                    pg_cur.execute("SELECT nom FROM series WHERE id = %s", (verre['serie_id'],))
                    s = pg_cur.fetchone()
                    if not s:
                        logging.warning(f"Série non trouvée pour l'ID {verre['serie_id']}")
                        continue

                    tags_data = {
                        'nom': verre['nom'],
                        'fournisseur': f[0],
                        'materiau': m[0],
                        'indice': str(verre['indice']) if verre['indice'] else None,
                        'gamme': g[0],
                        'serie': s[0]
                    }

                    pg_cur.execute("""
                        INSERT INTO tags (verre_id, tags, created_at)
                        VALUES (%s, %s::jsonb, CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                    """, (verre_id, json.dumps(tags_data)))

                except Exception as e:
                    logging.error(f"Erreur lors de la création des tags pour le verre {verre['nom']}: {str(e)}")
                    continue

            except Exception as e:
                logging.error(f"Erreur lors du traitement du verre {verre['nom']}: {str(e)}")
                continue
        
        pg_conn.commit()
        logging.info("Migration complète terminée avec succès")
        
        # Création des index pour optimiser les performances
        logging.info("Création des index...")
        pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_verres_nom ON verres(nom)")
        pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_verres_fournisseur ON verres(fournisseur_id)")
        pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_verres_materiau ON verres(materiau_id)")
        pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_verres_gamme ON verres(gamme_id)")
        pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_verres_serie ON verres(serie_id)")
        pg_conn.commit()
        logging.info("Création des index terminée")
        
    except Exception as e:
        if pg_conn:
            pg_conn.rollback()
        logging.error(f"Erreur lors de la migration: {str(e)}")
        raise
        
    finally:
        if sqlite_conn:
            sqlite_conn.close()
        if pg_conn:
            pg_conn.close()

if __name__ == "__main__":
    try:
        migrate_all()
    except Exception as e:
        logging.error(f"Erreur fatale: {e}") 