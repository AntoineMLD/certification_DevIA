import sqlite3 
import json 
import os

# Définir le chemin absolu vers la base de données
# Utiliser un chemin relatif qui a fonctionné dans l'ancien script
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../E1_GestionDonnees/Base_de_donnees/france_optique.db"))

# Vérifier si le fichier existe au démarrage
if not os.path.exists(DB_PATH):
    print(f"ATTENTION: La base de données n'a pas été trouvée à: {DB_PATH}")
else:
    print(f"Base de données trouvée à: {DB_PATH}")

def find_matching_verres(tags):
    if not os.path.exists(DB_PATH):
        print(f"Erreur: La base de données n'existe pas à {DB_PATH}")
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Requête SQL améliorée pour récupérer plus d'informations
        cursor.execute("""
                      SELECT v.id, v.nom, v.variante, v.hauteur_min, v.hauteur_max, 
                            v.indice, v.gravure, v.url_source, f.nom as fournisseur_nom, t.tags
                      FROM verres v 
                      JOIN tags t ON v.id = t.verre_id
                      LEFT JOIN fournisseurs f ON v.fournisseur_id = f.id
                      """)
        
        verres = []
        for row in cursor.fetchall():
            verre_id, nom, variante, hauteur_min, hauteur_max, indice, gravure, url_source, fournisseur_nom, tags_json = row
            try:
                verre_tags = json.loads(tags_json or "[]")
                # Convertir tous les tags en minuscules pour la comparaison
                verre_tags_lower = [vt.strip().lower() for vt in verre_tags]
                search_tags_lower = [tag.strip().lower() for tag in tags]
                
                # Vérifier que TOUS les tags recherchés sont présents EXACTEMENT
                if all(tag in verre_tags_lower for tag in search_tags_lower):
                    verres.append({
                        "id": verre_id,
                        "nom": nom,
                        "variante": variante,
                        "hauteur_min": hauteur_min,
                        "hauteur_max": hauteur_max,
                        "indice": indice,
                        "gravure": gravure,
                        "url_source": url_source,
                        "fournisseur": fournisseur_nom,
                        "tags": verre_tags  # Garder les tags originaux dans la réponse
                    })
            except Exception as e:
                print(f"Erreur lors du traitement des tags pour le verre {verre_id}: {e}")
                continue

        conn.close()
        return verres
    except Exception as e:
        print(f"Erreur de connexion à la base de données: {e}")
        return []

def get_verre_details(verre_id):
    """
    Récupère les détails complets d'un verre avec les informations des tables liées
    (materiaux, series, traitements, verres_traitements)
    """
    if not os.path.exists(DB_PATH):
        print(f"Erreur: La base de données n'existe pas à {DB_PATH}")
        return None
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
        cursor = conn.cursor()
        
        # 1. Récupérer les informations de base du verre
        cursor.execute("""
            SELECT v.*, f.nom as fournisseur_nom, t.tags
            FROM verres v 
            LEFT JOIN tags t ON v.id = t.verre_id
            LEFT JOIN fournisseurs f ON v.fournisseur_id = f.id
            WHERE v.id = ?
        """, (verre_id,))
        
        verre_row = cursor.fetchone()
        if not verre_row:
            conn.close()
            return None
        
        # Convertir en dictionnaire
        verre = dict(verre_row)
        
        # Convertir les tags JSON en liste Python
        if verre.get("tags"):
            try:
                verre["tags"] = json.loads(verre["tags"])
            except:
                verre["tags"] = []
        
        # 2. Récupérer le matériau
        if verre.get("materiau_id"):
            cursor.execute("SELECT * FROM materiaux WHERE id = ?", (verre["materiau_id"],))
            materiau = cursor.fetchone()
            if materiau:
                verre["materiau"] = dict(materiau)
            
        # 3. Récupérer la série
        if verre.get("serie_id"):
            cursor.execute("SELECT * FROM series WHERE id = ?", (verre["serie_id"],))
            serie = cursor.fetchone()
            if serie:
                verre["serie"] = dict(serie)
        
        # 4. Récupérer les traitements associés
        cursor.execute("""
            SELECT t.*
            FROM traitements t
            JOIN verres_traitements vt ON t.id = vt.traitement_id
            WHERE vt.verre_id = ?
        """, (verre_id,))
        
        traitements = cursor.fetchall()
        if traitements:
            verre["traitements"] = [dict(traitement) for traitement in traitements]
        
        conn.close()
        return verre
    
    except Exception as e:
        print(f"Erreur lors de la récupération des détails du verre: {e}")
        if 'conn' in locals():
            conn.close()
        return None
