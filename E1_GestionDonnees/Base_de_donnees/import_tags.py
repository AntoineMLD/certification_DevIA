import json
import sqlite3
import os

def import_tags():
    # Connexion à la base de données
    conn = sqlite3.connect('france_optique.db')
    cursor = conn.cursor()

    # Vérifier si la table tags existe déjà
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tags'")
    if not cursor.fetchone():
        print("Création de la table tags...")
        # Création de la table tags
        cursor.execute('''
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verre_id INTEGER,
            tags TEXT,  -- Stockage des tags sous forme de liste JSON
            FOREIGN KEY (verre_id) REFERENCES verres(id)
        )
        ''')
        conn.commit()
        print("Table tags créée avec succès!")
    else:
        print("La table tags existe déjà")

    try:
        # D'abord, ajouter les verres sans gravure https
        print("Recherche des verres sans gravure https...")
        cursor.execute("""
            SELECT id, gravure 
            FROM verres 
            WHERE gravure NOT LIKE '%https%' 
            AND gravure IS NOT NULL 
            AND gravure != ''
        """)
        
        verres_sans_gravure = cursor.fetchall()
        print(f"Nombre de verres sans gravure https trouvés : {len(verres_sans_gravure)}")
        
        for verre_id, gravure in verres_sans_gravure:
            try:
                # Vérifier si le verre a déjà des tags
                cursor.execute("SELECT id FROM tags WHERE verre_id = ?", (verre_id,))
                if not cursor.fetchone():
                    # Créer une liste avec la gravure comme tag
                    tags = [gravure]
                    tags_json = json.dumps(tags)
                    print(f"Tentative d'insertion pour le verre {verre_id} avec les tags: {tags_json}")
                    cursor.execute(
                        "INSERT INTO tags (verre_id, tags) VALUES (?, ?)",
                        (verre_id, tags_json)
                    )
                    conn.commit()  # Commit après chaque insertion
                    print(f"Ajout de la gravure comme tag pour le verre {verre_id}")
            except sqlite3.Error as e:
                print(f"Erreur lors de l'ajout de la gravure comme tag pour le verre {verre_id}: {e}")
                conn.rollback()

        # Ensuite, importer les tags du fichier JSON
        print("\nImport des tags depuis le fichier JSON...")
        json_path = os.path.join('tag', 'images_tags.json')

        # Lecture du fichier JSON
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Compteur pour suivre la progression
        total = len(data)
        current = 0

        # Pour chaque entrée dans le JSON
        for image_id, tags in data.items():
            current += 1
            print(f"Traitement de l'image {current}/{total}: {image_id}")
            
            # Extraire l'ID du verre
            verre_id = int(image_id.split('_')[0])

            # Vérifier si le verre existe
            cursor.execute("SELECT id FROM verres WHERE id = ?", (verre_id,))
            if cursor.fetchone():
                try:
                    # Convertir la liste de tags en chaîne JSON
                    tags_json = json.dumps(tags)
                    print(f"Tentative d'insertion pour le verre {verre_id} avec les tags: {tags_json}")
                    cursor.execute(
                        "INSERT INTO tags (verre_id, tags) VALUES (?, ?)",
                        (verre_id, tags_json)
                    )
                    conn.commit()  # Commit après chaque insertion
                except sqlite3.Error as e:
                    print(f"Erreur lors de l'insertion des tags pour le verre {verre_id}: {e}")
                    conn.rollback()
            else:
                print(f"Attention: Verre avec l'ID {verre_id} non trouvé dans la base de données")

        print("\nImport des tags terminé avec succès!")
        
        # Vérifier le nombre total de tags insérés
        cursor.execute("SELECT COUNT(*) FROM tags")
        total_tags = cursor.fetchone()[0]
        print(f"Nombre total de tags dans la base de données : {total_tags}")

    except FileNotFoundError:
        print(f"Le fichier {json_path} n'a pas été trouvé")
    except json.JSONDecodeError as e:
        print(f"Erreur de format JSON: {str(e)}")
        print("Vérifiez que le fichier JSON est correctement formaté")
    except Exception as e:
        print(f"Erreur lors de l'import des tags: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    import_tags() 