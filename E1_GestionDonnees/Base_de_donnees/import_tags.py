import json
import sqlite3
import os

def import_tags():
    # Connexion à la base de données
    conn = sqlite3.connect('france_optique.db')
    cursor = conn.cursor()

    # Création de la table tags
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        verre_id INTEGER,
        tags TEXT,  -- Stockage des tags sous forme de liste JSON
        FOREIGN KEY (verre_id) REFERENCES verres(id)
    )
    ''')

    # Chemin vers le fichier JSON
    json_path = os.path.join('tag', 'images_tags.json')

    try:
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
                    cursor.execute(
                        "INSERT INTO tags (verre_id, tags) VALUES (?, ?)",
                        (verre_id, tags_json)
                    )
                except sqlite3.Error as e:
                    print(f"Erreur lors de l'insertion des tags pour le verre {verre_id}: {e}")
            else:
                print(f"Attention: Verre avec l'ID {verre_id} non trouvé dans la base de données")

        # Valider les changements
        conn.commit()
        print("Import des tags terminé avec succès!")

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