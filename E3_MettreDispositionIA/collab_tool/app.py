from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_descriptions():
    """Charge les descriptions existantes depuis le fichier JSON."""
    descriptions_file = os.path.join(UPLOAD_FOLDER, 'descriptions.json')
    if os.path.exists(descriptions_file):
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_image_list():
    """Récupère la liste des images et les trie par nombre de descriptions croissant."""
    images = []
    descriptions = get_descriptions()
    
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.endswith('.png'):
            try:
                image_id = int(filename.split('_')[0])
                # Compte le nombre de descriptions pour cette image
                description_count = 0
                if str(image_id) in descriptions:
                    if isinstance(descriptions[str(image_id)], list):
                        description_count = len(descriptions[str(image_id)])
                    else:
                        description_count = 1
                
                images.append({
                    'id': image_id,
                    'filename': filename,
                    'description_count': description_count
                })
            except (ValueError, IndexError):
                continue
    
    # Trie les images par nombre de descriptions croissant, puis par ID
    return sorted(images, key=lambda x: (x['description_count'], x['id']))

@app.route('/')
def index():
    images = get_image_list()
    current_index = request.args.get('index', 0, type=int)
    
    if not images:
        return render_template('index.html', error="Aucune image trouvée")
    
    current_index = max(0, min(current_index, len(images) - 1))
    current_image = images[current_index]
    
    # Charge la description existante si elle existe
    descriptions = get_descriptions()
    existing_description = descriptions.get(str(current_image['id']), [])
    if not isinstance(existing_description, list):
        existing_description = [existing_description]
    
    return render_template('index.html',
                         image_filename=current_image['filename'],
                         image_id=current_image['id'],
                         current_index=current_index,
                         total_images=len(images),
                         description_count=current_image['description_count'],
                         existing_descriptions=existing_description)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

@app.route('/save_description', methods=['POST'])
def save_description():
    data = request.json
    description = data.get('description')
    verre_id = data.get('verre_id')
    
    if not description or not verre_id:
        return jsonify({'error': 'Description et ID du verre requis'}), 400
    
    # Charger les descriptions existantes
    descriptions_file = os.path.join(UPLOAD_FOLDER, 'descriptions.json')
    descriptions = get_descriptions()
    
    # Convertir l'ID en string pour le stockage JSON
    verre_id = str(verre_id)
    
    # Si c'est la première description pour cette image, créer une liste
    if verre_id not in descriptions:
        descriptions[verre_id] = []
    elif not isinstance(descriptions[verre_id], list):
        # Convertir une ancienne description unique en liste
        descriptions[verre_id] = [descriptions[verre_id]]
    
    # Ajouter la nouvelle description à la liste
    descriptions[verre_id].append(description)
    
    # Sauvegarder les descriptions
    with open(descriptions_file, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True) 