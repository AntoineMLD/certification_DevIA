import secrets
import base64
import os
from pathlib import Path
import shutil

def generate_secret_key():
    """
    Génère une clé secrète sécurisée
    """
    return base64.b64encode(secrets.token_bytes(32)).decode()

def create_env_file():
    """
    Crée le fichier .env à partir du template .env.example
    """
    current_dir = Path(__file__).parent.absolute()
    env_example = current_dir / ".env.example"
    env_path = current_dir / ".env"
    
    # Vérifier si .env.example existe
    if not env_example.exists():
        raise FileNotFoundError("Le fichier .env.example est manquant")
    
    # Copier .env.example vers .env
    shutil.copy(env_example, env_path)
    
    # Générer et remplacer la clé secrète
    secret_key = generate_secret_key()
    
    # Lire le contenu actuel
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remplacer la clé secrète
    content = content.replace("your_secret_key_here", secret_key)
    
    # Écrire le nouveau contenu
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return secret_key

if __name__ == "__main__":
    try:
        # Créer le fichier .env
        secret_key = create_env_file()
        
        print("\nFichier .env créé avec succès!")
        print("\nVotre nouvelle clé secrète :")
        print("-----------------------------")
        print(secret_key)
        print("-----------------------------")
        print("\nIMPORTANT : ")
        print("1. Modifiez les valeurs dans le fichier .env")
        print("2. Ne partagez JAMAIS le fichier .env")
        print("3. Utilisez des mots de passe forts")
        
    except Exception as e:
        print(f"Erreur : {str(e)}") 