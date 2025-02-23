import subprocess
import sqlite3
import os
from datetime import datetime

# Configuration
SPIDERS = [
    'glass_spider_hoya',
    'glass_spider_full_xpath',
    'glass_spider',  
    'glass_spider_particular',
    'glass_spider_optovision',
    'glass_spider_indo_optical',
]

DB_PATH = os.path.join('..', 'Base_de_donnees', 'france_optique.db')

def count_database_rows():
    """
    Compte le nombre de lignes dans la table staging.
    
    Returns:
        int: Nombre de lignes
    """
    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM staging')
            return cursor.fetchone()[0]
    except sqlite3.Error as error:
        print(f"❌ Erreur base de données: {error}")
        return 0

def run_spider(spider_name):
    """
    Exécute un spider et affiche ses résultats.
    
    Args:
        spider_name (str): Nom du spider à exécuter
    """
    print(f"\n{'='*50}")
    print(f"Spider: {spider_name}")
    print(f"Démarrage: {datetime.now().strftime('%H:%M:%S')}")
    print('='*50)
    
    try:
        # Compte initial
        initial_count = count_database_rows()
        
        # Exécute le spider
        result = subprocess.run(
            ['scrapy', 'crawl', spider_name],
            check=True,
            text=True,
            capture_output=True
        )
        
        # Affiche les logs
        print(result.stdout)
        
        # Statistiques
        final_count = count_database_rows()
        new_items = final_count - initial_count
        
        print(f"\nRésultats:")
        print(f"✅ Nouveaux items: {new_items}")
        print(f"📊 Total en base: {final_count}")
        
    except subprocess.CalledProcessError as error:
        print(f"\n❌ Erreur spider: {error.stderr}")

def main():
    """Point d'entrée du script."""
    start_time = datetime.now()
    print(f"\n🚀 Démarrage du scraping: {start_time.strftime('%H:%M:%S')}")
    
    for spider in SPIDERS:
        run_spider(spider)
        print('-'*50)
    
    duration = datetime.now() - start_time
    print(f"\n✨ Scraping terminé en {duration.seconds} secondes")

if __name__ == "__main__":
    main() 