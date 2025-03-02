import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Imports
import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Lancer l'API avec uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # Forcer l'écoute sur toutes les interfaces
        port=8001,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True  # Activer les logs d'accès
    ) 