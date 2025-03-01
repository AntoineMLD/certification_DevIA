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
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    ) 