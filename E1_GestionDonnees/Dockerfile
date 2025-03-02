# Image de base Python
FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    WORKDIR=/app

# Création et définition du répertoire de travail
WORKDIR ${WORKDIR}

# Installation des dépendances système
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY api/app ./app
COPY api/run.py .

# Exposition du port
EXPOSE 8001

# Commande de démarrage
CMD ["python", "run.py"] 