# ============================================================
#  Stock Intelligence — Cegid XRP Flex
#  Timsoft — Usage interne uniquement
#  Image Docker : Python 3.11 slim + Streamlit
# ============================================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Timsoft"
LABEL description="Dashboard prévision demande & optimisation stocks"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Répertoire de travail dans le conteneur
WORKDIR /app

# ── 1. Installer les dépendances système légères ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# libgomp1 est requis par LightGBM pour le parallélisme

# ── 2. Copier et installer les dépendances Python ───────────
# On copie requirements.txt en premier pour profiter du cache Docker :
# si requirements.txt ne change pas, cette couche n'est pas reconstruite
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── 3. Copier les fichiers du dashboard ─────────────────────
COPY dashboard.py .

# ── 4. Copier les données ML (fichiers xlsx + modèle pkl) ───
COPY base_ml_finale.xlsx .
COPY predictions_demande.xlsx .
COPY predictions_rupture.xlsx .
COPY predictions_anomalie_stock.xlsx .
COPY performances_par_article.xlsx .
COPY performances_rupture_par_article.xlsx .
COPY performances_anomalie_par_article.xlsx .
COPY resultats_regression.xlsx .
COPY modeles_ml.pkl .

# ── 5. Copier la configuration Streamlit ────────────────────
COPY .streamlit/ .streamlit/

# ── 6. Exposer le port Streamlit ────────────────────────────
EXPOSE 8501

# ── 7. Vérification de santé (Azure l'utilise pour savoir ───
#       si le conteneur est prêt)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" \
    || exit 1

# ── 8. Commande de démarrage ─────────────────────────────────
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
