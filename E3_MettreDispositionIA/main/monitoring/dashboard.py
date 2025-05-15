"""
Dashboard de monitoring en temps réel avec Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import json
import glob
from metrics_collector import ModelMonitor
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page - DOIT ÊTRE LE PREMIER APPEL STREAMLIT
st.set_page_config(
    page_title="Monitoring IA - Dashboard",
    page_icon="📊",
    layout="wide"
)

# Titre
st.title("📊 Dashboard de Monitoring en Temps Réel")

# Sidebar pour les contrôles
st.sidebar.title("Contrôles")
update_interval = st.sidebar.slider(
    "Intervalle de mise à jour (secondes)",
    min_value=1,
    max_value=60,
    value=5
)

# Initialisation du monitor comme variable de session
if 'monitor' not in st.session_state:
    st.session_state.monitor = ModelMonitor()
    st.session_state.last_update = time.time()

def load_metrics():
    """Charge les métriques depuis le monitor."""
    try:
        # Ne recharger que si l'intervalle de temps est écoulé
        current_time = time.time()
        if current_time - st.session_state.last_update >= update_interval:
            logger.info("[DASHBOARD] Rechargement des métriques")
            st.session_state.monitor._load_history()
            st.session_state.last_update = current_time
            
        metrics = st.session_state.monitor.generate_report()
        if metrics:
            logger.info(f"[DASHBOARD] Métriques chargées: {len(st.session_state.monitor.validated_predictions)} prédictions")
        else:
            logger.warning("[DASHBOARD] Aucune métrique disponible")
        return metrics
    except Exception as e:
        logger.error(f"[DASHBOARD] Erreur lors du chargement des métriques: {str(e)}")
        return None

# Fonction pour créer un graphique de métrique
def create_metric_chart(data, metric_name, color):
    if not data:
        return None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[data['timestamp']],
        y=[data[metric_name]],
        mode='markers',
        name=metric_name,
        line=dict(color=color)
    ))
    fig.update_layout(
        title=metric_name.replace('_', ' ').title(),
        xaxis_title="Temps",
        yaxis_title="Valeur",
        height=300
    )
    return fig

# Fonction pour afficher les métriques actuelles
def display_current_metrics(metrics):
    if not metrics:
        st.warning("Pas de métriques disponibles. Validez quelques prédictions pour voir les statistiques.")
        return
        
    cols = st.columns(4)
    
    # Score de confiance moyen
    with cols[0]:
        st.metric(
            "Score de Confiance Moyen",
            f"{metrics['avg_confidence']:.2%}",
            delta=None,
            help="Score moyen de similarité entre l'image soumise et la classe prédite"
        )
    
    # Temps de traitement moyen
    with cols[1]:
        st.metric(
            "Temps de Traitement Moyen",
            f"{metrics['avg_processing_time']:.3f}s",
            delta=None,
            help="Temps moyen nécessaire pour analyser une image"
        )
    
    # Nombre de prédictions validées
    with cols[2]:
        st.metric(
            "Prédictions Validées",
            metrics['n_predictions'],
            delta=None,
            help="Nombre total de prédictions validées par les utilisateurs"
        )
    
    # Précision des prédictions
    with cols[3]:
        accuracy_value = metrics.get('prediction_accuracy', 0.0)
        if accuracy_value is None:
            accuracy_value = 0.0
        st.metric(
            "Correspondance Prédictions/Validations",
            f"{accuracy_value:.2%}",
            delta=None,
            help="Pourcentage de prédictions correspondant aux validations des utilisateurs"
        )

# Fonction principale pour mettre à jour le dashboard
def update_dashboard():
    # Récupérer les dernières métriques
    metrics = load_metrics()
    
    # Afficher les métriques actuelles
    display_current_metrics(metrics)
    
    if metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique de confiance
            confidence_chart = create_metric_chart(metrics, 'avg_confidence', 'orange')
            if confidence_chart:
                st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Graphique de temps de traitement
            processing_time_chart = create_metric_chart(metrics, 'avg_processing_time', 'green')
            if processing_time_chart:
                st.plotly_chart(processing_time_chart, use_container_width=True)
            
        with col2:
            # Distribution des classes prédites
            if 'predictions_per_class' in metrics and metrics['predictions_per_class']:
                st.subheader("Distribution des Types de Verres Validés")
                fig = px.pie(
                    values=list(metrics['predictions_per_class'].values()),
                    names=list(metrics['predictions_per_class'].keys()),
                    title='Répartition des Prédictions Validées par Type de Verre'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des prédictions
                st.subheader("Détails des Prédictions par Type")
                pred_df = pd.DataFrame(
                    list(metrics['predictions_per_class'].items()),
                    columns=['Type de Verre', 'Nombre de Validations']
                ).sort_values('Nombre de Validations', ascending=False)
                st.dataframe(pred_df, use_container_width=True)
            else:
                st.info("Aucune prédiction validée disponible pour le moment.")

# Utiliser le système de rafraîchissement natif de Streamlit
if __name__ == "__main__":
    update_dashboard()
    # Utiliser le système de rafraîchissement natif de Streamlit
    time.sleep(update_interval)
    st.rerun() 