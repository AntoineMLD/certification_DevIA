"""
Collecteur de métriques en temps réel pour le monitoring de performance du modèle d'IA.

Ce module définit la classe ModelMonitor, un singleton responsable de :
- L'enregistrement des prédictions temporaires du modèle.
- La validation de ces prédictions (par exemple, par une intervention humaine).
- Le stockage persistant des prédictions validées.
- Le calcul et la génération de rapports de métriques de performance
  basées sur les prédictions validées.

Les données de prédiction gérées ont typiquement la structure suivante :
{
    'timestamp': str,          # Timestamp ISO de la prédiction
    'predicted_label': str,    # Label de classe prédit par le modèle (ou validé par l'utilisateur)
    'confidence': float,       # Score de confiance de la prédiction originale du modèle
    'embedding': list[float],  # (Optionnel) Embedding vectoriel de l'input
    'processing_time': float,  # Temps de traitement pour la prédiction en secondes
    'original_prediction': str # (Optionnel) Label prédit par le modèle avant validation humaine
}
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from collections import deque
import logging
from functools import lru_cache

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Singleton pour le monitoring des performances du modèle d'IA.

    Gère une fenêtre glissante de prédictions validées pour calculer des métriques
    pertinentes telles que la confiance moyenne, le temps de traitement moyen,
    la distribution des classes et la précision (si la validation humaine est fournie).
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("Nouvelle instance de ModelMonitor créée")
        return cls._instance
    
    def __init__(self, window_size=1000):
        """
        Initialise le moniteur de modèle.
        
        Args:
            window_size (int): Taille de la fenêtre glissante pour stocker
                               les prédictions validées et calculer les métriques.
        """
        # Éviter la réinitialisation si déjà initialisé
        if hasattr(self, 'initialized'):
            return
            
        self.initialized = True
        self.window_size = window_size
        # Stocke les prédictions après validation par l'utilisateur
        self.validated_predictions = deque(maxlen=window_size)
        # Stocke les prédictions brutes du modèle avant validation
        self.temp_predictions = deque(maxlen=10) 
        # DataFrame Pandas contenant les données des prédictions validées pour calculs
        self.current_data = None
        
        # Création des dossiers pour sauvegarder les rapports et l'historique
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        self.history_file = os.path.join(self.reports_dir, "predictions_history.json")
        os.makedirs(self.reports_dir, exist_ok=True)
        logger.info(f"Dossier des rapports créé: {self.reports_dir}")
        
        # Charger l'historique des prédictions
        self._load_history()
        
    def _validate_prediction_data(self, pred):
        """Valide la structure et les types des données de prédiction."""
        required_fields = ['timestamp', 'predicted_label', 'confidence', 'processing_time']
        
        # Vérifier les champs requis
        for field in required_fields:
            if field not in pred:
                logger.warning(f"Champ manquant dans la prédiction : {field}")
                return None
        
        # S'assurer que le timestamp est une chaîne ISO
        if isinstance(pred['timestamp'], datetime):
            pred['timestamp'] = pred['timestamp'].isoformat()
        
        # Convertir les valeurs numériques si nécessaire
        try:
            pred['confidence'] = float(pred['confidence'])
            pred['processing_time'] = float(pred['processing_time'])
        except (ValueError, TypeError):
            logger.warning("Erreur de conversion des valeurs numériques")
            return None
            
        return pred
        
    def _load_history(self):
        """Charge l'historique des prédictions validées depuis le fichier JSON."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    self.validated_predictions.clear()
                    history.sort(key=lambda x: x['timestamp'])
                    seen_timestamps = set()
                    for pred_data in history: # Renommé pred en pred_data pour clarté
                        if pred_data['timestamp'] not in seen_timestamps:
                            validated_pred_data = self._validate_prediction_data(pred_data)
                            if validated_pred_data:
                                self.validated_predictions.append(validated_pred_data)
                                seen_timestamps.add(pred_data['timestamp'])
                    logger.info(f"Historique chargé : {len(self.validated_predictions)} prédictions uniques depuis {self.history_file}")
                    self._update_current_data()
            else:
                logger.info(f"Aucun fichier d'historique trouvé à {self.history_file}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique : {str(e)}")
            
    def _save_history(self):
        """Sauvegarde l'historique des prédictions validées dans un fichier JSON."""
        try:
            history = []
            logger.info(f"[SAVE] Début de la sauvegarde - {len(self.validated_predictions)} prédictions à sauvegarder")
            
            for pred_data in self.validated_predictions: # Renommé pred en pred_data
                clean_pred_data = self._validate_prediction_data(pred_data)
                if clean_pred_data:
                    if isinstance(clean_pred_data.get('embedding'), np.ndarray):
                        clean_pred_data['embedding'] = clean_pred_data['embedding'].tolist()
                    history.append(clean_pred_data)
                    
            logger.info(f"[SAVE] {len(history)} prédictions nettoyées et prêtes à être sauvegardées")
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"[SAVE] Historique sauvegardé avec succès dans {self.history_file}")
            
            if os.path.exists(self.history_file):
                file_size = os.path.getsize(self.history_file)
                logger.info(f"[SAVE] Fichier créé avec succès, taille: {file_size} bytes")
            else:
                logger.error("[SAVE] Le fichier n'a pas été créé!")
                
            self.generate_report()
        except Exception as e:
            logger.error(f"[SAVE] Erreur lors de la sauvegarde de l'historique : {str(e)}")
            logger.exception("[SAVE] Détails de l'erreur:")

    def add_temp_prediction(self, prediction_data):
        """
        Ajoute une nouvelle prédiction brute du modèle à une file d'attente temporaire,
        en attendant une éventuelle validation par l'utilisateur.
        """
        logger.info(f"[TEMP] État actuel des prédictions temporaires: {len(self.temp_predictions)} prédictions")
        logger.info(f"[TEMP] Tentative d'ajout d'une prédiction temporaire: {prediction_data.get('predicted_label', 'Unknown')}")
        
        clean_data = self._validate_prediction_data(prediction_data)
        if clean_data:
            if isinstance(clean_data.get('embedding'), np.ndarray):
                clean_data['embedding'] = clean_data['embedding'].tolist()
            self.temp_predictions.append(clean_data)
            logger.info(f"[TEMP] Prédiction temporaire ajoutée avec succès: {clean_data['predicted_label']} (confiance: {clean_data['confidence']:.2%})")
            logger.info(f"[TEMP] Nombre total de prédictions temporaires après ajout: {len(self.temp_predictions)}")
        else:
            logger.error("[TEMP] Données de prédiction invalides")
        
    def validate_last_prediction(self, validated_class: str):
        """
        Valide la dernière prédiction temporaire avec un label de vérité terrain (validated_class)
        fourni par l'utilisateur, puis l'ajoute à la liste des prédictions validées.
        """
        logger.info(f"[VALIDATION] Début de la validation pour la classe: {validated_class}")
        
        if not self.temp_predictions:
            logger.warning("[VALIDATION] Pas de prédiction temporaire à valider - la file est vide")
            return
            
        last_pred = self.temp_predictions[-1] # On prend la dernière, on pourrait aussi la retirer avec pop()
        logger.info(f"[VALIDATION] Dernière prédiction temporaire trouvée: {last_pred['predicted_label']} -> {validated_class}")
        logger.info(f"[VALIDATION] Nombre total de prédictions temporaires: {len(self.temp_predictions)}")
        
        validated_pred_data = { # Renommé validated_pred
            'timestamp': last_pred['timestamp'],
            'predicted_label': validated_class, # C'est maintenant le label "vérité terrain"
            'confidence': last_pred['confidence'], # Confiance de la prédiction originale
            'embedding': last_pred['embedding'],
            'processing_time': last_pred['processing_time'],
            'original_prediction': last_pred['predicted_label'] # Label original du modèle
        }
        
        logger.info(f"[VALIDATION] Prédiction validée créée avec timestamp: {validated_pred_data['timestamp']}")
        
        clean_pred_data = self._validate_prediction_data(validated_pred_data)
        if clean_pred_data:
            self.validated_predictions.append(clean_pred_data)
            logger.info(f"[VALIDATION] Prédiction validée ajoutée avec succès (total: {len(self.validated_predictions)})")
            
            try:
                self._update_current_data()
                logger.info("[VALIDATION] Données courantes mises à jour")
                self._save_history()
                logger.info("[VALIDATION] Historique sauvegardé avec succès")
            except Exception as e:
                logger.error(f"[VALIDATION] Erreur lors de la sauvegarde: {str(e)}")
        else:
            logger.error("[VALIDATION] Impossible de valider la prédiction : données invalides")
        
    def _update_current_data(self):
        """Met à jour le DataFrame Pandas self.current_data avec les prédictions validées actuelles."""
        if not self.validated_predictions:
            self.current_data = None # Assurer que current_data est None si pas de prédictions
            return
            
        data_list = [] # Renommé data en data_list pour éviter confusion avec self.current_data
        for pred_data in self.validated_predictions: # Renommé pred en pred_data
            row = {
                'timestamp': pred_data['timestamp'],
                'predicted_label': pred_data['predicted_label'], # Label validé
                'confidence': pred_data['confidence'],
                'processing_time': pred_data['processing_time'],
                'original_prediction': pred_data.get('original_prediction', pred_data['predicted_label'])
            }
            data_list.append(row)
            
        self.current_data = pd.DataFrame(data_list)
        
    def generate_report(self):
        """
        Génère un rapport contenant un ensemble de métriques de performance du modèle,
        calculées sur la fenêtre actuelle de prédictions validées.

        Le rapport est également sauvegardé en JSON dans le dossier 'reports'.

        Les métriques incluses sont :
        - timestamp: Date et heure de génération du rapport.
        - avg_confidence: Confiance moyenne des prédictions originales du modèle
                          pour les cas qui ont été validés.
        - avg_processing_time: Temps de traitement moyen par prédiction.
        - n_predictions: Nombre total de prédictions validées dans la fenêtre actuelle.
        - predictions_per_class: Distribution des labels de classe validés.
        - prediction_accuracy: Précision du modèle, calculée en comparant la prédiction
                               originale du modèle au label validé par l'utilisateur.
                               (label validé est considéré comme la vérité terrain).
        
        Returns:
            dict or None: Un dictionnaire contenant les métriques, ou None si
                          aucune donnée de prédiction validée n'est disponible.
        """
        if self.current_data is None or len(self.current_data) < 1:
            logger.warning("Aucune donnée de prédiction validée disponible pour générer un rapport.")
            return None
            
        metrics = {
            # Timestamp de la génération de ce rapport
            'timestamp': datetime.now().isoformat(),
            # Confiance moyenne des prédictions originales du modèle
            'avg_confidence': self._calculate_avg_confidence(),
            # Temps de traitement moyen pour une prédiction par le modèle
            'avg_processing_time': self._calculate_avg_processing_time(),
            # Nombre total de prédictions validées utilisées pour ce rapport
            'n_predictions': len(self.current_data),
            # Distribution des classes (labels validés)
            'predictions_per_class': self._count_predictions_per_class(),
            # Précision du modèle basée sur la validation utilisateur
            'prediction_accuracy': self._calculate_prediction_accuracy()
        }
        
        metrics_path = os.path.join(
            self.reports_dir,
            f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Rapport de métriques sauvegardé dans {metrics_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du rapport de métriques : {e}")
        
        return metrics
        
    def _calculate_avg_confidence(self):
        """
        Calcule la confiance moyenne des prédictions originales du modèle
        sur la fenêtre courante de prédictions validées.
        La confiance est celle initialement retournée par le modèle, même si le label
        a été corrigé lors de la validation.
        """
        if self.current_data is None or 'confidence' not in self.current_data.columns or self.current_data['confidence'].empty:
            return 0.0
        return self.current_data['confidence'].mean()
        
    def _calculate_avg_processing_time(self):
        """
        Calcule le temps de traitement moyen des prédictions
        sur la fenêtre courante de prédictions validées.
        """
        if self.current_data is None or 'processing_time' not in self.current_data.columns or self.current_data['processing_time'].empty:
            return 0.0
        return self.current_data['processing_time'].mean()
        
    def _count_predictions_per_class(self):
        """
        Compte le nombre de prédictions validées pour chaque classe (label validé)
        sur la fenêtre courante. Utile pour voir la distribution des labels
        confirmés par l'utilisateur.
        """
        if self.current_data is None or 'predicted_label' not in self.current_data.columns or self.current_data['predicted_label'].empty:
            return {}
        return self.current_data['predicted_label'].value_counts().to_dict()

    def _calculate_prediction_accuracy(self):
        """
        Calcule la précision des prédictions du modèle sur la fenêtre courante.
        La précision est définie comme le pourcentage de prédictions où le label
        prédit originalement par le modèle ('original_prediction') correspond au
        label validé par l'utilisateur ('predicted_label').
        Nécessite que 'original_prediction' et 'predicted_label' soient présents.
        """
        if self.current_data is None or \
           'original_prediction' not in self.current_data.columns or \
           'predicted_label' not in self.current_data.columns:
            logger.warning("Données insuffisantes pour calculer la précision (original_prediction ou predicted_label manquant).")
            return 0.0
        
        correct_predictions = (self.current_data['original_prediction'] == self.current_data['predicted_label']).sum()
        total_predictions = len(self.current_data)
        
        if total_predictions == 0:
            return 0.0
            
        return correct_predictions / total_predictions

# Instance globale du moniteur
monitor = ModelMonitor() 