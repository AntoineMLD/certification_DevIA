"""
Collecteur de métriques en temps réel
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
            window_size: Taille de la fenêtre glissante pour les métriques
        """
        # Éviter la réinitialisation si déjà initialisé
        if hasattr(self, 'initialized'):
            return
            
        self.initialized = True
        self.window_size = window_size
        self.validated_predictions = deque(maxlen=window_size)
        self.temp_predictions = deque(maxlen=10)
        self.current_data = None
        
        # Création des dossiers pour sauvegarder les rapports et l'historique
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        self.history_file = os.path.join(self.reports_dir, "predictions_history.json")
        os.makedirs(self.reports_dir, exist_ok=True)
        logger.info(f"Dossier des rapports créé: {self.reports_dir}")
        
        # Charger l'historique des prédictions
        self._load_history()
        
    def _validate_prediction_data(self, pred):
        """Valide et nettoie les données de prédiction."""
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
                    # Vider les prédictions existantes avant de charger
                    self.validated_predictions.clear()
                    # Trier l'historique par timestamp pour garder l'ordre chronologique
                    history.sort(key=lambda x: x['timestamp'])
                    # Utiliser un set pour suivre les timestamps déjà vus
                    seen_timestamps = set()
                    for pred in history:
                        if pred['timestamp'] not in seen_timestamps:
                            validated_pred = self._validate_prediction_data(pred)
                            if validated_pred:
                                self.validated_predictions.append(validated_pred)
                                seen_timestamps.add(pred['timestamp'])
                    logger.info(f"Historique chargé : {len(self.validated_predictions)} prédictions uniques depuis {self.history_file}")
                    self._update_current_data()
            else:
                logger.info(f"Aucun fichier d'historique trouvé à {self.history_file}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique : {str(e)}")
            
    def _save_history(self):
        """Sauvegarde l'historique des prédictions validées dans un fichier JSON."""
        try:
            # Convertir les données en format JSON-compatible
            history = []
            logger.info(f"[SAVE] Début de la sauvegarde - {len(self.validated_predictions)} prédictions à sauvegarder")
            
            for pred in self.validated_predictions:
                clean_pred = self._validate_prediction_data(pred)
                if clean_pred:
                    if isinstance(clean_pred.get('embedding'), np.ndarray):
                        clean_pred['embedding'] = clean_pred['embedding'].tolist()
                    history.append(clean_pred)
                    
            logger.info(f"[SAVE] {len(history)} prédictions nettoyées et prêtes à être sauvegardées")
            
            # Sauvegarder l'historique complet
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"[SAVE] Historique sauvegardé avec succès dans {self.history_file}")
            
            # Vérifier que le fichier a bien été créé
            if os.path.exists(self.history_file):
                file_size = os.path.getsize(self.history_file)
                logger.info(f"[SAVE] Fichier créé avec succès, taille: {file_size} bytes")
            else:
                logger.error("[SAVE] Le fichier n'a pas été créé!")
                
            # Générer le rapport des métriques actuelles
            self.generate_report()
        except Exception as e:
            logger.error(f"[SAVE] Erreur lors de la sauvegarde de l'historique : {str(e)}")
            logger.exception("[SAVE] Détails de l'erreur:")

    def add_temp_prediction(self, prediction_data):
        """
        Ajoute une nouvelle prédiction temporaire.
        """
        logger.info(f"[TEMP] État actuel des prédictions temporaires: {len(self.temp_predictions)} prédictions")
        logger.info(f"[TEMP] Tentative d'ajout d'une prédiction temporaire: {prediction_data.get('predicted_label', 'Unknown')}")
        
        # Valider et nettoyer les données
        clean_data = self._validate_prediction_data(prediction_data)
        if clean_data:
            if isinstance(clean_data.get('embedding'), np.ndarray):
                clean_data['embedding'] = clean_data['embedding'].tolist()
            self.temp_predictions.append(clean_data)
            logger.info(f"[TEMP] Prédiction temporaire ajoutée avec succès: {clean_data['predicted_label']} (confiance: {clean_data['confidence']:.2%})")
            logger.info(f"[TEMP] Nombre total de prédictions temporaires après ajout: {len(self.temp_predictions)}")
        else:
            logger.error("[TEMP] Données de prédiction invalides")
        
    def validate_last_prediction(self, validated_class):
        """
        Valide la dernière prédiction et l'ajoute aux prédictions validées.
        """
        logger.info(f"[VALIDATION] Début de la validation pour la classe: {validated_class}")
        
        if not self.temp_predictions:
            logger.warning("[VALIDATION] Pas de prédiction temporaire à valider - la file est vide")
            return
            
        # Récupérer la dernière prédiction temporaire
        last_pred = self.temp_predictions[-1]
        logger.info(f"[VALIDATION] Dernière prédiction temporaire trouvée: {last_pred['predicted_label']} -> {validated_class}")
        logger.info(f"[VALIDATION] Nombre total de prédictions temporaires: {len(self.temp_predictions)}")
        
        # Créer une prédiction validée
        validated_pred = {
            'timestamp': last_pred['timestamp'],
            'predicted_label': validated_class,
            'confidence': last_pred['confidence'],
            'embedding': last_pred['embedding'],
            'processing_time': last_pred['processing_time'],
            'original_prediction': last_pred['predicted_label']
        }
        
        logger.info(f"[VALIDATION] Prédiction validée créée avec timestamp: {validated_pred['timestamp']}")
        
        # Valider et nettoyer les données
        clean_pred = self._validate_prediction_data(validated_pred)
        if clean_pred:
            # Ajouter aux prédictions validées
            self.validated_predictions.append(clean_pred)
            logger.info(f"[VALIDATION] Prédiction validée ajoutée avec succès (total: {len(self.validated_predictions)})")
            
            # Mettre à jour les données courantes et sauvegarder l'historique
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
        """Met à jour le DataFrame des données courantes avec les prédictions validées."""
        if not self.validated_predictions:
            return
            
        data = []
        for pred in self.validated_predictions:
            row = {
                'timestamp': pred['timestamp'],
                'predicted_label': pred['predicted_label'],
                'confidence': pred['confidence'],
                'processing_time': pred['processing_time'],
                'original_prediction': pred.get('original_prediction', pred['predicted_label'])
            }
            data.append(row)
            
        self.current_data = pd.DataFrame(data)
        
    def generate_report(self):
        """
        Génère un rapport des métriques de performance.
        
        Returns:
            dict: Rapport contenant les métriques
        """
        if self.current_data is None or len(self.current_data) < 1:
            return None
            
        # Calcul des métriques de base
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'avg_confidence': self._calculate_avg_confidence(),
            'avg_processing_time': self._calculate_avg_processing_time(),
            'n_predictions': len(self.current_data),
            'predictions_per_class': self._count_predictions_per_class(),
            'prediction_accuracy': self._calculate_prediction_accuracy()
        }
        
        # Sauvegarde des métriques
        metrics_path = os.path.join(
            self.reports_dir,
            f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
        
    def _calculate_avg_confidence(self):
        """Calcule la confiance moyenne sur la fenêtre courante."""
        if self.current_data is None:
            return 0.0
        return self.current_data['confidence'].mean()
        
    def _calculate_avg_processing_time(self):
        """Calcule le temps de traitement moyen sur la fenêtre courante."""
        if self.current_data is None:
            return 0.0
        return self.current_data['processing_time'].mean()
        
    def _count_predictions_per_class(self):
        """Compte le nombre de prédictions validées par classe."""
        if self.current_data is None:
            return {}
        return self.current_data['predicted_label'].value_counts().to_dict()
        
    def _calculate_prediction_accuracy(self):
        """Calcule le pourcentage de prédictions correctes (où la prédiction originale correspond à la classe validée)."""
        if self.current_data is None:
            return 0.0
        matches = (self.current_data['predicted_label'] == self.current_data['original_prediction'])
        return matches.mean()

# Instance globale du moniteur
monitor = ModelMonitor() 