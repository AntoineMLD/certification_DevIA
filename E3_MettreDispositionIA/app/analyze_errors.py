#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour analyser les images mal classées par le modèle et
générer un rapport des classes les plus souvent confondues.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path
import shutil
import cv2
from sklearn.metrics import confusion_matrix

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les fonctions nécessaires
try:
    from app.efficientnet_model import load_model as load_efficientnet_model
except ImportError:
    print("Erreur: Impossible d'importer le module efficientnet_model")
    sys.exit(1)

def load_misclassified_images(incorrect_dir):
    """
    Charge les informations sur les images mal classées.
    
    Supporte deux formats:
    1. Fichiers au format: true_class_vs_predicted_class_confidence.jpg
    2. Organisation en dossiers par classe prédite: predicted_class/true_class_pred_predicted_class_id.jpg
    
    Args:
        incorrect_dir: Chemin vers le dossier contenant les images mal classées
        
    Returns:
        Une liste de tuples (vrai_label, label_prédit, confiance, chemin_image)
    """
    misclassified = []
    
    if not os.path.exists(incorrect_dir):
        print(f"Erreur: Le répertoire {incorrect_dir} n'existe pas!")
        return misclassified
    
    # Vérifier si le dossier contient des sous-dossiers (format 2)
    subdirs = [f for f in os.listdir(incorrect_dir) if os.path.isdir(os.path.join(incorrect_dir, f))]
    
    if subdirs:
        print(f"Détection de l'organisation en dossiers ({len(subdirs)} classes de prédiction)")
        # Format 2: Organisation en dossiers par classe prédite
        for predicted_class in subdirs:
            pred_dir = os.path.join(incorrect_dir, predicted_class)
            for filename in os.listdir(pred_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                try:
                    # Format attendu: true_class_name_pred_predicted_class_name_id.png
                    parts = filename.split('_')
                    if len(parts) < 4 or 'pred' not in parts:
                        print(f"Format incorrect pour {filename}")
                        continue
                    
                    # Trouver l'index de 'pred'
                    pred_index = parts.index('pred')
                    
                    # Extraire la vraie classe (tout avant 'pred')
                    true_class = '_'.join(parts[:pred_index])
                    if true_class.startswith('true_'):
                        true_class = true_class[5:]  # Enlever le préfixe 'true_'
                    
                    # Le dossier contient déjà la classe prédite
                    predicted_class_name = predicted_class
                    
                    # Utiliser une confiance par défaut (car non disponible dans le nom de fichier)
                    confidence = 0.5
                    
                    image_path = os.path.join(pred_dir, filename)
                    misclassified.append((true_class, predicted_class_name, confidence, image_path))
                
                except Exception as e:
                    print(f"Erreur lors du traitement de {filename}: {str(e)}")
                    continue
    else:
        # Format 1: Fichiers au format standard
        for filename in os.listdir(incorrect_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Extraire les informations du nom de fichier
            # Format: true_class_vs_predicted_class_confidence.jpg
            try:
                # Enlever l'extension
                base_name = os.path.splitext(filename)[0]
                
                # Séparer en parties
                parts = base_name.split('_vs_')
                if len(parts) != 2:
                    print(f"Format incorrect pour {filename}. Format attendu: true_class_vs_predicted_class_confidence.jpg")
                    continue
                    
                true_class = parts[0]
                
                # Extraire la classe prédite et la confiance
                pred_parts = parts[1].rsplit('_', 1)
                if len(pred_parts) != 2:
                    print(f"Format incorrect pour {filename}. Format attendu: true_class_vs_predicted_class_confidence.jpg")
                    continue
                    
                predicted_class = pred_parts[0]
                confidence = float(pred_parts[1])
                
                image_path = os.path.join(incorrect_dir, filename)
                
                misclassified.append((true_class, predicted_class, confidence, image_path))
                
            except Exception as e:
                print(f"Erreur lors du traitement de {filename}: {str(e)}")
                continue
    
    print(f"Total d'images mal classées chargées: {len(misclassified)}")
    return misclassified

def build_confusion_pairs(misclassified):
    """
    Construit un dictionnaire des paires de classes confondues.
    
    Args:
        misclassified: Liste de tuples (vrai_label, label_prédit, confiance, chemin_image)
        
    Returns:
        Un dictionnaire où les clés sont des tuples (vrai_label, label_prédit) et 
        les valeurs sont des listes de tuples (confiance, chemin_image)
    """
    confusion_pairs = defaultdict(list)
    
    for true_class, predicted_class, confidence, image_path in misclassified:
        confusion_pairs[(true_class, predicted_class)].append((confidence, image_path))
    
    return confusion_pairs

def analyze_confusion_matrix(misclassified):
    """
    Crée et analyse une matrice de confusion.
    
    Args:
        misclassified: Liste de tuples (vrai_label, label_prédit, confiance, chemin_image)
        
    Returns:
        Un DataFrame pandas avec les statistiques de confusion
    """
    # Extraire les classes uniques
    all_classes = set()
    for true_class, predicted_class, _, _ in misclassified:
        all_classes.add(true_class)
        all_classes.add(predicted_class)
    
    all_classes = sorted(list(all_classes))
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    
    # Préparer les données pour la matrice de confusion
    y_true = [class_to_idx[true_class] for true_class, _, _, _ in misclassified]
    y_pred = [class_to_idx[pred_class] for _, pred_class, _, _ in misclassified]
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=range(len(all_classes)))
    
    # Créer un DataFrame pour les statistiques
    confusion_stats = []
    
    for i, true_class in enumerate(all_classes):
        for j, pred_class in enumerate(all_classes):
            if i != j and cm[i, j] > 0:
                confusion_stats.append({
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'count': cm[i, j],
                    'pair': f"{true_class} → {pred_class}"
                })
    
    # Convertir en DataFrame et trier
    df_confusion = pd.DataFrame(confusion_stats)
    if not df_confusion.empty:
        df_confusion = df_confusion.sort_values(by='count', ascending=False)
    
    return df_confusion, all_classes, cm

def create_confusion_matrix_plot(cm, classes, output_dir):
    """
    Crée une visualisation de la matrice de confusion.
    
    Args:
        cm: Matrice de confusion calculée
        classes: Liste des noms de classes
        output_dir: Répertoire où sauvegarder le graphique
    """
    plt.figure(figsize=(12, 10))
    
    # Pour une meilleure lisibilité, afficher seulement les erreurs
    # (mettre les diagonales à 0)
    np.fill_diagonal(cm, 0)
    
    # Normaliser pour obtenir des pourcentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Remplacer NaN par 0
    
    # Créer un masque pour ignorer les cases à 0
    mask = cm == 0
    
    # Plot avec heatmap
    ax = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Reds',
                     xticklabels=classes, yticklabels=classes,
                     mask=mask, cbar=True)
    
    # Améliorer le formatage
    plt.title('Matrice de confusion des erreurs')
    plt.xlabel('Classe prédite')
    plt.ylabel('Vraie classe')
    
    # Rotation des labels pour lisibilité
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matrice de confusion sauvegardée dans {confusion_matrix_path}")

def identify_common_confusion_patterns(df_confusion):
    """
    Identifie les motifs de confusion récurrents.
    
    Args:
        df_confusion: DataFrame avec les statistiques de confusion
        
    Returns:
        Un DataFrame avec les groupes de classes qui se confondent mutuellement
    """
    if df_confusion.empty:
        return pd.DataFrame()
    
    # Identifier les paires bidirectionnelles (A confond B et B confond A)
    bidirectional_pairs = []
    processed_pairs = set()
    
    for _, row in df_confusion.iterrows():
        true_class = row['true_class']
        pred_class = row['predicted_class']
        count_forward = row['count']
        
        # Vérifier si la paire inverse existe
        inverse_pair = f"{pred_class} → {true_class}"
        inverse_entries = df_confusion[df_confusion['pair'] == inverse_pair]
        
        if not inverse_entries.empty and (true_class, pred_class) not in processed_pairs:
            count_backward = inverse_entries.iloc[0]['count']
            total_confusion = count_forward + count_backward
            
            bidirectional_pairs.append({
                'class1': true_class,
                'class2': pred_class,
                'confusion_total': total_confusion,
                'confusion_class1_as_class2': count_forward,
                'confusion_class2_as_class1': count_backward,
                'confusion_ratio': max(count_forward, count_backward) / min(count_forward, count_backward) if min(count_forward, count_backward) > 0 else float('inf')
            })
            
            processed_pairs.add((true_class, pred_class))
            processed_pairs.add((pred_class, true_class))
    
    # Convertir en DataFrame et trier
    df_patterns = pd.DataFrame(bidirectional_pairs)
    if not df_patterns.empty:
        df_patterns = df_patterns.sort_values(by='confusion_total', ascending=False)
    
    return df_patterns

def create_error_examples_grid(misclassified, output_dir, top_n=20):
    """
    Crée une grille des exemples d'erreurs les plus fréquentes.
    
    Args:
        misclassified: Liste de tuples (vrai_label, label_prédit, confiance, chemin_image)
        output_dir: Répertoire où sauvegarder les images
        top_n: Nombre maximum d'exemples à montrer
    """
    confusion_pairs = build_confusion_pairs(misclassified)
    
    # Trier les paires par nombre d'occurrences
    sorted_pairs = sorted(confusion_pairs.items(), 
                         key=lambda x: len(x[1]), 
                         reverse=True)
    
    # Limiter aux top_n paires les plus fréquentes
    top_pairs = sorted_pairs[:top_n]
    
    for (true_class, pred_class), examples in top_pairs:
        # Limiter à 4 exemples par paire
        examples = sorted(examples, key=lambda x: x[0], reverse=True)[:4]
        
        # Créer une figure avec les exemples
        num_examples = len(examples)
        if num_examples == 0:
            continue
            
        fig, axes = plt.subplots(1, num_examples, figsize=(4*num_examples, 4))
        if num_examples == 1:
            axes = [axes]
        
        for i, (confidence, image_path) in enumerate(examples):
            img = Image.open(image_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(f"Conf: {confidence:.2f}")
            axes[i].axis('off')
        
        plt.suptitle(f"Confusion: {true_class} → {pred_class} ({len(confusion_pairs[(true_class, pred_class)])} cas)")
        plt.tight_layout()
        
        # Sauvegarder
        output_path = os.path.join(output_dir, f"confusion_{true_class}_vs_{pred_class}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

def generate_html_report(df_confusion, df_patterns, output_dir):
    """
    Génère un rapport HTML avec les résultats de l'analyse.
    
    Args:
        df_confusion: DataFrame avec les statistiques de confusion
        df_patterns: DataFrame avec les motifs récurrents
        output_dir: Répertoire où sauvegarder le rapport
    """
    # Début du HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport d'analyse des erreurs de classification</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .container { display: flex; flex-wrap: wrap; }
            .image-container { margin: 10px; }
            .recommendations { background-color: #e6f7ff; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Rapport d'analyse des erreurs de classification</h1>
    """
    
    # Ajouter la matrice de confusion
    html += """
        <h2>1. Matrice de Confusion</h2>
        <p>La matrice de confusion ci-dessous montre les erreurs de classification entre les différentes classes.</p>
        <img src="confusion_matrix.png" alt="Matrice de confusion" style="max-width:100%; height:auto;">
    """
    
    # Ajouter la table des confusions les plus fréquentes
    html += """
        <h2>2. Confusions les plus fréquentes</h2>
        <p>Le tableau suivant présente les confusions les plus fréquentes entre les classes.</p>
        <table>
            <tr>
                <th>Vraie classe</th>
                <th>Classe prédite</th>
                <th>Nombre de cas</th>
            </tr>
    """
    
    # Ajouter chaque ligne du DataFrame
    if not df_confusion.empty:
        for _, row in df_confusion.head(20).iterrows():
            html += f"""
            <tr>
                <td>{row['true_class']}</td>
                <td>{row['predicted_class']}</td>
                <td>{row['count']}</td>
            </tr>
            """
    
    html += """
        </table>
    """
    
    # Ajouter la section sur les motifs récurrents
    html += """
        <h2>3. Paires de classes fréquemment confondues</h2>
        <p>Ces classes se confondent mutuellement et pourraient bénéficier d'une fusion ou d'une redéfinition.</p>
        <table>
            <tr>
                <th>Classe 1</th>
                <th>Classe 2</th>
                <th>Total confusions</th>
                <th>Classe1 → Classe2</th>
                <th>Classe2 → Classe1</th>
                <th>Ratio</th>
            </tr>
    """
    
    # Ajouter chaque ligne du DataFrame
    if not df_patterns.empty:
        for _, row in df_patterns.head(10).iterrows():
            html += f"""
            <tr>
                <td>{row['class1']}</td>
                <td>{row['class2']}</td>
                <td>{row['confusion_total']}</td>
                <td>{row['confusion_class1_as_class2']}</td>
                <td>{row['confusion_class2_as_class1']}</td>
                <td>{row['confusion_ratio']:.2f}</td>
            </tr>
            """
    
    html += """
        </table>
    """
    
    # Ajouter la section des exemples visuels
    html += """
        <h2>4. Exemples visuels de confusions</h2>
        <p>Voici quelques exemples visuels des confusions les plus fréquentes.</p>
        <div class="container">
    """
    
    # Ajouter les images d'exemples
    image_files = [f for f in os.listdir(output_dir) if f.startswith("confusion_") and f.endswith(".png")]
    for image_file in sorted(image_files)[:15]:  # Limiter à 15 exemples
        html += f"""
            <div class="image-container">
                <img src="{image_file}" alt="Exemple de confusion" style="width:100%; max-width:600px;">
            </div>
        """
    
    html += """
        </div>
    """
    
    # Ajouter les recommandations automatiques
    html += """
        <h2>5. Recommandations</h2>
        <div class="recommendations">
            <h3>Classes potentiellement à fusionner ou redéfinir:</h3>
            <ul>
    """
    
    # Générer des recommandations basées sur les motifs de confusion
    if not df_patterns.empty:
        for i, row in df_patterns.head(5).iterrows():
            html += f"""
                <li><strong>{row['class1']} et {row['class2']}</strong>: 
                Ces classes sont mutuellement confondues {row['confusion_total']} fois. 
                Considérer {'une fusion' if row['confusion_ratio'] < 2 else 'une redéfinition des critères de classification'}.</li>
            """
    
    html += """
            </ul>
            <h3>Recommandations générales:</h3>
            <ol>
                <li>Envisager d'ajouter plus d'images d'entraînement pour les classes souvent mal classées.</li>
                <li>Pour les classes fréquemment confondues, considérer soit:
                    <ul>
                        <li>Les fusionner si elles sont conceptuellement similaires</li>
                        <li>Améliorer leurs distinctions visuelles et d'étiquetage</li>
                        <li>Créer des sous-classes plus précises</li>
                    </ul>
                </li>
                <li>Examiner les images à faible confiance pour détecter des problèmes de qualité ou d'étiquetage.</li>
            </ol>
        </div>
    """
    
    # Fin du HTML
    html += """
    </body>
    </html>
    """
    
    # Sauvegarder le rapport
    report_path = os.path.join(output_dir, "error_analysis_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Rapport HTML sauvegardé dans {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyse des erreurs de classification")
    parser.add_argument("--incorrect_dir", type=str, required=True,
                       help="Répertoire contenant les images mal classées")
    parser.add_argument("--output_dir", type=str, default="error_analysis",
                       help="Répertoire pour sauvegarder le rapport d'analyse")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Nombre de paires de confusion à analyser en détail")
    
    args = parser.parse_args()
    
    # Vérifier que le répertoire des images incorrectes existe
    if not os.path.exists(args.incorrect_dir):
        print(f"Erreur: Le répertoire {args.incorrect_dir} n'existe pas!")
        return
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyse des erreurs dans {args.incorrect_dir}...")
    
    # Charger les informations sur les images mal classées
    misclassified = load_misclassified_images(args.incorrect_dir)
    print(f"Nombre d'images mal classées trouvées: {len(misclassified)}")
    
    if len(misclassified) == 0:
        print("Aucune image mal classée n'a été trouvée. Vérifiez le format des noms de fichiers.")
        return
    
    # Calculer la matrice de confusion
    df_confusion, all_classes, cm = analyze_confusion_matrix(misclassified)
    print(f"Nombre de classes impliquées dans les erreurs: {len(all_classes)}")
    
    # Créer une visualisation de la matrice de confusion
    create_confusion_matrix_plot(cm, all_classes, args.output_dir)
    
    # Identifier les motifs de confusion récurrents
    df_patterns = identify_common_confusion_patterns(df_confusion)
    
    # Créer une grille d'exemples d'erreurs
    create_error_examples_grid(misclassified, args.output_dir, args.top_n)
    
    # Générer un rapport HTML
    generate_html_report(df_confusion, df_patterns, args.output_dir)
    
    # Sauvegarder les DataFrames au format CSV
    if not df_confusion.empty:
        confusion_csv_path = os.path.join(args.output_dir, 'confusion_statistics.csv')
        df_confusion.to_csv(confusion_csv_path, index=False)
        print(f"Statistiques de confusion sauvegardées dans {confusion_csv_path}")
    
    if not df_patterns.empty:
        patterns_csv_path = os.path.join(args.output_dir, 'confusion_patterns.csv')
        df_patterns.to_csv(patterns_csv_path, index=False)
        print(f"Motifs de confusion sauvegardés dans {patterns_csv_path}")
    
    print("\nAnalyse des erreurs terminée!")
    print(f"Consultez le rapport complet dans {os.path.join(args.output_dir, 'error_analysis_report.html')}")

if __name__ == "__main__":
    main() 