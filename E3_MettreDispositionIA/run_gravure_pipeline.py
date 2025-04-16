#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour exécuter le pipeline complet de traitement des gravures :
1. Augmentation des données
2. Équilibrage des classes sous-représentées
3. Entraînement du modèle avec les paramètres optimaux
4. Évaluation du modèle et analyse des erreurs
5. Lancement optionnel de l'application Streamlit
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main():
    # Analyser les arguments
    parser = argparse.ArgumentParser(description="Pipeline complet de traitement des gravures")
    parser.add_argument("--input_dir", type=str, default=os.path.join(current_dir, "data/raw_gravures"),
                      help="Répertoire contenant les images brutes de gravures")
    parser.add_argument("--output_dir", type=str, default=os.path.join(current_dir, "data/augmented_gravures"),
                      help="Répertoire pour stocker les images augmentées")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Nombre d'époques pour l'entraînement")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Taille du batch pour l'entraînement")
    parser.add_argument("--balance_classes", action="store_true", default=True,
                      help="Activer l'équilibrage des classes")
    parser.add_argument("--progressive_unfreeze", action="store_true", default=True,
                      help="Activer le dégel progressif des couches")
    parser.add_argument("--initial_freeze", type=float, default=0.8,
                      help="Pourcentage initial de couches à geler (0.8 = 80%)")
    parser.add_argument("--onecycle", action="store_true", default=True,
                      help="Utiliser le scheduler OneCycle")
    parser.add_argument("--mining_type", type=str, default="semi-hard",
                      help="Type de mining pour triplet loss (random, semi-hard, hard)")
    parser.add_argument("--min_threshold", type=int, default=10,
                      help="Nombre minimum d'images requis par classe")
    parser.add_argument("--target_count", type=int, default=20,
                      help="Nombre cible d'images après oversampling")
    parser.add_argument("--skip_augmentation", action="store_true",
                      help="Ignorer l'étape d'augmentation des données")
    parser.add_argument("--skip_oversampling", action="store_true",
                      help="Ignorer l'étape d'équilibrage des classes")
    parser.add_argument("--skip_evaluation", action="store_true",
                      help="Ignorer l'étape d'évaluation du modèle")
    parser.add_argument("--launch_streamlit", action="store_true",
                      help="Lancer l'application Streamlit après l'entraînement")
    
    args = parser.parse_args()
    
    # Vérifier que les répertoires requis existent
    if not os.path.exists(args.input_dir):
        print(f"Erreur: Le répertoire d'entrée {args.input_dir} n'existe pas!")
        os.makedirs(args.input_dir)
        print(f"Le répertoire {args.input_dir} a été créé. Veuillez y ajouter vos images de gravures.")
        print("Format requis: Un sous-répertoire par classe contenant les images.")
        return
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Chemin du modèle de sortie
    model_dir = os.path.join(current_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Répertoires d'évaluation
    evaluation_dir = os.path.join(model_dir, "evaluation_results")
    error_analysis_dir = os.path.join(model_dir, "error_analysis")
    incorrect_dir = os.path.join(model_dir, "incorrect_predictions")
    
    # Créer les répertoires d'évaluation
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(error_analysis_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    print(f"Répertoire d'entrée: {args.input_dir}")
    print(f"Répertoire de sortie: {args.output_dir}")
    print(f"Répertoire des modèles: {model_dir}")
    
    # Étape 1 : Augmentation des données
    if not args.skip_augmentation:
        print("\n1. AUGMENTATION DES DONNÉES")
        print("===========================")
        from app.data_augmentation import process_directory
        
        # Vider d'abord le répertoire de sortie
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        
        # Appliquer l'augmentation des données
        process_directory(args.input_dir, args.output_dir, num_augmentations=5)
        print("Augmentation des données terminée!")
    else:
        print("\nIgnorant l'étape d'augmentation des données...")
    
    # Étape 2: Équilibrage des classes sous-représentées
    if not args.skip_oversampling:
        print("\n2. ÉQUILIBRAGE DES CLASSES SOUS-REPRÉSENTÉES")
        print("==========================================")
        try:
            # Sauvegarde des arguments système
            sys_args = sys.argv
            
            # Configuration pour l'équilibrage des classes
            sys.argv = [
                sys.argv[0],
                "--raw_dir", args.input_dir,
                "--augmented_dir", args.output_dir,
                "--min_threshold", str(args.min_threshold),
                "--target_count", str(args.target_count)
            ]
            
            from app.oversample_small_classes import main as oversample_main
            print(f"Démarrage de l'équilibrage des classes avec les paramètres: {' '.join(sys.argv[1:])}")
            oversample_main()
            
            # Restaurer les arguments système
            sys.argv = sys_args
            
            print("Équilibrage des classes terminé!")
        except Exception as e:
            print(f"Erreur lors de l'équilibrage des classes: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Restaurer les arguments système si nécessaire
            sys.argv = sys_args
    else:
        print("\nIgnorant l'étape d'équilibrage des classes...")
    
    # Étape 3 : Entraînement du modèle avec les données augmentées
    print("\n3. ENTRAÎNEMENT DU MODÈLE")
    print("=======================")
    
    # Modification temporaire des arguments système pour le script d'entraînement
    sys_args = sys.argv
    sys.argv = [
        sys.argv[0],
        "--data_dir", args.output_dir,  # Utiliser les données augmentées
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--mining_type", args.mining_type,
        "--save_dir", model_dir  # Spécifier le répertoire de sauvegarde
    ]
    
    # Ajouter les flags optionnels
    if args.balance_classes:
        sys.argv.append("--balance_classes")
    if args.progressive_unfreeze:
        sys.argv.append("--progressive_unfreeze")
    if args.initial_freeze:
        sys.argv.extend(["--initial_freeze", str(args.initial_freeze)])
    if args.onecycle:
        sys.argv.append("--onecycle")
    
    # Import et exécution du script d'entraînement
    try:
        from app.train_efficientnet import main as train_main
        print(f"Démarrage de l'entraînement avec les paramètres: {' '.join(sys.argv[1:])}")
        train_main()
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restaurer les arguments système
        sys.argv = sys_args
    
    # Chemin du modèle final
    best_model_path = os.path.join(model_dir, "best_efficientnet_triplet.pt")
    
    # Étape 4: Évaluation du modèle
    if not args.skip_evaluation and os.path.exists(best_model_path):
        print("\n4. ÉVALUATION DU MODÈLE")
        print("=====================")
        try:
            # Évaluation des performances
            print("Évaluation des performances du modèle...")
            sys_args = sys.argv
            sys.argv = [
                sys.argv[0],
                "--model_path", best_model_path,
                "--model_type", "efficientnet",
                "--data_dir", args.output_dir,
                "--output_dir", evaluation_dir
            ]
            
            from app.evaluate_model import main as evaluate_main
            print(f"Démarrage de l'évaluation avec les paramètres: {' '.join(sys.argv[1:])}")
            evaluate_main()
            
            # Restaurer les arguments système
            sys.argv = sys_args
            
            # Analyse des erreurs si des prédictions incorrectes existent
            if os.path.exists(incorrect_dir) and len(os.listdir(incorrect_dir)) > 0:
                print("\nAnalyse des erreurs de classification...")
                sys_args = sys.argv
                sys.argv = [
                    sys.argv[0],
                    "--incorrect_dir", incorrect_dir,
                    "--output_dir", error_analysis_dir
                ]
                
                from app.analyze_errors import main as analyze_main
                print(f"Démarrage de l'analyse des erreurs avec les paramètres: {' '.join(sys.argv[1:])}")
                analyze_main()
                
                # Restaurer les arguments système
                sys.argv = sys_args
                
                print(f"Rapport d'analyse des erreurs généré dans: {error_analysis_dir}")
            else:
                print("Pas d'erreurs à analyser ou dossier de prédictions incorrectes vide.")
                
        except Exception as e:
            print(f"Erreur lors de l'évaluation: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Restaurer les arguments système
            sys.argv = sys_args
    else:
        if args.skip_evaluation:
            print("\nIgnorant l'étape d'évaluation du modèle...")
        elif not os.path.exists(best_model_path):
            print(f"\nImpossible d'évaluer le modèle: {best_model_path} n'existe pas.")
    
    # Étape 5: Lancement de l'application Streamlit
    if args.launch_streamlit:
        print("\n5. LANCEMENT DE L'APPLICATION STREAMLIT")
        print("===================================")
        try:
            # Vérifier si le script shell existe pour Linux/Mac
            streamlit_script = os.path.join(current_dir, "start_streamlit_app.sh")
            streamlit_app = os.path.join(current_dir, "streamlit_app.py")
            
            if os.path.exists(streamlit_script) and sys.platform in ['linux', 'darwin']:
                # Sous Linux/Mac
                print(f"Lancement de l'application via {streamlit_script}...")
                subprocess.Popen(["bash", streamlit_script])
            else:
                # Sous Windows ou si le script n'existe pas
                print("Lancement de l'application Streamlit...")
                subprocess.Popen(["streamlit", "run", streamlit_app])
            
            print("Application Streamlit lancée! Accès via http://localhost:8501")
        except Exception as e:
            print(f"Erreur lors du lancement de Streamlit: {str(e)}")
            print("Vous pouvez lancer manuellement l'application avec: streamlit run streamlit_app.py")
    
    print("\nPIPELINE TERMINÉ!")
    print(f"Modèle final enregistré dans: {best_model_path}")
    print("Vous pouvez maintenant utiliser ce modèle pour la reconnaissance de gravures.")
    
    if not args.launch_streamlit:
        print("\nPour lancer l'application:")
        if sys.platform in ['linux', 'darwin']:
            print("  bash start_streamlit_app.sh")
        else:
            print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 