#!/usr/bin/env python
"""
Script pour exécuter tous les tests d'intégration
"""

import unittest
import sys
import os

# Ajouter le dossier parent au chemin pour pouvoir importer les modules de l'application
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    # Découvrir et exécuter tous les tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__), pattern='test_*.py')
    
    # Exécuter les tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Afficher un rapport de couverture
    print("\nRésumé des tests:")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Erreurs: {len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    
    # Sortir avec un code d'erreur si des tests ont échoué
    sys.exit(len(result.errors) + len(result.failures)) 