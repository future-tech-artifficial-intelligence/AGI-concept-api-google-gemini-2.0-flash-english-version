"""
Script de lancement rapide pour tester les capacités d'interaction web de Gemini
Usage: python run_web_interaction_test.py
"""

import asyncio
import sys
import os
from pathlib import Path

def main():
    print("🌐 Test des Capacités d'Interaction Web de l'API Gemini")
    print("=" * 60)
    
    # Vérifier que nous sommes dans le bon répertoire
    current_dir = Path.cwd()
    required_files = [
        'test_gemini_web_interaction.py',
        'interactive_web_navigator.py',
        'gemini_api_adapter.py',
        'ai_api_config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        print("Assurez-vous d'être dans le répertoire racine du projet.")
        return 1
    
    print("✅ Tous les fichiers requis sont présents")
    print("🚀 Lancement du test d'interaction web...")
    print()
    
    try:
        # Importer et lancer le test
        from test_gemini_web_interaction import main as test_main
        asyncio.run(test_main())
        
        print()
        print("🎉 Tests terminés avec succès!")
        print("📁 Consultez le dossier 'test_results_web_interaction' pour les rapports détaillés")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrompu par l'utilisateur")
        return 1
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {str(e)}")
        print("Vérifiez :")
        print("1. Que votre clé API Gemini est configurée dans ai_api_config.json")
        print("2. Que Chrome/Chromium est installé pour Selenium")
        print("3. Que toutes les dépendances sont installées")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
