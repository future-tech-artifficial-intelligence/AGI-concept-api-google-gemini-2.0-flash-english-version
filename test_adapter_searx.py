#!/usr/bin/env python3
"""
Test de l'adaptateur Gemini avec Searx
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api_adapter import GeminiAPI

def test_adapter():
    """Test de l'adaptateur avec Searx"""
    
    print("🔧 Test de l'adaptateur Gemini avec Searx")
    print("=" * 60)
    
    # Créer une instance de l'adaptateur
    gemini_adapter = GeminiAPI()
    
    # Test avec une question qui devrait déclencher Searx
    print("\n1. Test avec question nécessitant Searx:")
    test_question = "Quelles sont les dernières actualités en intelligence artificielle ?"
    
    print(f"   Question: {test_question}")
    print("   Traitement en cours...")
    
    response = gemini_adapter.get_response(test_question, user_id=1, session_id="test_session")
    
    print(f"   Statut: {response['status']}")
    print(f"   Longueur réponse: {len(response['response'])} caractères")
    print(f"   Début de la réponse: {response['response'][:300]}...")
    
    # Vérifier si des URLs réelles sont présentes
    if "https://" in response['response'] or "http://" in response['response']:
        print("\n   ✅ URLs détectées dans la réponse")
    else:
        print("\n   ⚠️ Aucune URL détectée dans la réponse")
    
    print("\n✅ Test de l'adaptateur terminé!")

if __name__ == "__main__":
    test_adapter()
