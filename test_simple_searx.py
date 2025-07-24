#!/usr/bin/env python3
"""
Test simple de validation de l'intégration Searx
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api import get_gemini_response, get_searx_status

def test_simple_integration():
    """Test simple de l'intégration Searx"""
    
    print("🔧 Test de validation simple de l'intégration Searx")
    print("=" * 60)
    
    # 1. Vérifier le statut
    print("\n1. Statut Searx:")
    status = get_searx_status()
    print(f"   {status}")
    
    # 2. Test d'une question simple qui déclenche Searx
    print("\n2. Test avec question déclenchant Searx:")
    test_question = "Quelles sont les dernières nouvelles en technologie ?"
    
    print(f"   Question: {test_question}")
    print("   Traitement en cours...")
    
    response = get_gemini_response(test_question)
    
    print(f"   Statut: {response['status']}")
    print(f"   Longueur réponse: {len(response['response'])} caractères")
    print(f"   Début de la réponse: {response['response'][:200]}...")
    
    # 3. Test d'une question normale sans Searx
    print("\n3. Test avec question normale (sans Searx):")
    normal_question = "Comment allez-vous aujourd'hui ?"
    
    print(f"   Question: {normal_question}")
    response2 = get_gemini_response(normal_question)
    
    print(f"   Statut: {response2['status']}")
    print(f"   Longueur réponse: {len(response2['response'])} caractères")
    print(f"   Début de la réponse: {response2['response'][:200]}...")
    
    print("\n✅ Tests de validation terminés!")

if __name__ == "__main__":
    test_simple_integration()
