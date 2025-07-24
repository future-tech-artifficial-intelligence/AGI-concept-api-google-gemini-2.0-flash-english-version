#!/usr/bin/env python3
"""
Script de test pour l'intégration Searx dans l'API Gemini
"""

import sys
import os

# Ajouter le répertoire actuel au chemin Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api import get_gemini_response, get_searx_status, trigger_searx_search_session

def test_searx_integration():
    """Test de l'intégration Searx"""
    print("🔧 Test de l'intégration Searx dans l'API Gemini")
    print("=" * 60)
    
    # Test 1: Statut de Searx
    print("\n1. Test du statut Searx:")
    status = get_searx_status()
    print(status)
    
    # Test 2: Recherche manuelle
    print("\n2. Test de recherche manuelle:")
    search_result = trigger_searx_search_session("dernières actualités IA")
    print(search_result)
    
    # Test 3: Requête nécessitant une recherche automatique
    print("\n3. Test de requête avec recherche automatique:")
    response = get_gemini_response("Quelles sont les dernières actualités en intelligence artificielle ?")
    print(f"Réponse: {response['response'][:200]}...")
    print(f"Statut: {response['status']}")
    
    # Test 4: Requête technique
    print("\n4. Test de requête technique:")
    response = get_gemini_response("Quelles sont les nouvelles fonctionnalités de Python 3.12 ?")
    print(f"Réponse: {response['response'][:200]}...")
    print(f"Statut: {response['status']}")
    
    print("\n✅ Tests terminés!")

if __name__ == "__main__":
    test_searx_integration()
