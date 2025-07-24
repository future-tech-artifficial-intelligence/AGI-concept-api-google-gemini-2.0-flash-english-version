#!/usr/bin/env python3
"""
Test spécifique pour les URLs vidéo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api import get_gemini_response, trigger_searx_search_session

def test_video_urls():
    """Test spécifique pour les URLs vidéo"""
    
    print("🎥 Test des URLs vidéo avec Searx + Gemini")
    print("=" * 60)
    
    # 1. Test de recherche vidéo directe
    print("\n1. Test de recherche vidéo Searx:")
    video_search = trigger_searx_search_session("ovni videos témoignages", "videos")
    print(f"   Résultat: {video_search}")
    
    # 2. Test avec Gemini - question sur OVNI qui devrait déclencher une recherche
    print("\n2. Test Gemini avec question OVNI:")
    test_question = "Trouve-moi des vidéos récentes sur les observations d'OVNI"
    
    print(f"   Question: {test_question}")
    print("   Traitement en cours...")
    
    response = get_gemini_response(test_question)
    
    print(f"   Statut: {response['status']}")
    print(f"   Longueur réponse: {len(response['response'])} caractères")
    print(f"\n   Réponse complète:")
    print(f"   {response['response']}")
    
    # Vérifier si la réponse contient des "xxxxxxxxxx"
    if "xxxxxxxxxx" in response['response']:
        print("\n   ❌ PROBLÈME: La réponse contient encore des 'xxxxxxxxxx'")
    else:
        print("\n   ✅ Aucun 'xxxxxxxxxx' détecté dans la réponse")
    
    print("\n✅ Test des URLs vidéo terminé!")

if __name__ == "__main__":
    test_video_urls()
