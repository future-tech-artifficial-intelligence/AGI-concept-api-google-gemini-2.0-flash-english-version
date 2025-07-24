#!/usr/bin/env python3
"""
Test de robustesse de l'interface Searx
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from searx_interface import SearxInterface
import time

def test_searx_robustness():
    """Test de la robustesse de Searx"""
    
    print("🔧 Test de robustesse de l'interface Searx")
    print("=" * 60)
    
    # Créer une instance de SearxInterface
    searx = SearxInterface()
    
    # 1. Test de statut de base
    print("\n1. Test de statut Searx:")
    status = searx.check_health()
    print(f"   Searx opérationnel: {'✅ Oui' if status else '❌ Non'}")
    
    if not status:
        print("   Tentative de démarrage de Searx...")
        start_result = searx.start_searx()
        print(f"   Démarrage réussi: {'✅ Oui' if start_result else '❌ Non'}")
        
        if start_result:
            print("   Attente de 10 secondes pour que Searx soit prêt...")
            time.sleep(10)
            status = searx.check_health()
            print(f"   Searx opérationnel après démarrage: {'✅ Oui' if status else '❌ Non'}")
    
    # 2. Test de recherche simple
    print("\n2. Test de recherche simple:")
    query = "test simple"
    print(f"   Recherche: '{query}'")
    
    results = searx.search(query, max_results=3)
    print(f"   Résultats obtenus: {len(results)}")
    
    for i, result in enumerate(results[:2], 1):
        print(f"   Résultat {i}: {result.title[:50]}...")
        print(f"   URL: {result.url}")
    
    # 3. Test de recherche avec retry
    print("\n3. Test de recherche avec retry:")
    query2 = "intelligence artificielle actualités"
    print(f"   Recherche: '{query2}'")
    
    start_time = time.time()
    results2 = searx.search(query2, category="general", max_results=5, retry_count=3)
    end_time = time.time()
    
    print(f"   Résultats obtenus: {len(results2)}")
    print(f"   Temps de recherche: {end_time - start_time:.2f} secondes")
    
    if results2:
        print("   ✅ Recherche avec retry réussie")
        for i, result in enumerate(results2[:2], 1):
            print(f"   Résultat {i}: {result.title[:50]}...")
            print(f"   URL: {result.url[:80]}...")
    else:
        print("   ❌ Aucun résultat obtenu même avec retry")
    
    print("\n✅ Test de robustesse terminé!")

if __name__ == "__main__":
    test_searx_robustness()
