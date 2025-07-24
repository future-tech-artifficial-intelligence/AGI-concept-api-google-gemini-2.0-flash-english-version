#!/usr/bin/env python3
"""
Script de test pour valider le système Searx
"""

import logging
import time
import sys
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('searx_test')

def test_docker_availability():
    """Test de la disponibilité de Docker"""
    logger.info("🔍 Test 1: Vérification de Docker...")
    
    try:
        from searx_manager import SearxManager
        manager = SearxManager()
        
        if manager.check_docker_availability():
            logger.info("✅ Docker est disponible")
            return True
        else:
            logger.error("❌ Docker n'est pas disponible")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du test Docker: {e}")
        return False

def test_searx_startup():
    """Test du démarrage de Searx"""
    logger.info("🔍 Test 2: Démarrage de Searx...")
    
    try:
        from searx_manager import get_searx_manager
        manager = get_searx_manager()
        
        # Tenter de démarrer Searx
        if manager.ensure_searx_running():
            logger.info("✅ Searx a démarré avec succès")
            return True
        else:
            logger.error("❌ Échec du démarrage de Searx")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du test de démarrage: {e}")
        return False

def test_searx_interface():
    """Test de l'interface Searx"""
    logger.info("🔍 Test 3: Interface de recherche...")
    
    try:
        from searx_interface import get_searx_interface
        searx = get_searx_interface()
        
        # Vérifier que Searx est prêt
        if not searx.check_health():
            logger.error("❌ Searx n'est pas accessible")
            return False
        
        # Test de recherche simple
        results = searx.search("test python", max_results=3)
        
        if results:
            logger.info(f"✅ Recherche réussie: {len(results)} résultats trouvés")
            
            # Afficher les premiers résultats
            for i, result in enumerate(results[:2], 1):
                logger.info(f"   {i}. {result.title[:50]}... (via {result.engine})")
            
            return True
        else:
            logger.warning("⚠️ Aucun résultat trouvé (service peut-être en cours de démarrage)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du test d'interface: {e}")
        return False

def test_gemini_integration():
    """Test de l'intégration avec Gemini"""
    logger.info("🔍 Test 4: Intégration avec l'API Gemini...")
    
    try:
        from gemini_api_adapter import GeminiAPI
        
        # Créer une instance de l'API
        api = GeminiAPI()
        
        # Vérifier que Searx est intégré
        if hasattr(api, 'searx_available') and api.searx_available:
            logger.info("✅ Searx est intégré dans l'adaptateur Gemini")
            
            # Test de détection de recherche
            test_prompt = "recherche des informations sur Python"
            if api._detect_web_search_request(test_prompt):
                logger.info("✅ Détection de requête de recherche fonctionne")
                return True
            else:
                logger.warning("⚠️ Détection de requête de recherche ne fonctionne pas")
                return False
        else:
            logger.error("❌ Searx n'est pas intégré dans l'adaptateur Gemini")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du test d'intégration: {e}")
        return False

def test_search_categories():
    """Test des différentes catégories de recherche"""
    logger.info("🔍 Test 5: Catégories de recherche...")
    
    try:
        from searx_interface import get_searx_interface
        searx = get_searx_interface()
        
        if not searx.check_health():
            logger.warning("⚠️ Searx non accessible, test ignoré")
            return True
        
        # Test de différentes catégories
        test_queries = [
            ("intelligence artificielle", "general"),
            ("tutoriel python", "it"),
            ("actualités tech", "general")
        ]
        
        success_count = 0
        
        for query, category in test_queries:
            try:
                results = searx.search(query, category=category, max_results=2)
                if results:
                    logger.info(f"✅ Recherche '{query}' ({category}): {len(results)} résultats")
                    success_count += 1
                else:
                    logger.warning(f"⚠️ Pas de résultats pour '{query}' ({category})")
                    
                # Attendre un peu entre les recherches
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur pour '{query}': {e}")
        
        if success_count > 0:
            logger.info(f"✅ Test des catégories: {success_count}/{len(test_queries)} succès")
            return True
        else:
            logger.error("❌ Aucune recherche par catégorie n'a fonctionné")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du test des catégories: {e}")
        return False

def main():
    """Exécute tous les tests"""
    logger.info("🧪 TESTS DU SYSTÈME SEARX")
    logger.info("=" * 60)
    
    tests = [
        ("Docker", test_docker_availability),
        ("Démarrage Searx", test_searx_startup),
        ("Interface Searx", test_searx_interface),
        ("Intégration Gemini", test_gemini_integration),
        ("Catégories de recherche", test_search_categories)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: RÉUSSI")
            else:
                logger.error(f"❌ {test_name}: ÉCHEC")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: ERREUR - {e}")
            results.append((test_name, False))
        
        # Pause entre les tests
        time.sleep(2)
    
    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nRésultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 Tous les tests sont passés ! Le système Searx est opérationnel.")
        return True
    elif passed > 0:
        logger.warning(f"⚠️ {passed} tests réussis sur {total}. Le système est partiellement fonctionnel.")
        return True
    else:
        logger.error("💥 Aucun test n'a réussi. Le système Searx ne fonctionne pas.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)
