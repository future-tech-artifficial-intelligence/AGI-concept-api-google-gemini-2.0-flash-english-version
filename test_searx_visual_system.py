#!/usr/bin/env python3
"""
Script de test pour le système de capture visuelle Searx
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
logger = logging.getLogger('searx_visual_test')

def test_dependencies():
    """Test des dépendances nécessaires"""
    logger.info("🔍 Test 1: Vérification des dépendances...")
    
    missing_deps = []
    
    try:
        import selenium
        logger.info("✅ Selenium disponible")
    except ImportError:
        missing_deps.append("selenium")
    
    try:
        from PIL import Image
        logger.info("✅ Pillow disponible")
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import requests
        logger.info("✅ Requests disponible")
    except ImportError:
        missing_deps.append("requests")
    
    if missing_deps:
        logger.error(f"❌ Dépendances manquantes: {', '.join(missing_deps)}")
        return False
    
    logger.info("✅ Toutes les dépendances sont disponibles")
    return True

def test_webdriver_initialization():
    """Test de l'initialisation du WebDriver"""
    logger.info("🔍 Test 2: Initialisation du WebDriver...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        
        if capture._initialize_webdriver():
            logger.info("✅ WebDriver initialisé avec succès")
            capture.close()
            return True
        else:
            logger.error("❌ Échec de l'initialisation du WebDriver")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du test WebDriver: {e}")
        return False

def test_searx_accessibility():
    """Test de l'accessibilité de Searx"""
    logger.info("🔍 Test 3: Accessibilité de Searx...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:8080/", timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Searx est accessible")
            return True
        else:
            logger.warning(f"⚠️ Searx répond avec le code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Searx inaccessible: {e}")
        logger.error("Assurez-vous que Searx est démarré avec: start_with_searx.bat")
        return False

def test_visual_capture():
    """Test de la capture visuelle"""
    logger.info("🔍 Test 4: Capture visuelle...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        
        # Test de capture simple
        result = capture.capture_search_results("test python", category="general")
        
        if result and result.get('success'):
            logger.info(f"✅ Capture réussie: {result['screenshot_path']}")
            
            # Vérifier que le fichier existe
            if os.path.exists(result['screenshot_path']):
                logger.info("✅ Fichier de capture créé")
                
                # Vérifier la taille du fichier
                file_size = os.path.getsize(result['screenshot_path'])
                logger.info(f"📄 Taille du fichier: {file_size} bytes")
                
                if file_size > 1000:  # Au moins 1KB
                    logger.info("✅ Fichier de capture valide")
                    capture.close()
                    return True
                else:
                    logger.warning("⚠️ Fichier de capture trop petit")
            else:
                logger.error("❌ Fichier de capture non créé")
        else:
            logger.error(f"❌ Échec de la capture: {result.get('error', 'Erreur inconnue')}")
        
        capture.close()
        return False
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test de capture: {e}")
        return False

def test_visual_annotation():
    """Test des annotations visuelles"""
    logger.info("🔍 Test 5: Annotations visuelles...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        
        # Test de capture avec annotations
        result = capture.capture_with_annotations("intelligence artificielle", category="general")
        
        if result and result.get('success') and result.get('has_annotations'):
            logger.info("✅ Capture avec annotations réussie")
            
            if result.get('annotated_image'):
                logger.info("✅ Image annotée générée (base64)")
            
            capture.close()
            return True
        else:
            logger.error("❌ Échec de la capture avec annotations")
            capture.close()
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du test d'annotation: {e}")
        return False

def test_integration_with_searx_interface():
    """Test de l'intégration avec l'interface Searx"""
    logger.info("🔍 Test 6: Intégration avec SearxInterface...")
    
    try:
        from searx_interface import get_searx_interface
        
        searx = get_searx_interface()
        
        # Test de recherche avec visuel
        if hasattr(searx, 'search_with_visual'):
            result = searx.search_with_visual("test recherche", category="general", max_results=3)
            
            if result.get('has_visual'):
                logger.info("✅ Recherche avec capture visuelle réussie")
                
                # Test du résumé
                summary = searx.get_visual_search_summary(result)
                if summary and len(summary) > 100:
                    logger.info("✅ Résumé visuel généré")
                    return True
                else:
                    logger.warning("⚠️ Résumé visuel trop court")
            else:
                logger.warning("⚠️ Pas de données visuelles dans le résultat")
        else:
            logger.error("❌ Méthode search_with_visual non disponible")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du test d'intégration: {e}")
        return False

def test_cleanup():
    """Test du nettoyage"""
    logger.info("🔍 Test 7: Nettoyage...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        capture.cleanup_old_screenshots(max_age_hours=0)  # Nettoyer tout
        
        # Vérifier le répertoire
        screenshots_dir = capture.screenshots_dir
        if os.path.exists(screenshots_dir):
            remaining_files = len([f for f in os.listdir(screenshots_dir) 
                                 if f.endswith('.png')])
            logger.info(f"📁 Fichiers restants: {remaining_files}")
        
        logger.info("✅ Nettoyage terminé")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage: {e}")
        return False

def main():
    """Exécute tous les tests"""
    logger.info("🧪 TESTS DU SYSTÈME DE CAPTURE VISUELLE SEARX")
    logger.info("=" * 70)
    
    tests = [
        ("Dépendances", test_dependencies),
        ("WebDriver", test_webdriver_initialization),
        ("Accessibilité Searx", test_searx_accessibility),
        ("Capture visuelle", test_visual_capture),
        ("Annotations", test_visual_annotation),
        ("Intégration", test_integration_with_searx_interface),
        ("Nettoyage", test_cleanup)
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
        time.sleep(1)
    
    # Résumé final
    logger.info("\n" + "=" * 70)
    logger.info("📊 RÉSUMÉ DES TESTS")
    logger.info("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nRésultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 Tous les tests sont passés ! Le système de capture visuelle est opérationnel.")
        logger.info("\n💡 L'IA peut maintenant voir les résultats de recherche comme un humain !")
        return True
    elif passed > total // 2:
        logger.warning(f"⚠️ {passed} tests réussis sur {total}. Le système est partiellement fonctionnel.")
        return True
    else:
        logger.error("💥 Échec critique. Le système de capture visuelle ne fonctionne pas.")
        logger.error("\n🔧 Solutions possibles:")
        logger.error("1. Installer les dépendances: python install_searx_visual_deps.py")
        logger.error("2. Démarrer Searx: start_with_searx.bat")
        logger.error("3. Vérifier que Chrome ou Edge est installé")
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
