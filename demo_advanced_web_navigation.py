"""
Script de Test et Démonstration du Système de Navigation Web Avancé
Ce script montre les capacités du système avec des exemples concrets
"""

import logging
import json
import time
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_header(title):
    """Affiche un header de démonstration"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def demo_content_extraction():
    """Démonstraton d'extraction de contenu"""
    demo_header("EXTRACTION DE CONTENU WEB")
    
    try:
        from advanced_web_navigator import extract_website_content
        
        test_urls = [
            "https://httpbin.org/json",
            "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
            "https://www.python.org"
        ]
        
        for url in test_urls:
            print(f"\n🔍 Extraction depuis: {url}")
            
            start_time = time.time()
            content = extract_website_content(url)
            extraction_time = time.time() - start_time
            
            if content.success:
                print(f"✅ Extraction réussie en {extraction_time:.2f}s")
                print(f"  📄 Titre: {content.title}")
                print(f"  📊 Contenu: {len(content.cleaned_text)} caractères")
                print(f"  🏆 Score qualité: {content.content_quality_score}/10")
                print(f"  🌐 Langue: {content.language}")
                print(f"  🔗 Liens trouvés: {len(content.links)}")
                print(f"  🖼️ Images: {len(content.images)}")
                print(f"  🔑 Mots-clés: {', '.join(content.keywords[:5])}")
                
                if content.summary:
                    print(f"  📝 Résumé: {content.summary[:150]}...")
            else:
                print(f"❌ Extraction échouée: {content.error_message}")
            
            time.sleep(1)  # Délai entre les requêtes
            
    except Exception as e:
        logger.error(f"Erreur lors de la démo d'extraction: {str(e)}")

def demo_navigation_detection():
    """Démonstraton de détection de navigation"""
    demo_header("DÉTECTION DE NAVIGATION AUTOMATIQUE")
    
    try:
        from gemini_navigation_adapter import detect_navigation_need, initialize_gemini_navigation_adapter
        
        # Initialiser l'adaptateur
        initialize_gemini_navigation_adapter()
        
        test_prompts = [
            "Recherche et navigue sur l'intelligence artificielle",
            "Extrait le contenu de https://example.com/article",
            "Explore le site https://wikipedia.org en profondeur avec 3 niveaux",
            "Simule un parcours d'achat sur https://shop.example.com",
            "Qu'est-ce que l'apprentissage automatique ?",
            "Peux-tu m'aider avec Python ?",
            "Navigue dans le site de documentation Python",
            "Comment faire une recherche Google ?",
            "Analyse cette page web complètement",
            "Trouve des informations sur les voitures électriques"
        ]
        
        print("🧪 Test de détection de navigation sur différents prompts:\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            detection = detect_navigation_need(prompt)
            
            requires_nav = detection.get('requires_navigation', False)
            nav_type = detection.get('navigation_type', 'aucun')
            confidence = detection.get('confidence', 0)
            params = detection.get('extracted_params', {})
            
            status_icon = "🟢" if requires_nav else "🔴"
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            
            print(f"{i:2}. {status_icon} '{prompt}'")
            print(f"     Type: {nav_type} | Confiance: {confidence:.1f} [{confidence_bar}]")
            
            if params:
                print(f"     Paramètres: {params}")
            print()
            
    except Exception as e:
        logger.error(f"Erreur lors de la démo de détection: {str(e)}")

def demo_deep_navigation():
    """Démonstraton de navigation profonde"""
    demo_header("NAVIGATION PROFONDE DE SITE WEB")
    
    try:
        from advanced_web_navigator import navigate_website_deep
        
        # Test avec un site simple
        test_url = "https://httpbin.org"
        
        print(f"🚀 Navigation profonde depuis: {test_url}")
        print("   Paramètres: max_depth=2, max_pages=5")
        
        start_time = time.time()
        nav_path = navigate_website_deep(test_url, max_depth=2, max_pages=5)
        navigation_time = time.time() - start_time
        
        print(f"\n✅ Navigation terminée en {navigation_time:.2f}s")
        print(f"📊 Statistiques de navigation:")
        print(f"  - Pages visitées: {len(nav_path.visited_pages)}")
        print(f"  - Profondeur atteinte: {nav_path.navigation_depth}")
        print(f"  - Contenu total extrait: {nav_path.total_content_extracted} caractères")
        print(f"  - Stratégie: {nav_path.navigation_strategy}")
        print(f"  - Session ID: {nav_path.session_id}")
        
        if nav_path.visited_pages:
            print(f"\n📄 Pages explorées:")
            for i, page in enumerate(nav_path.visited_pages, 1):
                print(f"  {i}. {page.title} (Score: {page.content_quality_score:.1f})")
                print(f"     URL: {page.url}")
                print(f"     Contenu: {len(page.cleaned_text)} caractères")
                if page.keywords:
                    print(f"     Mots-clés: {', '.join(page.keywords[:3])}")
                print()
                
    except Exception as e:
        logger.error(f"Erreur lors de la démo de navigation: {str(e)}")

def demo_gemini_integration():
    """Démonstraton d'intégration Gemini"""
    demo_header("INTÉGRATION GEMINI COMPLÈTE")
    
    try:
        from gemini_web_integration import search_web_for_gemini, initialize_gemini_web_integration
        
        # Initialiser l'intégration
        initialize_gemini_web_integration()
        
        # Test de recherche simple
        query = "intelligence artificielle 2024"
        user_context = "développeur cherchant les dernières tendances"
        
        print(f"🔍 Recherche et navigation pour Gemini:")
        print(f"   Requête: '{query}'")
        print(f"   Contexte: {user_context}")
        
        start_time = time.time()
        result = search_web_for_gemini(query, user_context)
        processing_time = time.time() - start_time
        
        print(f"\n⏱️ Traitement terminé en {processing_time:.2f}s")
        
        if result.get('success', False):
            print("✅ Recherche et navigation réussies!")
            
            search_summary = result.get('search_summary', {})
            print(f"\n📊 Résumé de la recherche:")
            print(f"  - Sites recherchés: {search_summary.get('sites_searched', 0)}")
            print(f"  - Sites navigués: {search_summary.get('sites_navigated', 0)}")
            print(f"  - Pages visitées: {search_summary.get('total_pages_visited', 0)}")
            print(f"  - Pages de haute qualité: {search_summary.get('high_quality_pages', 0)}")
            
            if 'content_synthesis' in result:
                print(f"\n📝 Synthèse du contenu:")
                synthesis = result['content_synthesis']
                print(f"   {synthesis[:300]}{'...' if len(synthesis) > 300 else ''}")
            
            if 'aggregated_keywords' in result and result['aggregated_keywords']:
                keywords = ', '.join(result['aggregated_keywords'][:10])
                print(f"\n🔑 Mots-clés identifiés: {keywords}")
            
            if 'navigation_insights' in result:
                print(f"\n💡 Insights de navigation:")
                for insight in result['navigation_insights'][:3]:
                    print(f"   • {insight}")
            
            if 'recommended_actions' in result:
                print(f"\n💭 Recommandations:")
                for action in result['recommended_actions'][:3]:
                    print(f"   • {action}")
                    
        else:
            print(f"❌ Recherche échouée: {result.get('reason', 'Erreur inconnue')}")
            
    except Exception as e:
        logger.error(f"Erreur lors de la démo d'intégration Gemini: {str(e)}")

def demo_api_endpoints():
    """Démonstraton des endpoints API"""
    demo_header("API REST - ENDPOINTS")
    
    try:
        from web_navigation_api import register_web_navigation_api, initialize_web_navigation_api
        from flask import Flask
        
        # Créer une app Flask de test
        app = Flask(__name__)
        register_web_navigation_api(app)
        initialize_web_navigation_api()
        
        print("🌐 Test des endpoints API REST:\n")
        
        with app.test_client() as client:
            # Test 1: Health check
            print("1. 🏥 Health Check")
            response = client.get('/api/web-navigation/health')
            if response.status_code == 200:
                health_data = response.get_json()
                status = health_data.get('overall_status', 'unknown')
                print(f"   Status: {status}")
                
                components = health_data.get('components', {})
                for component, comp_status in components.items():
                    if component != 'timestamp':
                        icon = "✅" if comp_status == 'healthy' else "⚠️"
                        print(f"   {icon} {component}: {comp_status}")
            else:
                print(f"   ❌ Erreur: {response.status_code}")
            
            # Test 2: Documentation
            print("\n2. 📚 Documentation API")
            response = client.get('/api/web-navigation/docs')
            if response.status_code == 200:
                docs = response.get_json()
                print(f"   ✅ Documentation disponible")
                print(f"   API: {docs.get('api_name', 'N/A')}")
                print(f"   Version: {docs.get('version', 'N/A')}")
                
                endpoints = docs.get('endpoints', {})
                print(f"   Endpoints disponibles: {len(endpoints)}")
            else:
                print(f"   ❌ Erreur: {response.status_code}")
            
            # Test 3: Statistiques
            print("\n3. 📊 Statistiques")
            response = client.get('/api/web-navigation/stats')
            if response.status_code == 200:
                stats_data = response.get_json()
                if stats_data.get('success', False):
                    stats = stats_data.get('stats', {})
                    api_stats = stats.get('api_stats', {})
                    
                    print(f"   ✅ Statistiques récupérées")
                    print(f"   Sessions actives: {stats.get('active_sessions', 0)}")
                    print(f"   Taille du cache: {stats.get('cache_size', 0)}")
                    print(f"   Taux de cache hit: {stats.get('cache_hit_rate', 0):.1f}%")
                    print(f"   Recherches totales: {api_stats.get('total_searches', 0)}")
                else:
                    print(f"   ❌ Erreur dans les données")
            else:
                print(f"   ❌ Erreur: {response.status_code}")
            
            # Test 4: Création de session
            print("\n4. 🔐 Création de Session")
            session_data = {
                "user_id": "demo_user",
                "config": {
                    "max_depth": 2,
                    "max_pages": 5
                }
            }
            
            response = client.post('/api/web-navigation/create-session', json=session_data)
            if response.status_code == 200:
                result = response.get_json()
                if result.get('success', False):
                    session_id = result.get('session_id')
                    print(f"   ✅ Session créée: {session_id}")
                    print(f"   Configuration: {result.get('config', {})}")
                    
                    # Test 5: Info de session
                    print("\n5. ℹ️ Informations de Session")
                    response = client.get(f'/api/web-navigation/session/{session_id}')
                    if response.status_code == 200:
                        session_info = response.get_json()
                        if session_info.get('success', False):
                            print(f"   ✅ Session trouvée")
                            print(f"   Utilisateur: {session_info.get('user_id')}")
                            print(f"   Créée le: {session_info.get('created_at', 'N/A')[:19]}")
                            print(f"   Requêtes: {session_info.get('requests_count', 0)}")
                            print(f"   Active: {session_info.get('is_active', False)}")
                        else:
                            print(f"   ❌ Session non trouvée")
                    else:
                        print(f"   ❌ Erreur: {response.status_code}")
                else:
                    print(f"   ❌ Erreur: {result.get('error', 'Inconnue')}")
            else:
                print(f"   ❌ Erreur: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Erreur lors de la démo API: {str(e)}")

def demo_performance_test():
    """Test de performance simple"""
    demo_header("TEST DE PERFORMANCE")
    
    try:
        from advanced_web_navigator import extract_website_content
        
        test_urls = [
            "https://httpbin.org/json",
            "https://httpbin.org/html",
            "https://httpbin.org/robots.txt"
        ]
        
        print("⚡ Test de performance sur plusieurs URLs:\n")
        
        total_start = time.time()
        results = []
        
        for i, url in enumerate(test_urls, 1):
            print(f"{i}. Test: {url}")
            
            start_time = time.time()
            content = extract_website_content(url)
            end_time = time.time()
            
            processing_time = end_time - start_time
            results.append({
                'url': url,
                'success': content.success,
                'time': processing_time,
                'content_length': len(content.cleaned_text) if content.success else 0,
                'quality_score': content.content_quality_score if content.success else 0
            })
            
            status = "✅" if content.success else "❌"
            print(f"   {status} Temps: {processing_time:.2f}s | "
                  f"Contenu: {len(content.cleaned_text)} chars | "
                  f"Qualité: {content.content_quality_score:.1f}")
            
            time.sleep(0.5)  # Petit délai entre les requêtes
        
        total_time = time.time() - total_start
        
        print(f"\n📊 Résumé de performance:")
        print(f"   Temps total: {total_time:.2f}s")
        print(f"   Temps moyen par URL: {total_time/len(test_urls):.2f}s")
        
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_content = sum(r['content_length'] for r in successful_results) / len(successful_results)
            avg_quality = sum(r['quality_score'] for r in successful_results) / len(successful_results)
            print(f"   Contenu moyen: {avg_content:.0f} caractères")
            print(f"   Qualité moyenne: {avg_quality:.1f}/10")
        
        success_rate = (len(successful_results) / len(results)) * 100
        print(f"   Taux de succès: {success_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"Erreur lors du test de performance: {str(e)}")

def main():
    """Fonction principale de démonstration"""
    print("🌟 DÉMONSTRATION DU SYSTÈME DE NAVIGATION WEB AVANCÉ")
    print("=" * 70)
    print(f"⏰ Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    demos = [
        ("Extraction de Contenu", demo_content_extraction),
        ("Détection de Navigation", demo_navigation_detection),
        ("Navigation Profonde", demo_deep_navigation),
        ("Intégration Gemini", demo_gemini_integration),
        ("Endpoints API", demo_api_endpoints),
        ("Test de Performance", demo_performance_test)
    ]
    
    print("\n🎯 Démonstrations disponibles:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 70)
    
    try:
        choice = input("\n🔢 Choisissez une démo (1-6) ou 'all' pour toutes: ").strip().lower()
        
        if choice == 'all':
            for name, demo_func in demos:
                print(f"\n🚀 Démarrage: {name}")
                demo_func()
                print(f"✅ Terminé: {name}")
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            demo_index = int(choice) - 1
            name, demo_func = demos[demo_index]
            print(f"\n🚀 Démarrage: {name}")
            demo_func()
            print(f"✅ Terminé: {name}")
        else:
            print("❌ Choix invalide. Exécution de toutes les démos...")
            for name, demo_func in demos:
                demo_func()
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Démonstration interrompue par l'utilisateur")
    
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration: {str(e)}")
    
    finally:
        print("\n" + "=" * 70)
        print("🎉 DÉMONSTRATION TERMINÉE")
        print("📚 Consultez ADVANCED_WEB_NAVIGATION_DOCUMENTATION.md pour plus d'infos")
        print("=" * 70)

if __name__ == "__main__":
    main()
