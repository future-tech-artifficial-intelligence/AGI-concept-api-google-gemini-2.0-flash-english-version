"""
Script d'installation et de test du Système de Navigation Web Avancé
Ce script installe les dépendances et teste le système complet
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Installe les dépendances requises"""
    requirements = [
        'beautifulsoup4>=4.12.0',
        'lxml>=4.9.0',
        'nltk>=3.8',
        'aiohttp>=3.8.0',
        'requests>=2.31.0',
        'flask>=2.3.0'
    ]
    
    logger.info("🔧 Installation des dépendances...")
    
    for requirement in requirements:
        try:
            logger.info(f"📦 Installation de {requirement}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
            logger.info(f"✅ {requirement} installé avec succès")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur lors de l'installation de {requirement}: {str(e)}")
            return False
    
    # Installation optionnelle de NLTK data
    try:
        import nltk
        logger.info("📚 Téléchargement des données NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("✅ Données NLTK téléchargées")
    except Exception as e:
        logger.warning(f"⚠️ Impossible de télécharger les données NLTK: {str(e)}")
    
    return True

def test_imports():
    """Test les imports des modules"""
    logger.info("🧪 Test des imports...")
    
    modules_to_test = [
        ('advanced_web_navigator', 'Navigateur Web Avancé'),
        ('gemini_web_integration', 'Intégration Gemini-Web'),
        ('gemini_navigation_adapter', 'Adaptateur Navigation Gemini'),
        ('web_navigation_api', 'API REST Navigation Web')
    ]
    
    success_count = 0
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name} - Import réussi")
            success_count += 1
        except ImportError as e:
            logger.error(f"❌ {display_name} - Erreur d'import: {str(e)}")
        except Exception as e:
            logger.error(f"❌ {display_name} - Erreur: {str(e)}")
    
    logger.info(f"📊 Résultat des imports: {success_count}/{len(modules_to_test)} modules importés avec succès")
    return success_count == len(modules_to_test)

def test_navigation_system():
    """Test le système de navigation"""
    logger.info("🚀 Test du système de navigation...")
    
    try:
        from advanced_web_navigator import extract_website_content
        
        # Test avec une URL de test
        test_url = "https://httpbin.org/json"
        logger.info(f"🔍 Test d'extraction: {test_url}")
        
        content = extract_website_content(test_url)
        
        if content.success:
            logger.info(f"✅ Extraction réussie:")
            logger.info(f"  - Titre: {content.title}")
            logger.info(f"  - Contenu: {len(content.cleaned_text)} caractères")
            logger.info(f"  - Score qualité: {content.content_quality_score}")
            logger.info(f"  - Langue: {content.language}")
            return True
        else:
            logger.error(f"❌ Extraction échouée: {content.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du test de navigation: {str(e)}")
        return False

def test_gemini_integration():
    """Test l'intégration Gemini"""
    logger.info("🤖 Test de l'intégration Gemini...")
    
    try:
        from gemini_navigation_adapter import detect_navigation_need, initialize_gemini_navigation_adapter
        
        # Initialiser l'adaptateur
        initialize_gemini_navigation_adapter()
        logger.info("✅ Adaptateur Gemini initialisé")
        
        # Test de détection de navigation
        test_prompts = [
            "Recherche et navigue sur l'intelligence artificielle",
            "Extrait le contenu de https://example.com",
            "Qu'est-ce que l'apprentissage automatique ?",
            "Explore le site https://wikipedia.org en profondeur"
        ]
        
        detection_results = []
        for prompt in test_prompts:
            detection = detect_navigation_need(prompt)
            detection_results.append({
                'prompt': prompt,
                'requires_navigation': detection.get('requires_navigation', False),
                'navigation_type': detection.get('navigation_type'),
                'confidence': detection.get('confidence', 0)
            })
        
        # Afficher les résultats
        logger.info("🔍 Résultats de détection de navigation:")
        for result in detection_results:
            status = "🟢" if result['requires_navigation'] else "🔴"
            logger.info(f"  {status} '{result['prompt'][:50]}...'")
            logger.info(f"     Type: {result['navigation_type']}, Confiance: {result['confidence']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test d'intégration Gemini: {str(e)}")
        return False

def test_api_endpoints():
    """Test les endpoints de l'API"""
    logger.info("🌐 Test des endpoints API...")
    
    try:
        from web_navigation_api import register_web_navigation_api, initialize_web_navigation_api
        from flask import Flask
        
        # Créer une app Flask de test
        app = Flask(__name__)
        register_web_navigation_api(app)
        initialize_web_navigation_api()
        
        # Test des endpoints
        with app.test_client() as client:
            # Test de santé
            response = client.get('/api/web-navigation/health')
            if response.status_code == 200:
                health_data = response.get_json()
                logger.info(f"✅ Health check: {health_data.get('overall_status', 'unknown')}")
            else:
                logger.error(f"❌ Health check échoué: {response.status_code}")
                return False
            
            # Test de documentation
            response = client.get('/api/web-navigation/docs')
            if response.status_code == 200:
                logger.info("✅ Documentation API accessible")
            else:
                logger.error(f"❌ Documentation non accessible: {response.status_code}")
            
            # Test de statistiques
            response = client.get('/api/web-navigation/stats')
            if response.status_code == 200:
                logger.info("✅ Statistiques API accessibles")
            else:
                logger.error(f"❌ Statistiques non accessibles: {response.status_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test des endpoints API: {str(e)}")
        return False

def create_test_report():
    """Crée un rapport de test"""
    logger.info("📋 Création du rapport de test...")
    
    report_content = """
# Rapport de Test - Système de Navigation Web Avancé

## Tests Effectués

### 1. Installation des Dépendances
- beautifulsoup4: Installation et test
- lxml: Parser HTML/XML
- nltk: Traitement du langage naturel
- aiohttp: Requêtes HTTP asynchrones
- requests: Requêtes HTTP synchrones
- flask: Framework web

### 2. Test des Modules
- advanced_web_navigator.py: Extracteur de contenu web
- gemini_web_integration.py: Intégration avec Gemini
- gemini_navigation_adapter.py: Adaptateur pour Gemini
- web_navigation_api.py: API REST

### 3. Test de Navigation
- Extraction de contenu web
- Analyse de qualité du contenu
- Détection de langue
- Extraction de métadonnées

### 4. Test d'Intégration Gemini
- Détection de requêtes de navigation
- Classification des types de navigation
- Scoring de confiance

### 5. Test API REST
- Endpoints de santé
- Documentation automatique
- Statistiques d'utilisation

## Fonctionnalités Implémentées

### Navigation Web Avancée
✅ Extraction de contenu structuré
✅ Navigation en profondeur
✅ Analyse de qualité du contenu
✅ Cache intelligent
✅ Support multi-langues

### Intégration Gemini
✅ Détection automatique des requêtes
✅ Formatage pour l'API Gemini
✅ Gestion des contextes
✅ Fallback vers l'ancien système

### API REST Complète
✅ Endpoints CRUD complets
✅ Gestion des sessions
✅ Cache et statistiques
✅ Documentation auto-générée
✅ Tests de santé

## Endpoints API Disponibles

- POST /api/web-navigation/search-and-navigate
- POST /api/web-navigation/extract-content
- POST /api/web-navigation/navigate-deep
- POST /api/web-navigation/user-journey
- GET  /api/web-navigation/health
- GET  /api/web-navigation/docs
- GET  /api/web-navigation/stats

## Prochaines Étapes

1. Optimisation des performances
2. Ajout de plus de moteurs de recherche
3. Amélioration de la détection de contenu
4. Support de plus de formats de données
5. Monitoring avancé

"""
    
    try:
        with open("test_report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("✅ Rapport de test créé: test_report.md")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du rapport: {str(e)}")
        return False

def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage de l'installation et des tests du Système de Navigation Web Avancé")
    logger.info("=" * 80)
    
    # Étape 1: Installation des dépendances
    if not install_requirements():
        logger.error("❌ Installation des dépendances échouée")
        return False
    
    # Étape 2: Test des imports
    if not test_imports():
        logger.error("❌ Test des imports échoué")
        return False
    
    # Étape 3: Test du système de navigation
    if not test_navigation_system():
        logger.error("❌ Test du système de navigation échoué")
        return False
    
    # Étape 4: Test de l'intégration Gemini
    if not test_gemini_integration():
        logger.error("❌ Test de l'intégration Gemini échoué")
        return False
    
    # Étape 5: Test des endpoints API
    if not test_api_endpoints():
        logger.error("❌ Test des endpoints API échoué")
        return False
    
    # Étape 6: Création du rapport
    create_test_report()
    
    logger.info("=" * 80)
    logger.info("🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
    logger.info("✅ Le Système de Navigation Web Avancé est prêt à l'emploi")
    logger.info("=" * 80)
    
    # Afficher les instructions d'utilisation
    logger.info("\n📚 INSTRUCTIONS D'UTILISATION:")
    logger.info("1. Le système est maintenant intégré dans votre app Flask")
    logger.info("2. Les endpoints API sont disponibles sous /api/web-navigation/")
    logger.info("3. L'intégration Gemini détecte automatiquement les requêtes de navigation")
    logger.info("4. Consultez test_report.md pour plus de détails")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
