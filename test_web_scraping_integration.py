
#!/usr/bin/env python3
"""
Test d'intégration du système de web scraping autonome
"""

def test_web_scraping_integration():
    """Test complet de l'intégration du web scraping"""
    print("🧪 Test d'intégration du système de web scraping autonome")
    
    try:
        # Test 1: Import des modules
        from autonomous_web_scraper import start_autonomous_web_learning, get_autonomous_learning_status
        from web_learning_integration import trigger_autonomous_learning, force_web_learning_session
        print("✅ Modules de web scraping importés avec succès")
        
        # Test 2: Vérifier le statut du système
        status = get_autonomous_learning_status()
        print(f"✅ Statut du système: {status.get('autonomous_learning_active', False)}")
        
        # Test 3: Test de session d'apprentissage (courte)
        print("🔍 Déclenchement d'une session de test...")
        result = force_web_learning_session()
        
        if result.get("forced") and result.get("session_result", {}).get("success"):
            session = result["session_result"]
            print(f"✅ Session réussie:")
            print(f"   - Pages traitées: {session.get('pages_processed', 0)}")
            print(f"   - Domaine: {session.get('domain_focus', 'Non spécifié')}")
            print(f"   - Fichiers créés: {len(session.get('files_created', []))}")
            return True
        else:
            print("⚠️ Session de test non concluante")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_web_scraping_integration()
    if success:
        print("\n🎉 Le système de web scraping autonome est opérationnel !")
        print("L'IA peut maintenant effectuer des recherches web autonomes.")
    else:
        print("\n⚠️ Des problèmes ont été détectés avec le système de web scraping.")
