"""
Intégration Navigation Web + Vision Gemini
Permet à Gemini de naviguer ET voir visuellement l'intérieur des sites web
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Import des systèmes existants
try:
    from advanced_web_navigator import AdvancedWebNavigator
    from gemini_visual_adapter import GeminiVisualAdapter, initialize_gemini_visual_adapter
    from intelligent_web_capture import IntelligentWebCapture, initialize_intelligent_capture
    NAVIGATION_AVAILABLE = True
except ImportError as e:
    NAVIGATION_AVAILABLE = False
    logger.error(f"❌ Modules de navigation non disponibles: {e}")

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GeminiWebVision')

class GeminiWebVisionIntegration:
    """Intégration complète Navigation Web + Vision Gemini"""
    
    def __init__(self, api_key: str = None):
        """
        Initialise l'intégration Navigation + Vision
        
        Args:
            api_key: Clé API Gemini
        """
        self.api_key = api_key
        
        # Initialiser les composants
        self.visual_adapter = initialize_gemini_visual_adapter(api_key)
        self.capture_system = initialize_intelligent_capture()
        
        # Initialiser le navigateur web si disponible
        if NAVIGATION_AVAILABLE:
            self.web_navigator = AdvancedWebNavigator()
            logger.info("✅ Navigateur web avancé intégré")
        else:
            self.web_navigator = None
            logger.warning("⚠️ Navigateur web non disponible")
        
        # Configuration
        self.config = {
            'auto_capture': True,  # Capturer automatiquement pendant la navigation
            'capture_types': ['visible_area', 'full_page'],
            'analyze_during_navigation': True,
            'save_analysis': True,
            'max_captures_per_site': 5
        }
        
        # Répertoires
        self.data_dir = Path("data/gemini_web_vision")
        self.reports_dir = self.data_dir / "reports"
        self.navigation_logs_dir = self.data_dir / "navigation_logs"
        
        for dir_path in [self.data_dir, self.reports_dir, self.navigation_logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Sessions actives
        self.active_sessions = {}
        
        # Statistiques
        self.stats = {
            'sessions_created': 0,
            'sites_navigated': 0,
            'captures_taken': 0,
            'analyses_performed': 0,
            'total_processing_time': 0
        }
        
        logger.info("🚀 Intégration Gemini Web + Vision initialisée")
    
    def create_vision_navigation_session(self, 
                                       session_id: str,
                                       user_query: str,
                                       navigation_goals: List[str] = None) -> Dict[str, Any]:
        """
        Crée une session de navigation avec vision intégrée
        
        Args:
            session_id: Identifiant unique de la session
            user_query: Requête utilisateur
            navigation_goals: Objectifs de navigation spécifiques
            
        Returns:
            Information sur la session créée
        """
        try:
            # Validation des paramètres
            if not session_id or not isinstance(session_id, str) or len(session_id.strip()) == 0:
                return {
                    'success': False,
                    'error': 'session_id invalide: doit être une chaîne non vide'
                }
            
            if not user_query or not isinstance(user_query, str) or len(user_query.strip()) == 0:
                return {
                    'success': False,
                    'error': 'user_query invalide: doit être une chaîne non vide'
                }
            
            if navigation_goals is None:
                navigation_goals = ['extract_content', 'analyze_ui', 'capture_visuals']
            
            session_info = {
                'session_id': session_id,
                'user_query': user_query,
                'navigation_goals': navigation_goals,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'sites_visited': [],
                'captures_taken': [],
                'analyses_performed': [],
                'total_content_extracted': 0
            }
            
            self.active_sessions[session_id] = session_info
            self.stats['sessions_created'] += 1
            
            logger.info(f"🆕 Session vision-navigation créée: {session_id}")
            return {
                'success': True,
                'session_id': session_id,
                'session_info': session_info
            }
            
        except Exception as e:
            error_msg = f"Erreur création session {session_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def navigate_with_vision(self, 
                           session_id: str,
                           url: str,
                           navigation_type: str = "smart_exploration",
                           capture_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Navigue sur un site avec capture et analyse visuelle automatique
        
        Args:
            session_id: ID de la session
            url: URL à visiter
            navigation_type: Type de navigation (smart_exploration, content_focus, ui_analysis)
            capture_config: Configuration de capture personnalisée
            
        Returns:
            Résultats de la navigation avec analyses visuelles
        """
        start_time = datetime.now()
        
        try:
            if session_id not in self.active_sessions:
                return {
                    'success': False,
                    'error': f'Session {session_id} non trouvée'
                }
            
            session = self.active_sessions[session_id]
            
            # Configuration par défaut de capture
            if capture_config is None:
                capture_config = {
                    'capture_type': 'full_page',
                    'viewport': 'desktop',
                    'analyze_elements': True
                }
            
            logger.info(f"🌐 Navigation avec vision: {url} (session: {session_id})")
            
            # 1. Capturer le site avant navigation
            initial_capture = self.capture_system.capture_website_intelligent(
                url=url,
                **capture_config
            )
            
            if not initial_capture['success']:
                return {
                    'success': False,
                    'error': f'Échec capture initiale: {initial_capture.get("error")}'
                }
            
            # 2. Analyser visuellement les captures
            visual_analyses = []
            for capture in initial_capture['captures']:
                if 'optimized_path' in capture:
                    analysis_prompt = self._generate_analysis_prompt(navigation_type, session['user_query'])
                    
                    analysis_result = self.visual_adapter.analyze_website_screenshot(
                        image_path=capture['optimized_path'],
                        analysis_prompt=analysis_prompt,
                        context=f"Navigation {navigation_type} pour: {session['user_query']}"
                    )
                    
                    if analysis_result['success']:
                        visual_analyses.append({
                            'capture_info': capture,
                            'analysis': analysis_result['analysis'],
                            'processing_time': analysis_result['processing_time']
                        })
                        
                        logger.info(f"✅ Analyse visuelle réussie pour section {capture.get('section', 1)}")
            
            # 3. Navigation basée sur l'analyse visuelle (si navigateur disponible)
            navigation_result = None
            if self.web_navigator and navigation_type != "visual_only":
                # Utiliser les analyses visuelles pour guider la navigation
                navigation_guidance = self._generate_navigation_guidance(visual_analyses)
                
                if hasattr(self.web_navigator, 'navigate_with_guidance'):
                    navigation_result = self.web_navigator.navigate_with_guidance(
                        url=url,
                        guidance=navigation_guidance,
                        session_id=session_id
                    )
            
            # 4. Mettre à jour la session
            session['sites_visited'].append({
                'url': url,
                'timestamp': start_time.isoformat(),
                'navigation_type': navigation_type,
                'captures_count': len(initial_capture['captures']),
                'analyses_count': len(visual_analyses)
            })
            
            session['captures_taken'].extend(initial_capture['captures'])
            session['analyses_performed'].extend(visual_analyses)
            
            # 5. Calculer les métriques
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Mettre à jour les statistiques globales
            self.stats['sites_navigated'] += 1
            self.stats['captures_taken'] += len(initial_capture['captures'])
            self.stats['analyses_performed'] += len(visual_analyses)
            self.stats['total_processing_time'] += processing_time
            
            # 6. Sauvegarder le rapport de session
            self._save_session_report(session_id, {
                'url': url,
                'navigation_type': navigation_type,
                'captures': initial_capture['captures'],
                'visual_analyses': visual_analyses,
                'navigation_result': navigation_result,
                'processing_time': processing_time
            })
            
            logger.info(f"✅ Navigation avec vision terminée: {url} en {processing_time:.2f}s")
            
            return {
                'success': True,
                'session_id': session_id,
                'url': url,
                'navigation_type': navigation_type,
                'captures': initial_capture['captures'],
                'visual_analyses': visual_analyses,
                'navigation_result': navigation_result,
                'processing_time': processing_time,
                'stats': {
                    'captures_taken': len(initial_capture['captures']),
                    'analyses_performed': len(visual_analyses),
                    'total_content_length': sum(len(a.get('analysis', '')) for a in visual_analyses)
                }
            }
            
        except Exception as e:
            error_msg = f"Erreur navigation avec vision {url}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'session_id': session_id
            }
    
    def _generate_analysis_prompt(self, navigation_type: str, user_query: str) -> str:
        """Génère un prompt d'analyse adapté au type de navigation"""
        
        base_prompts = {
            'smart_exploration': f"""
🔍 **EXPLORATION INTELLIGENTE DU SITE**

**Contexte utilisateur**: {user_query}

**Analysez cette capture en tant qu'explorateur intelligent**:
1. 🏗️ **Architecture**: Structure générale, organisation du contenu
2. 🎯 **Points d'intérêt**: Éléments qui répondent à la requête utilisateur
3. 🧭 **Navigation**: Menus, liens importants, chemins de navigation
4. 📄 **Contenu clé**: Informations principales visibles
5. 🔗 **Prochaines étapes** : Où naviguer ensuite pour répondre à la requête
""",
            
            'content_focus': f"""
📖 **ANALYSE FOCALISÉE SUR LE CONTENU**

**Recherche pour**: {user_query}

**Concentrez-vous sur**:
1. 📝 **Contenu textuel**: Titre, paragraphes, informations pertinentes
2. 📊 **Données structurées**: Listes, tableaux, statistiques
3. 🖼️ **Médias informatifs**: Images, graphiques avec du contenu
4. 🔍 **Pertinence**: Lien avec la requête utilisateur
5. 📋 **Extraction**: Résumé du contenu le plus important
""",
            
            'ui_analysis': f"""
🎨 **ANALYSE UX/UI DÉTAILLÉE**

**Dans le contexte de**: {user_query}

**Évaluez l'interface**:
1. 🖥️ **Design**: Cohérence visuelle, hiérarchie, lisibilité
2. 🎛️ **Utilisabilité**: Facilité de navigation, accessibilité
3. 📱 **Responsive**: Adaptation à différents écrans
4. ⚡ **Performance visuelle**: Temps de chargement apparent
5. 🏆 **Qualité globale**: Note et recommandations d'amélioration
""",
            
            'visual_only': f"""
👁️ **ANALYSE VISUELLE PURE**

**Contexte**: {user_query}

**Décrivez ce que vous voyez**:
1. 🖼️ **Éléments visuels**: Couleurs, formes, mise en page
2. 📐 **Composition**: Équilibre, alignement, espacement
3. 🎭 **Ambiance**: Impression générale, ton du site
4. 🔍 **Détails importants**: Éléments qui attirent l'attention
5. 💭 **Interprétation**: Ce que le site communique visuellement
"""
        }
        
        return base_prompts.get(navigation_type, base_prompts['smart_exploration'])
    
    def _generate_navigation_guidance(self, visual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Génère des conseils de navigation basés sur les analyses visuelles"""
        try:
            # Extraire les éléments d'intérêt des analyses
            navigation_elements = []
            content_areas = []
            ui_insights = []
            
            for analysis in visual_analyses:
                analysis_text = analysis.get('analysis', '')
                
                # Chercher des mentions de navigation
                if any(keyword in analysis_text.lower() for keyword in ['menu', 'navigation', 'lien', 'bouton']):
                    navigation_elements.append(analysis_text[:200])
                
                # Chercher du contenu intéressant
                if any(keyword in analysis_text.lower() for keyword in ['contenu', 'information', 'article', 'données']):
                    content_areas.append(analysis_text[:200])
                
                # Insights UI
                if any(keyword in analysis_text.lower() for keyword in ['design', 'interface', 'utilisabilité']):
                    ui_insights.append(analysis_text[:200])
            
            return {
                'navigation_elements': navigation_elements,
                'content_areas': content_areas,
                'ui_insights': ui_insights,
                'guidance_generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération guidance: {e}")
            return {}
    
    def _save_session_report(self, session_id: str, navigation_data: Dict[str, Any]):
        """Sauvegarde un rapport détaillé de la session"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"vision_navigation_{session_id}_{timestamp}.json"
            report_path = self.reports_dir / report_filename
            
            # Préparer le rapport complet
            report = {
                'session_id': session_id,
                'session_info': self.active_sessions.get(session_id, {}),
                'navigation_data': navigation_data,
                'system_stats': self.get_statistics(),
                'generated_at': datetime.now().isoformat()
            }
            
            # Sauvegarder
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Rapport de session sauvegardé: {report_filename}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde rapport: {e}")
    
    def analyze_site_comparison(self, 
                              session_id: str,
                              url1: str, 
                              url2: str,
                              comparison_focus: str = "general") -> Dict[str, Any]:
        """
        Compare visuellement deux sites web
        
        Args:
            session_id: ID de session
            url1: Premier site
            url2: Deuxième site  
            comparison_focus: Focus de comparaison (general, ui, content, performance)
            
        Returns:
            Résultat de la comparaison
        """
        try:
            logger.info(f"🔍 Comparaison visuelle: {url1} vs {url2}")
            
            # Capturer les deux sites
            capture1 = self.capture_system.capture_website_intelligent(url1, capture_type="visible_area")
            capture2 = self.capture_system.capture_website_intelligent(url2, capture_type="visible_area")
            
            if not capture1['success'] or not capture2['success']:
                return {
                    'success': False,
                    'error': 'Échec capture d\'un ou plusieurs sites'
                }
            
            # Obtenir les chemins des images optimisées
            image1_path = capture1['captures'][0]['optimized_path']
            image2_path = capture2['captures'][0]['optimized_path']
            
            # Effectuer la comparaison visuelle
            comparison_context = f"Comparaison {comparison_focus} entre {url1} et {url2}"
            
            comparison_result = self.visual_adapter.compare_website_changes(
                image_path_before=image1_path,
                image_path_after=image2_path,
                comparison_context=comparison_context
            )
            
            if comparison_result['success']:
                logger.info("✅ Comparaison visuelle réussie")
                
                return {
                    'success': True,
                    'session_id': session_id,
                    'url1': url1,
                    'url2': url2,
                    'comparison_focus': comparison_focus,
                    'comparison_analysis': comparison_result['comparison'],
                    'captures': {
                        'site1': capture1['captures'][0],
                        'site2': capture2['captures'][0]
                    }
                }
            else:
                return {
                    'success': False,
                    'error': comparison_result.get('error', 'Erreur comparaison')
                }
                
        except Exception as e:
            error_msg = f"Erreur comparaison sites: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """
        Ferme une session et génère le rapport final
        
        Args:
            session_id: ID de la session à fermer
            
        Returns:
            Rapport final de la session
        """
        try:
            if session_id not in self.active_sessions:
                return {
                    'success': False,
                    'error': f'Session {session_id} non trouvée'
                }
            
            session = self.active_sessions[session_id]
            session['status'] = 'closed'
            session['closed_at'] = datetime.now().isoformat()
            
            # Calculer les statistiques de session
            session_stats = {
                'sites_visited': len(session['sites_visited']),
                'total_captures': len(session['captures_taken']),
                'total_analyses': len(session['analyses_performed']),
                'session_duration': self._calculate_session_duration(session)
            }
            
            session['final_stats'] = session_stats
            
            # Sauvegarder le rapport final
            self._save_session_report(session_id, {
                'type': 'final_report',
                'session_summary': session,
                'final_stats': session_stats
            })
            
            # Supprimer de la liste des sessions actives
            del self.active_sessions[session_id]
            
            logger.info(f"🏁 Session fermée: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'final_stats': session_stats,
                'session_summary': session
            }
            
        except Exception as e:
            error_msg = f"Erreur fermeture session {session_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def _calculate_session_duration(self, session: Dict[str, Any]) -> float:
        """Calcule la durée d'une session en secondes"""
        try:
            start_time = datetime.fromisoformat(session['created_at'])
            end_time = datetime.fromisoformat(session.get('closed_at', datetime.now().isoformat()))
            return (end_time - start_time).total_seconds()
        except:
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques globales du système"""
        avg_time = self.stats['total_processing_time'] / max(self.stats['sites_navigated'], 1)
        
        return {
            'sessions_created': self.stats['sessions_created'],
            'active_sessions': len(self.active_sessions),
            'sites_navigated': self.stats['sites_navigated'],
            'captures_taken': self.stats['captures_taken'],
            'analyses_performed': self.stats['analyses_performed'],
            'average_processing_time': round(avg_time, 2),
            'total_processing_time': round(self.stats['total_processing_time'], 2),
            'components_status': {
                'visual_adapter': self.visual_adapter is not None,
                'capture_system': self.capture_system is not None,
                'web_navigator': self.web_navigator is not None
            }
        }
    
    def cleanup(self):
        """Nettoie les ressources du système"""
        try:
            # Fermer toutes les sessions actives
            for session_id in list(self.active_sessions.keys()):
                self.close_session(session_id)
            
            # Nettoyer le système de capture
            if self.capture_system:
                self.capture_system.close()
            
            logger.info("🧹 Nettoyage du système terminé")
            
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {e}")

# Instance globale
gemini_web_vision = None

def initialize_gemini_web_vision(api_key: str = None) -> GeminiWebVisionIntegration:
    """
    Initialise l'intégration globale Gemini Web + Vision
    
    Args:
        api_key: Clé API Gemini
        
    Returns:
        Instance de l'intégration
    """
    global gemini_web_vision
    
    if gemini_web_vision is None:
        gemini_web_vision = GeminiWebVisionIntegration(api_key)
        logger.info("🌟 Intégration Gemini Web + Vision initialisée globalement")
    
    return gemini_web_vision

def get_gemini_web_vision() -> Optional[GeminiWebVisionIntegration]:
    """
    Retourne l'instance globale de l'intégration
    
    Returns:
        Instance ou None si non initialisé
    """
    global gemini_web_vision
    return gemini_web_vision

if __name__ == "__main__":
    # Test de l'intégration
    integration = initialize_gemini_web_vision()
    print("🧪 Intégration Gemini Web + Vision prête pour les tests")
    print(f"📊 Statistiques: {integration.get_statistics()}")
