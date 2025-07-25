"""
artificial intelligence Google Gemini 2.0 Flash AI Web + Vision Integration
Allows Google Gemini 2.0 Flash AI to navigate AND visually see the inside of websites
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Import existing systems
try:
    from advanced_web_navigator import AdvancedWebNavigator
    from gemini_visual_adapter import GeminiVisualAdapter, initialize_gemini_visual_adapter
    from intelligent_web_capture import IntelligentWebCapture, initialize_intelligent_capture
    NAVIGATION_AVAILABLE = True
except ImportError as e:
    NAVIGATION_AVAILABLE = False
    logger.error(f"❌ Navigation modules not available: {e}")

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Google Gemini 2.0 Flash AI WebVision')

class GeminiWebVisionIntegration:
    """Complete Web Navigation + Google Gemini 2.0 Flash AI Vision Integration"""
    
    def __init__(self, api_key: str = None):
        """
        Initializes the Navigation + Vision integration
        
        Args:
            api_key: Google Gemini 2.0 Flash AI API Key
        """
        self.api_key = api_key
        
        # Initialize components
        self.visual_adapter = initialize_gemini_visual_adapter(api_key)
        self.capture_system = initialize_intelligent_capture()
        
        # Initialize web navigator if available
        if NAVIGATION_AVAILABLE:
            self.web_navigator = AdvancedWebNavigator()
            logger.info("✅ Advanced web navigator integrated")
        else:
            self.web_navigator = None
            logger.warning("⚠️ Web navigator not available")
        
        # Configuration
        self.config = {
            'auto_capture': True,  # Automatically capture during navigation
            'capture_types': ['visible_area', 'full_page'],
            'analyze_during_navigation': True,
            'save_analysis': True,
            'max_captures_per_site': 5
        }
        
        # Directories
        self.data_dir = Path("data/gemini_web_vision")
        self.reports_dir = self.data_dir / "reports"
        self.navigation_logs_dir = self.data_dir / "navigation_logs"
        
        for dir_path in [self.data_dir, self.reports_dir, self.navigation_logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Active sessions
        self.active_sessions = {}
        
        # Statistics
        self.stats = {
            'sessions_created': 0,
            'sites_navigated': 0,
            'captures_taken': 0,
            'analyses_performed': 0,
            'total_processing_time': 0
        }
        
        logger.info("🚀 Google Gemini 2.0 Flash AI Web + Vision Integration initialized")
    
    def create_vision_navigation_session(self, 
                                       session_id: str,
                                       user_query: str,
                                       navigation_goals: List[str] = None) -> Dict[str, Any]:
        """
        Creates a navigation session with integrated vision
        
        Args:
            session_id: Unique session identifier
            user_query: User query
            navigation_goals: Specific navigation goals
            
        Returns:
            Information about the created session
        """
        try:
            # Parameter validation
            if not session_id or not isinstance(session_id, str) or len(session_id.strip()) == 0:
                return {
                    'success': False,
                    'error': 'invalid session_id: must be a non-empty string'
                }
            
            if not user_query or not isinstance(user_query, str) or len(user_query.strip()) == 0:
                return {
                    'success': False,
                    'error': 'invalid user_query: must be a non-empty string'
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
            
            logger.info(f"🆕 Vision-navigation session created: {session_id}")
            return {
                'success': True,
                'session_id': session_id,
                'session_info': session_info
            }
            
        except Exception as e:
            error_msg = f"Error creating session {session_id}: {str(e)}"
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
        Navigates a site with automatic visual capture and analysis
        
        Args:
            session_id: Session ID
            url: URL to visit
            navigation_type: Navigation type (smart_exploration, content_focus, ui_analysis)
            capture_config: Custom capture configuration
            
        Returns:
            Navigation results with visual analyses
        """
        start_time = datetime.now()
        
        try:
            if session_id not in self.active_sessions:
                return {
                    'success': False,
                    'error': f'Session {session_id} not found'
                }
            
            session = self.active_sessions[session_id]
            
            # Default capture configuration
            if capture_config is None:
                capture_config = {
                    'capture_type': 'full_page',
                    'viewport': 'desktop',
                    'analyze_elements': True
                }
            
            logger.info(f"🌐 Navigation with vision: {url} (session: {session_id})")
            
            # 1. Capture the site before navigation
            initial_capture = self.capture_system.capture_website_intelligent(
                url=url,
                **capture_config
            )
            
            if not initial_capture['success']:
                return {
                    'success': False,
                    'error': f'Initial capture failed: {initial_capture.get("error")}'
                }
            
            # 2. Visually analyze the captures
            visual_analyses = []
            for capture in initial_capture['captures']:
                if 'optimized_path' in capture:
                    analysis_prompt = self._generate_analysis_prompt(navigation_type, session['user_query'])
                    
                    analysis_result = self.visual_adapter.analyze_website_screenshot(
                        image_path=capture['optimized_path'],
                        analysis_prompt=analysis_prompt,
                        context=f"Navigation {navigation_type} for: {session['user_query']}"
                    )
                    
                    if analysis_result['success']:
                        visual_analyses.append({
                            'capture_info': capture,
                            'analysis': analysis_result['analysis'],
                            'processing_time': analysis_result['processing_time']
                        })
                        
                        logger.info(f"✅ Visual analysis successful for section {capture.get('section', 1)}")
            
            # 3. Navigation based on visual analysis (if navigator available)
            navigation_result = None
            if self.web_navigator and navigation_type != "visual_only":
                # Use visual analyses to guide navigation
                navigation_guidance = self._generate_navigation_guidance(visual_analyses)
                
                if hasattr(self.web_navigator, 'navigate_with_guidance'):
                    navigation_result = self.web_navigator.navigate_with_guidance(
                        url=url,
                        guidance=navigation_guidance,
                        session_id=session_id
                    )
            
            # 4. Update the session
            session['sites_visited'].append({
                'url': url,
                'timestamp': start_time.isoformat(),
                'navigation_type': navigation_type,
                'captures_count': len(initial_capture['captures']),
                'analyses_count': len(visual_analyses)
            })
            
            session['captures_taken'].extend(initial_capture['captures'])
            session['analyses_performed'].extend(visual_analyses)
            
            # 5. Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update global statistics
            self.stats['sites_navigated'] += 1
            self.stats['captures_taken'] += len(initial_capture['captures'])
            self.stats['analyses_performed'] += len(visual_analyses)
            self.stats['total_processing_time'] += processing_time
            
            # 6. Save the session report
            self._save_session_report(session_id, {
                'url': url,
                'navigation_type': navigation_type,
                'captures': initial_capture['captures'],
                'visual_analyses': visual_analyses,
                'navigation_result': navigation_result,
                'processing_time': processing_time
            })
            
            logger.info(f"✅ Navigation with vision completed: {url} in {processing_time:.2f}s")
            
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
            error_msg = f"Navigation with vision error {url}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'session_id': session_id
            }
    
    def _generate_analysis_prompt(self, navigation_type: str, user_query: str) -> str:
        """Generates an analysis prompt adapted to the navigation type"""
        
        base_prompts = {
            'smart_exploration': f"""
🔍 **INTELLIGENT SITE EXPLORATION**

**User context**: {user_query}

**Analyze this capture as an intelligent explorer**:
1. 🏗️ **Architecture**: General structure, content organization
2. 🎯 **Points of interest**: Elements that respond to the user query
3. 🧭 **Navigation**: Menus, important links, navigation paths
4. 📄 **Key content**: Main visible information
5. 🔗 **Next steps**: Where to navigate next to answer the query
""",
            
            'content_focus': f"""
📖 **CONTENT-FOCUSED ANALYSIS**

**Searching for**: {user_query}

**Focus on**:
1. 📝 **Textual content**: Title, paragraphs, relevant information
2. 📊 **Structured data**: Lists, tables, statistics
3. 🖼️ **Informative media**: Images, graphics with content
4. 🔍 **Relevance**: Link with the user query
5. 📋 **Extraction**: Summary of the most important content
""",
            
            'ui_analysis': f"""
🎨 **DETAILED UX/UI ANALYSIS**

**In the context of**: {user_query}

**Evaluate the interface**:
1. 🖥️ **Design**: Visual consistency, hierarchy, readability
2. 🎛️ **Usability**: Ease of navigation, accessibility
3. 📱 **Responsive**: Adaptation to different screens
4. ⚡ **Visual performance**: Apparent loading time
5. 🏆 **Overall quality**: Rating and improvement recommendations
""",
            
            'visual_only': f"""
👁️ **PURE VISUAL ANALYSIS**

**Context**: {user_query}

**Describe what you see**:
1. 🖼️ **Visual elements**: Colors, shapes, layout
2. 📐 **Composition**: Balance, alignment, spacing
3. 🎭 **Atmosphere**: General impression, site tone
4. 🔍 **Important details**: Elements that draw attention
5. 💭 **Interpretation**: What the site visually communicates
"""
        }
        
        return base_prompts.get(navigation_type, base_prompts['smart_exploration'])
    
    def _generate_navigation_guidance(self, visual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates navigation advice based on visual analyses"""
        try:
            # Extract elements of interest from analyses
            navigation_elements = []
            content_areas = []
            ui_insights = []
            
            for analysis in visual_analyses:
                analysis_text = analysis.get('analysis', '')
                
                # Look for navigation mentions
                if any(keyword in analysis_text.lower() for keyword in ['menu', 'navigation', 'link', 'button']):
                    navigation_elements.append(analysis_text[:200])
                
                # Look for interesting content
                if any(keyword in analysis_text.lower() for keyword in ['content', 'information', 'article', 'data']):
                    content_areas.append(analysis_text[:200])
                
                # UI Insights
                if any(keyword in analysis_text.lower() for keyword in ['design', 'interface', 'usability']):
                    ui_insights.append(analysis_text[:200])
            
            return {
                'navigation_elements': navigation_elements,
                'content_areas': content_areas,
                'ui_insights': ui_insights,
                'guidance_generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating guidance: {e}")
            return {}
    
    def _save_session_report(self, session_id: str, navigation_data: Dict[str, Any]):
        """Saves a detailed session report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"vision_navigation_{session_id}_{timestamp}.json"
            report_path = self.reports_dir / report_filename
            
            # Prepare the complete report
            report = {
                'session_id': session_id,
                'session_info': self.active_sessions.get(session_id, {}),
                'navigation_data': navigation_data,
                'system_stats': self.get_statistics(),
                'generated_at': datetime.now().isoformat()
            }
            
            # Save
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Session report saved: {report_filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving report: {e}")
    
    def analyze_site_comparison(self, 
                              session_id: str,
                              url1: str, 
                              url2: str,
                              comparison_focus: str = "general") -> Dict[str, Any]:
        """
        Visually compares two websites
        
        Args:
            session_id: Session ID
            url1: First site
            url2: Second site  
            comparison_focus: Comparison focus (general, ui, content, performance)
            
        Returns:
            Comparison result
        """
        try:
            logger.info(f"🔍 Visual comparison: {url1} vs {url2}")
            
            # Capture both sites
            capture1 = self.capture_system.capture_website_intelligent(url1, capture_type="visible_area")
            capture2 = self.capture_system.capture_website_intelligent(url2, capture_type="visible_area")
            
            if not capture1['success'] or not capture2['success']:
                return {
                    'success': False,
                    'error': 'Failed to capture one or more sites'
                }
            
            # Get paths of optimized images
            image1_path = capture1['captures'][0]['optimized_path']
            image2_path = capture2['captures'][0]['optimized_path']
            
            # Perform visual comparison
            comparison_context = f"Comparison {comparison_focus} between {url1} and {url2}"
            
            comparison_result = self.visual_adapter.compare_website_changes(
                image_path_before=image1_path,
                image_path_after=image2_path,
                comparison_context=comparison_context
            )
            
            if comparison_result['success']:
                logger.info("✅ Visual comparison successful")
                
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
                    'error': comparison_result.get('error', 'Comparison error')
                }
                
        except Exception as e:
            error_msg = f"Site comparison error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """
        Closes a session and generates the final report
        
        Args:
            session_id: ID of the session to close
            
        Returns:
            Final session report
        """
        try:
            if session_id not in self.active_sessions:
                return {
                    'success': False,
                    'error': f'Session {session_id} not found'
                }
            
            session = self.active_sessions[session_id]
            session['status'] = 'closed'
            session['closed_at'] = datetime.now().isoformat()
            
            # Calculate session statistics
            session_stats = {
                'sites_visited': len(session['sites_visited']),
                'total_captures': len(session['captures_taken']),
                'total_analyses': len(session['analyses_performed']),
                'session_duration': self._calculate_session_duration(session)
            }
            
            session['final_stats'] = session_stats
            
            # Save the final report
            self._save_session_report(session_id, {
                'type': 'final_report',
                'session_summary': session,
                'final_stats': session_stats
            })
            
            # Remove from active sessions list
            del self.active_sessions[session_id]
            
            logger.info(f"🏁 Session closed: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'final_stats': session_stats,
                'session_summary': session
            }
            
        except Exception as e:
            error_msg = f"Error closing session {session_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def _calculate_session_duration(self, session: Dict[str, Any]) -> float:
        """Calculates the duration of a session in seconds"""
        try:
            start_time = datetime.fromisoformat(session['created_at'])
            end_time = datetime.fromisoformat(session.get('closed_at', datetime.now().isoformat()))
            return (end_time - start_time).total_seconds()
        except:
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns global system statistics"""
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
        """Cleans up system resources"""
        try:
            # Close all active sessions
            for session_id in list(self.active_sessions.keys()):
                self.close_session(session_id)
            
            # Clean up the capture system
            if self.capture_system:
                self.capture_system.close()
            
            logger.info("🧹 System cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Global instance
gemini_web_vision = None

def initialize_gemini_web_vision(api_key: str = None) -> GeminiWebVisionIntegration:
    """
    Initializes the global Google Gemini 2.0 Flash AI Web + Vision integration
    
    Args:
        api_key: Google Gemini 2.0 Flash AI API Key
        
    Returns:
        Integration instance
    """
    global gemini_web_vision
    
    if gemini_web_vision is None:
        gemini_web_vision = GeminiWebVisionIntegration(api_key)
        logger.info("🌟 Google Gemini 2.0 Flash AI Web + Vision Integration initialized globally")
    
    return gemini_web_vision

def get_gemini_web_vision() -> Optional[GeminiWebVisionIntegration]:
    """
    Returns the global integration instance
    
    Returns:
        Instance or None if not initialized
    """
    global gemini_web_vision
    return gemini_web_vision

if __name__ == "__main__":
    # Integration test
    integration = initialize_gemini_web_vision()
    print("🧪 Google Gemini 2.0 Flash AI Web + Vision Integration ready for tests")
    print(f"📊 Statistics: {integration.get_statistics()}")
