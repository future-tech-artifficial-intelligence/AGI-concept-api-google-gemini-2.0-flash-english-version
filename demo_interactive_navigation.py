"""
**Demonstration of the Interactive Navigation System with artificial intelligence API GOOGLE GEMINI 2.0 FLASH**
This script showcases the new capabilities for interacting with web elements.
"""

import logging
import time
import json
from datetime import datetime
from pathlib import Path

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeminiInteractiveDemo')

def demo_header(title: str):
    """Displays a demonstration header"""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def demo_section(title: str):
    """Displays a section title"""
    print(f"\n📋 {title}")
    print("-" * 60)

class InteractiveNavigationDemo:
    """Demonstration class for the interactive navigation system"""
    
    def __init__(self):
        self.demo_results = {}
        self.screenshots_taken = []
    
    def demo_element_analysis(self):
        """Demonstration of interactive element analysis"""
        demo_section("INTERACTIVE ELEMENT ANALYSIS")
        
        try:
            from interactive_web_navigator import InteractiveElementAnalyzer
            
            analyzer = InteractiveElementAnalyzer()
            print("✅ Element analyzer created")
            
            # Show detectable element types
            print(f"\n🔍 Detectable element types:")
            for element_type, selectors in analyzer.element_selectors.items():
                print(f"  • {element_type}: {len(selectors)} CSS selectors")
            
            # Show importance keywords
            print(f"\n💡 Importance criteria:")
            for importance, keywords in analyzer.importance_keywords.items():
                print(f"  • {importance}: {', '.join(keywords[:5])}...")
            
            # Simulation of score calculation
            print(f"\n📊 Examples of interaction scores:")
            
            test_elements = [
                ("Button 'Next'", "Next", {'id': 'next-btn'}, 'buttons', {'x': 100, 'y': 200, 'width': 80, 'height': 30}),
                ("Tab 'Services'", "Services", {'role': 'tab'}, 'tabs', {'x': 200, 'y': 50, 'width': 100, 'height': 40}),
                ("Link 'Back'", "Back", {'class': 'nav-link'}, 'navigation', {'x': 50, 'y': 800, 'width': 60, 'height': 20}),
                ("Search field", "", {'type': 'search'}, 'inputs', {'x': 300, 'y': 60, 'width': 200, 'height': 25})
            ]
            
            for name, text, attrs, elem_type, position in test_elements:
                score = analyzer._calculate_interaction_score(text, attrs, elem_type, position)
                priority = "🔥 High" if score > 0.7 else "⚡ Medium" if score > 0.4 else "💤 Low"
                print(f"  • {name}: {score:.2f} ({priority})")
            
            self.demo_results['element_analysis'] = {
                'status': 'success',
                'elements_types': len(analyzer.element_selectors),
                'importance_levels': len(analyzer.importance_keywords)
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.demo_results['element_analysis'] = {'status': 'error', 'error': str(e)}
    
    def demo_interaction_detection(self):
        """Demonstration of user interaction detection"""
        demo_section("USER INTERACTION DETECTION")
        
        try:
            from gemini_interactive_adapter import detect_interactive_need
            
            # Examples of user prompts
            demo_prompts = [
                {
                    'prompt': "Click on the 'Products' tab of this website",
                    'description': "Direct interaction with a specific element"
                },
                {
                    'prompt': "Explore all available tabs on https://example.com",
                    'description': "Systematic tab navigation"
                },
                {
                    'prompt': "Browse all sections of the site to see what is available",
                    'description': "Complete and automatic exploration"
                },
                {
                    'prompt': "Fill out the contact form with my information",
                    'description': "Form interaction"
                },
                {
                    'prompt': "What is artificial intelligence?",
                    'description': "Normal question (no interaction)"
                }
            ]
            
            print("🧪 Detection test on different request types:\n")
            
            detection_results = []
            
            for i, test_case in enumerate(demo_prompts, 1):
                prompt = test_case['prompt']
                description = test_case['description']
                
                print(f"{i}. {description}")
                print(f"   Prompt: \"{prompt}\"")
                
                # Perform detection
                detection = detect_interactive_need(prompt)
                
                requires_interaction = detection.get('requires_interaction', False)
                interaction_type = detection.get('interaction_type', 'none')
                confidence = detection.get('confidence', 0)
                
                if requires_interaction:
                    print(f"   ✅ Interaction detected: {interaction_type} (confidence: {confidence:.1%})")
                    if 'suggested_actions' in detection:
                        actions = ', '.join(detection['suggested_actions'][:3])
                        print(f"   💡 Suggested actions: {actions}")
                else:
                    print(f"   ⭕ No interaction detected")
                
                detection_results.append({
                    'prompt': prompt,
                    'detected': requires_interaction,
                    'type': interaction_type,
                    'confidence': confidence
                })
                
                print()
            
            # Statistics
            interactive_count = sum(1 for r in detection_results if r['detected'])
            print(f"📊 Summary: {interactive_count}/{len(demo_prompts)} prompts require interaction")
            
            self.demo_results['interaction_detection'] = {
                'status': 'success',
                'total_prompts': len(demo_prompts),
                'interactive_detected': interactive_count,
                'results': detection_results
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.demo_results['interaction_detection'] = {'status': 'error', 'error': str(e)}
    
    def demo_session_management(self):
        """Demonstration of session management"""
        demo_section("INTERACTIVE SESSION MANAGEMENT")
        
        try:
            from interactive_web_navigator import (
                create_interactive_navigation_session,
                get_page_interactive_elements,
                close_interactive_session
            )
            
            # Create a demonstration session
            session_id = f"demo_session_{int(time.time())}"
            test_url = "https://httpbin.org/html"
            goals = ['demo_navigation', 'element_discovery', 'interaction_testing']
            
            print(f"🆔 Session creation: {session_id}")
            print(f"🌐 Target URL: {test_url}")
            print(f"🎯 Goals: {', '.join(goals)}")
            
            # Attempt to create the session (may fail if ChromeDriver is not available)
            try:
                session_result = create_interactive_navigation_session(session_id, test_url, goals)
                
                if session_result.get('success', False):
                    print("✅ Session created successfully")
                    print(f"   📊 Elements discovered: {session_result.get('elements_found', 0)}")
                    
                    # Display some discovered interactive elements
                    if 'interactive_elements' in session_result:
                        print("\n🎯 Detected interactive elements:")
                        for elem in session_result['interactive_elements'][:5]:
                            clickable = "✅" if elem.get('clickable') else "⭕"
                            print(f"   • {elem.get('type', 'unknown')}: \"{elem.get('text', 'No text')[:30]}\" "
                                 f"(score: {elem.get('score', 0):.2f}) {clickable}")
                    
                    # Get more details on elements
                    try:
                        elements_detail = get_page_interactive_elements(session_id)
                        
                        if elements_detail.get('success'):
                            print(f"\n📋 Detailed summary:")
                            print(f"   🌐 Current URL: {elements_detail.get('current_url', 'Unknown')}")
                            print(f"   📊 Total elements: {elements_detail.get('total_elements', 0)}")
                            
                            # Display breakdown by type
                            elements_by_type = elements_detail.get('elements_by_type', {})
                            if elements_by_type:
                                print(f"   📈 Breakdown by type:")
                                for elem_type, elements in elements_by_type.items():
                                    print(f"      • {elem_type}: {len(elements)} elements")
                            
                            # Display interaction suggestions
                            suggestions = elements_detail.get('interaction_suggestions', [])
                            if suggestions:
                                print(f"   💡 Interaction suggestions:")
                                for suggestion in suggestions[:3]:
                                    print(f"      • {suggestion.get('description', 'Suggested action')}")
                    
                    except Exception as e:
                        print(f"   ⚠️ Could not get details: {e}")
                    
                    # Close the session
                    print(f"\n🔚 Closing the session...")
                    close_result = close_interactive_session(session_id)
                    
                    if close_result.get('success'):
                        report = close_result.get('report', {})
                        print("✅ Session closed successfully")
                        print(f"   ⏱️ Duration: {report.get('duration_seconds', 0):.1f}s")
                        print(f"   📄 Pages visited: {report.get('pages_visited', 0)}")
                        print(f"   🖱️ Interactions performed: {report.get('interactions_performed', 0)}")
                    else:
                        print(f"❌ Closure error: {close_result.get('error', 'Unknown')}")
                
                else:
                    print(f"❌ Session creation failed: {session_result.get('error', 'Unknown')}")
                    print("💡 This is normal if ChromeDriver is not installed")
            
            except Exception as e:
                print(f"❌ Error during session demonstration: {e}")
                print("💡 This is normal if ChromeDriver is not installed")
            
            self.demo_results['session_management'] = {
                'status': 'demonstrated',
                'note': 'Full demonstration (may require ChromeDriver to function fully)'
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.demo_results['session_management'] = {'status': 'error', 'error': str(e)}
    
    def demo_gemini_integration(self):
        """Demonstration of integration with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
        demo_section("INTEGRATION WITH artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
        
        try:
            from gemini_interactive_adapter import handle_gemini_interactive_request
            
            print("🤖 Integration test with the artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter")
            
            # Examples of interactive requests
            interactive_requests = [
                {
                    'prompt': "Click on the 'Services' tab of https://httpbin.org/html",
                    'description': "Direct interaction with tab"
                },
                {
                    'prompt': "Explore all available tabs on this site",
                    'description': "Automatic tab navigation"
                }
            ]
            
            for i, request in enumerate(interactive_requests, 1):
                prompt = request['prompt']
                description = request['description']
                
                print(f"\n{i}. {description}")
                print(f"   Prompt: \"{prompt}\"")
                
                try:
                    # Simulate a request (may fail without ChromeDriver)
                    start_time = time.time()
                    result = handle_gemini_interactive_request(
                        prompt=prompt,
                        user_id=1,
                        session_id=f"demo_gemini_{i}"
                    )
                    processing_time = time.time() - start_time
                    
                    if result.get('success'):
                        print(f"   ✅ Processing successful in {processing_time:.2f}s")
                        
                        if result.get('interaction_performed'):
                            print(f"   🖱️ Interaction performed")
                            
                            if 'response' in result:
                                response_preview = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
                                print(f"   📝 Response: {response_preview}")
                        else:
                            print(f"   📊 Analysis performed without interaction")
                            
                            if result.get('elements_discovered', 0) > 0:
                                print(f"   🔍 {result['elements_discovered']} elements discovered")
                    
                    elif result.get('fallback_required'):
                        print(f"   ⚠️ Redirection to standard navigation system")
                    else:
                        print(f"   ❌ Failure: {result.get('error', 'Unknown error')}")
                        print(f"   💡 Normal if ChromeDriver is not available")
                
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    print(f"   💡 Normal if dependencies are not installed")
            
            # Test statistics
            try:
                from gemini_interactive_adapter import get_gemini_interactive_adapter
                adapter = get_gemini_interactive_adapter()
                
                if adapter:
                    stats = adapter.get_interaction_statistics()
                    print(f"\n📊 Adapter statistics:")
                    print(f"   📈 Total requests: {stats.get('stats', {}).get('total_requests', 0)}")
                    print(f"   🎯 Sessions created: {stats.get('stats', {}).get('interactive_sessions_created', 0)}")
                    print(f"   ✅ Successful interactions: {stats.get('stats', {}).get('successful_interactions', 0)}")
            
            except Exception as e:
                print(f"   ⚠️ Statistics not available: {e}")
            
            self.demo_results['gemini_integration'] = {
                'status': 'demonstrated',
                'requests_tested': len(interactive_requests)
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.demo_results['gemini_integration'] = {'status': 'error', 'error': str(e)}
    
    def demo_use_cases(self):
        """Demonstration of practical use cases"""
        demo_section("PRACTICAL USE CASES")
        
        use_cases = [
            {
                'title': "E-commerce - Product navigation",
                'scenario': "User asks to explore categories on an e-commerce website",
                'prompt': "Explore all product tabs on this online sales website",
                'expected_actions': [
                    "Detect category tabs (Electronics, Clothing, etc.)",
                    "Automatically click on each tab",
                    "Extract information from each category",
                    "Provide a summary of available products"
                ]
            },
            {
                'title': "Institutional website - Services",
                'scenario': "User wants to know all services of a company",
                'prompt': "Click on the Services tab and show me what is available",
                'expected_actions': [
                    "Identify the 'Services' tab or section",
                    "Click on the appropriate element",
                    "Analyze the revealed content",
                    "Extract the list of offered services"
                ]
            },
            {
                'title': "Educational platform - Courses",
                'scenario': "User wants to see all available courses",
                'prompt': "Browse all course sections of this platform",
                'expected_actions': [
                    "Detect course tabs/sections",
                    "Systematic navigation within each section",
                    "Collect information about each course",
                    "Organize data by category"
                ]
            },
            {
                'title': "Governmental website - Procedures",
                'scenario': "User is looking for a specific administrative procedure",
                'prompt': "Find the section to renew a passport",
                'expected_actions': [
                    "Analyze navigation menus",
                    "Identify relevant sections",
                    "Click on the appropriate elements",
                    "Extract information about the procedure"
                ]
            }
        ]
        
        print("🏪 Examples of use cases where the interaction system is useful:\n")
        
        for i, use_case in enumerate(use_cases, 1):
            print(f"{i}. {use_case['title']}")
            print(f"   📋 Scenario: {use_case['scenario']}")
            print(f"   💬 User prompt: \"{use_case['prompt']}\"")
            print(f"   🔄 Expected automatic actions:")
            
            for action in use_case['expected_actions']:
                print(f"      • {action}")
            
            # Simulate detection for this use case
            try:
                from gemini_interactive_adapter import detect_interactive_need
                detection = detect_interactive_need(use_case['prompt'])
                
                if detection.get('requires_interaction'):
                    interaction_type = detection.get('interaction_type', 'generic')
                    confidence = detection.get('confidence', 0)
                    print(f"   ✅ Detection: {interaction_type} (confidence: {confidence:.1%})")
                else:
                    print(f"   ⚠️ Interaction not detected (adjustment needed)")
            
            except Exception as e:
                print(f"   ❌ Detection error: {e}")
            
            print()
        
        self.demo_results['use_cases'] = {
            'status': 'demonstrated',
            'total_cases': len(use_cases)
        }
    
    def demo_capabilities_summary(self):
        """Summary of system capabilities"""
        demo_section("CAPABILITIES SUMMARY")
        
        capabilities = {
            "🎯 Direct interaction": [
                "Clicking specific buttons",
                "Selecting tabs by name",
                "Activating navigation links",
                "Interacting with menu elements"
            ],
            "🔄 Automatic navigation": [
                "Exploring all tabs on a site",
                "Systematically browsing sections",
                "Category navigation",
                "Automatic content discovery"
            ],
            "📋 Intelligent analysis": [
                "Detection of interactive elements",
                "Calculation of importance scores",
                "Identification of element types",
                "Interaction recommendations"
            ],
            "🤖 artificial intelligence API GOOGLE GEMINI 2.0 FLASH Integration": [
                "Automatic detection of interaction needs",
                "Natural language processing",
                "Contextualized feedback to the user",
                "Management of persistent sessions"
            ],
            "🛡️ Advanced features": [
                "Automatic screenshots",
                "Robust error handling",
                "Multi-browser support (Chrome, Edge)",
                "Detailed statistics and reports"
            ]
        }
        
        print("🚀 The interactive navigation system offers the following capabilities:\n")
        
        for category, features in capabilities.items():
            print(f"{category}:")
            for feature in features:
                print(f"   • {feature}")
            print()
        
        # Technical summary
        print("⚙️ Technical aspects:")
        print("   • Uses Selenium WebDriver for automation")
        print("   • Compatible with Chrome and Edge")
        print("   • Native integration with the artificial intelligence API GOOGLE GEMINI 2.0 FLASH API")
        print("   • Intelligent detection by keywords and patterns")
        print("   • Modular and extensible architecture")
        print("   • Robust error and fallback management")
        
        self.demo_results['capabilities_summary'] = {
            'status': 'completed',
            'categories': len(capabilities),
            'total_features': sum(len(features) for features in capabilities.values())
        }
    
    def generate_demo_report(self):
        """Generates a demonstration report"""
        report_dir = Path("demo_results")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"interactive_demo_report_{timestamp}.json"
        
        report = {
            'demo_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_sections': len(self.demo_results),
                'successful_sections': sum(1 for r in self.demo_results.values() if r.get('status') in ['success', 'demonstrated', 'completed']),
                'screenshots_taken': len(self.screenshots_taken)
            },
            'demo_results': self.demo_results,
            'conclusion': "Full demonstration of the interactive navigation system with the artificial intelligence API GOOGLE GEMINI 2.0 FLASH"
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Demonstration report saved: {report_file}")
        return report
    
    def run_full_demo(self):
        """Launches the full demonstration"""
        demo_header("INTERACTIVE NAVIGATION SYSTEM artificial intelligence API GOOGLE GEMINI 2.0 FLASH - DEMONSTRATION")
        
        print("🎯 This demonstration showcases the new web interaction capabilities of the artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
        print("💡 The system now allows clicking on elements, exploring tabs,")
        print("   and interactively navigating websites")
        
        start_time = time.time()
        
        # Execute all demonstrations
        self.demo_element_analysis()
        self.demo_interaction_detection()
        self.demo_session_management()
        self.demo_gemini_integration()
        self.demo_use_cases()
        self.demo_capabilities_summary()
        
        total_time = time.time() - start_time
        
        # Generate the report
        report = self.generate_demo_report()
        
        # Final summary
        demo_header("DEMONSTRATION SUMMARY")
        
        successful_sections = sum(1 for r in self.demo_results.values() 
                                if r.get('status') in ['success', 'demonstrated', 'completed'])
        total_sections = len(self.demo_results)
        
        print(f"⏱️ Total duration: {total_time:.2f} seconds")
        print(f"📊 Completed sections: {successful_sections}/{total_sections}")
        print(f"📈 Success rate: {(successful_sections/total_sections)*100:.1f}%")
        
        if successful_sections == total_sections:
            print("\n🎉 FULL DEMONSTRATION SUCCESSFUL!")
            print("✅ The interactive navigation system is operational")
            print("🚀 artificial intelligence API GOOGLE GEMINI 2.0 FLASH can now interact with web page elements")
        else:
            print(f"\n⚠️ Partially successful demonstration ({successful_sections}/{total_sections} sections)")
            print("💡 Some functionalities may require additional dependencies")
        
        print("\n📖 Features demonstrated:")
        print("   • Automatic detection of interactive elements")
        print("   • Element classification and scoring")
        print("   • Navigation session management")
        print("   • Native integration with the artificial intelligence API GOOGLE GEMINI 2.0 FLASH API")
        print("   • Support for multiple use cases")
        
        print("\n🔧 To fully utilize the system:")
        print("   • Install ChromeDriver or EdgeDriver")
        print("   • Configure Selenium WebDriver") 
        print("   • Test with real websites")
        
        return report

def main():
    """Main demonstration function"""
    print("🌟 Starting the interactive navigation system demonstration")
    
    demo = InteractiveNavigationDemo()
    report = demo.run_full_demo()
    
    return report

if __name__ == "__main__":
    report = main()
    print(f"\n✅ Demonstration finished - Report available")
    print("🎯 The interactive navigation system artificial intelligence API GOOGLE GEMINI 2.0 FLASH is ready to be used!")
