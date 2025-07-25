# 🎯 User Guide - Gemini Interactive Navigation System

## 🎯 Overview

The interactive navigation system allows the artificial intelligence API GOOGLE GEMINI 2.0 FLASH to interact directly with website elements (tabs, buttons, links, etc.) for smarter, automated navigation.

## 📋 Table of Contents

1.  [Installation and Configuration](#installation-and-configuration)
2.  [Quick Start](#quick-start)
3.  [Supported Interaction Types](#supported-interaction-types)
4.  [Usage Examples](#usage-examples)
5.  [Advanced Configuration](#advanced-configuration)
6.  [Troubleshooting](#troubleshooting)
7.  [Developer API](#api-de-développeur)

---

## 🚀 Installation and Configuration

### Prerequisites

1.  **Python 3.8+** with existing project dependencies
2.  **Selenium WebDriver** for browser automation
3.  **ChromeDriver** or **EdgeDriver** installed

### Automatic Installation

```bash
# Run the dependency installation script
python install_dependencies.py

# Install specific Selenium dependencies
pip install selenium webdriver-manager
```

### Manual ChromeDriver Installation

**Windows:**
```bash
# Download and place ChromeDriver in PATH
# Or use webdriver-manager (recommended)
pip install webdriver-manager
```

**Installation verification:**
```python
python -c "from selenium import webdriver; print('Selenium OK')"
```

---

## ⚡ Quick Start

### 1. System Test

```bash
# Test the complete installation
python test_interactive_navigation.py

# See a demo
python demo_interactive_navigation.py
```

### 2. First Use with artificial intelligence API GOOGLE GEMINI 2.0 FLASH

```python
from gemini_api_adapter import GeminiAPI

# Create a Gemini instance with the interactive system
gemini = GeminiAPI()

# Examples of interactive prompts
prompts = [
    "Click on the 'Services' tab of https://example.com",
    "Explore all tabs on this website",
    "Browse all sections to see available options"
]

# Normal usage
for prompt in prompts:
    response = gemini.get_response(prompt, user_id=1)
    print(response['response'])
```

### 3. Feature Verification

```python
# Verify that the interactive system is active
from gemini_interactive_adapter import get_gemini_interactive_adapter

adapter = get_gemini_interactive_adapter()
if adapter:
    print("✅ Interactive system operational")
    stats = adapter.get_interaction_statistics()
    print(f"📊 Statistics: {stats}")
else:
    print("❌ Interactive system not available")
```

---

## 🎯 Supported Interaction Types

### 1. Direct Interaction
**Description:** Clicking a specific element mentioned by the user.

**Prompt examples:**
- `"Click the 'Next' button"`
- `"Press the 'Products' tab"`
- `"Select the 'Learn more' link"`

**Detected keywords:**
- `click on`, `click`
- `press on`, `press`
- `select`, `select`

### 2. Tab Navigation
**Description:** Systematically exploring all tabs on a page.

**Prompt examples:**
- `"Explore all tabs on this site"`
- `"Browse all available sections"`
- `"Go into all tabs to see the content"`

**Features:**
- Automatic tab detection
- Sequential navigation
- Content extraction from each tab
- Summary of findings

### 3. Full Exploration
**Description:** Automatic and exhaustive navigation of a site.

**Prompt examples:**
- `"Explore all options on this website"`
- `"Browse all menus and sections"`
- `"Complete analysis of all features"`

**Automatic actions:**
- Identification of interactive elements
- Clicks on important elements
- Navigation into subsections
- Compilation of found information

### 4. Form Interaction
**Description:** Analysis and interaction with web forms.

**Prompt examples:**
- `"Analyze the contact form"`
- `"Find the search fields"`
- `"Show me the filtering options"`

**Security Note:** The system identifies forms but does not automatically enter data for security reasons.

---

## 📖 Usage Examples

### Example 1: E-commerce - Explore Categories

```python
from gemini_api_adapter import GeminiAPI

gemini = GeminiAPI()

prompt = """
Explore all category tabs on https://example-shop.com 
and give me a summary of the products available in each section.
"""

response = gemini.get_response(prompt, user_id=1)
print(response['response'])
```

**Expected result:**
```
✅ I have explored 5 tabs on the site.

📋 Content of discovered tabs:
• Electronics: 150+ products including smartphones, computers, accessories
• Clothing: Men's/women's collection with 200+ fashion items
• Home & Garden: Furniture, decoration, gardening tools (80+ items)
• Sports: Sports equipment, technical clothing (120+ products)
• Books: Large selection of digital and paper books (500+ titles)

💡 Interaction suggestions:
• Explore electronics subcategories
• Check current promotions
• Analyze customer reviews
```

### Example 2: Institutional Site - Services

```python
prompt = """
Click on the 'Services' tab of https://company-website.com 
and list all services offered.
"""

response = gemini.get_response(prompt, user_id=1)
```

**Expected result:**
```
✅ I clicked on 'Services' and analyzed the content.

📄 The page changed after this interaction.

🏢 Services offered by the company:
• Digital strategy consulting
• Web application development
• Training in new technologies
• 24/7 Technical support
• IT security audit

📍 Current page: https://company-website.com/services
```

### Example 3: Information Search

```python
prompt = """
On this university's website, find the registration section 
and show me the steps to follow.
"""

response = gemini.get_response(prompt, user_id=1, session_id="university_search")
```

---

## ⚙️ Advanced Configuration

### 1. Browser Configuration

```python
from interactive_web_navigator import get_interactive_navigator

navigator = get_interactive_navigator()

# Modify configuration
navigator.config.update({
    'max_interactions_per_session': 100,  # Interaction limit
    'interaction_timeout': 45,            # Timeout in seconds
    'page_load_timeout': 20,              # Load timeout
    'screenshot_on_interaction': True     # Automatic screenshots
})
```

### 2. CSS Selector Configuration

```python
from interactive_web_navigator import InteractiveElementAnalyzer

analyzer = InteractiveElementAnalyzer()

# Add custom selectors
analyzer.element_selectors['custom_buttons'] = [
    '.my-custom-button',
    '[data-action="submit"]',
    '.special-interactive-element'
]

# Modify importance keywords
analyzer.importance_keywords['high'].extend(['buy', 'order', 'reserve'])
```

### 3. Statistics Configuration

```python
from gemini_interactive_adapter import get_gemini_interactive_adapter

adapter = get_gemini_interactive_adapter()

# Display detailed statistics
stats = adapter.get_interaction_statistics()
print(f"📊 Full statistics:")
print(f"   🔢 Total requests: {stats['stats']['total_requests']}")
print(f"   🎯 Sessions created: {stats['stats']['interactive_sessions_created']}")
print(f"   ✅ Successful interactions: {stats['stats']['successful_interactions']}")
print(f"   📂 Tabs explored: {stats['stats']['tabs_explored']}")

# Clean up old sessions
adapter.cleanup_sessions(max_age_hours=1)
```

---

## 🔧 Troubleshooting

### Common Problems

#### 1. ChromeDriver not found
**Error:**
```
selenium.common.exceptions.WebDriverException: 'chromedriver' executable needs to be in PATH
```

**Solutions:**
```bash
# Option 1: Install webdriver-manager
pip install webdriver-manager

# Option 2: Manually download ChromeDriver
# https://chromedriver.chromium.org/
# Place in system PATH
```

#### 2. Elements not clickable
**Error:**
```
Element not clickable at point (x, y)
```

**Solutions:**
- The system automatically attempts a JavaScript click
- Verify that the page is fully loaded
- Increase timeouts in the configuration

#### 3. Interaction detection fails
**Symptom:** The artificial intelligence API GOOGLE GEMINI 2.0 FLASH does not detect that an interaction is needed.

**Solutions:**
```python
# Test detection manually
from gemini_interactive_adapter import detect_interactive_need

result = detect_interactive_need("Your prompt here")
print(f"Detection: {result}")

# Adjust keywords if necessary
# See "Advanced Configuration" section
```

#### 4. Stuck sessions
**Symptom:** Sessions that do not close correctly.

**Solution:**
```python
# Force cleanup
from gemini_interactive_adapter import get_gemini_interactive_adapter

adapter = get_gemini_interactive_adapter()
adapter.cleanup_sessions(max_age_hours=0)  # Cleans all sessions
```

### Logs and Debugging

```python
import logging

# Enable detailed logs
logging.getLogger('InteractiveWebNavigator').setLevel(logging.DEBUG)
logging.getLogger('GeminiInteractiveIntegration').setLevel(logging.DEBUG)

# See real-time logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

---

## 🔌 Developer API

### 1. Direct Browser Usage

```python
from interactive_web_navigator import initialize_interactive_navigator

# Initialize the navigator
navigator = initialize_interactive_navigator()

# Create a session
session_id = "my_custom_session"
session = navigator.create_interactive_session(
    session_id=session_id,
    start_url="https://example.com",
    navigation_goals=['explore_tabs', 'find_content']
)

# Navigate to a URL
result = navigator.navigate_to_url(session_id, "https://example.com")
print(f"Elements found: {result['elements_found']}")

# Interact with an element
elements = result['interactive_elements']
if elements:
    element_id = elements[0]['id']
    interaction_result = navigator.interact_with_element(session_id, element_id, 'click')
    print(f"Interaction successful: {interaction_result.success}")

# Close the session
navigator.close_session(session_id)
```

### 2. Create Custom Detectors

```python
from gemini_interactive_adapter import GeminiInteractiveWebAdapter

class CustomInteractiveAdapter(GeminiInteractiveWebAdapter):
    def detect_custom_interaction(self, prompt):
        """Custom detector for specific interactions"""
        if "my_special_keyword" in prompt.lower():
            return {
                'requires_interaction': True,
                'interaction_type': 'custom_action',
                'confidence': 0.95
            }
        return {'requires_interaction': False}
    
    def handle_custom_interaction(self, prompt, session_id):
        """Custom handler"""
        # Your custom logic here
        return {
            'success': True,
            'custom_action_performed': True,
            'details': 'Custom action performed'
        }

# Use the custom adapter
custom_adapter = CustomInteractiveAdapter()
```

### 3. Custom Element Analyzer

```python
from interactive_web_navigator import InteractiveElementAnalyzer

class CustomElementAnalyzer(InteractiveElementAnalyzer):
    def __init__(self):
        super().__init__()
        
        # Add custom selectors
        self.element_selectors['my_custom_elements'] = [
            '.my-special-button',
            '[data-custom="interactive"]'
        ]
    
    def custom_scoring_logic(self, element_text, attributes):
        """Custom scoring logic"""
        score = 0.5  # Base score
        
        # Your custom logic
        if 'important' in element_text.lower():
            score += 0.3
        
        return min(score, 1.0)

# Use the custom analyzer
analyzer = CustomElementAnalyzer()
```

---

## 📊 Metrics and Monitoring

### Available Statistics

```python
from gemini_interactive_adapter import get_gemini_interactive_adapter
from interactive_web_navigator import get_interactive_navigator

# Gemini adapter statistics
adapter = get_gemini_interactive_adapter()
adapter_stats = adapter.get_interaction_statistics()

print("📈 Adapter statistics:")
print(f"   Total requests: {adapter_stats['stats']['total_requests']}")
print(f"   Sessions created: {adapter_stats['stats']['interactive_sessions_created']}")
print(f"   Successful interactions: {adapter_stats['stats']['successful_interactions']}")
print(f"   Tabs explored: {adapter_stats['stats']['tabs_explored']}")
print(f"   Forms interacted: {adapter_stats['stats']['forms_interacted']}")

# Navigator statistics
navigator = get_interactive_navigator()
nav_stats = navigator.get_statistics()

print("\n🔍 Navigator statistics:")
print(f"   Active sessions: {nav_stats['active_sessions']}")
print(f"   Interactions performed: {nav_stats['stats']['interactions_performed']}")
print(f"   Elements discovered: {nav_stats['stats']['elements_discovered']}")
print(f"   Pages navigated: {nav_stats['stats']['pages_navigated']}")
```

### Real-time Monitoring

```python
import time
from datetime import datetime

def monitor_interactive_system():
    """Continuous system monitoring"""
    adapter = get_gemini_interactive_adapter()
    
    while True:
        stats = adapter.get_interaction_statistics()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{timestamp}] Interactions: {stats['stats']['successful_interactions']}, "
              f"Sessions: {stats['stats']['interactive_sessions_created']}")
        
        time.sleep(30)  # Check every 30 seconds

# Launch monitoring
# monitor_interactive_system()
```

---

## 🛡️ Best Practices and Security

### 1. Website Respect

```python
# Add delays between interactions
navigator.config['interaction_delay'] = 2.0  # 2 seconds between each action

# Limit the number of interactions per session
navigator.config['max_interactions_per_session'] = 20

# Respect robots.txt (to be implemented as needed)```

### 2. Error Handling

```python
try:
    result = navigator.interact_with_element(session_id, element_id, 'click')
    if not result.success:
        print(f"Interaction failed: {result.error_message}")
        # Fallback logic
except Exception as e:
    print(f"Critical error: {e}")
    # Cleanup and recovery
```

### 3. Responsible Use

-   **Request Frequency:** Avoid overloading servers
-   **Personal Data:** Never automatically enter sensitive information
-   **Terms of Service Compliance:** Verify that automation is permitted
-   **Monitoring:** Monitor performance and errors

---

## 📞 Support and Contribution

### Report a Problem

1.  **Create a test report:**
    ```bash
    python test_interactive_navigation.py
    ```

2.  **Include logs:**
    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # Reproduce the problem
    ```

3.  **System Information:**
    -   Python version
    -   Selenium version
    -   Browser used (Chrome/Edge)
    -   Operating system

### Contribute to the Project

1.  **Tests:** Add tests for new use cases
2.  **Selectors:** Improve element detection
3.  **Documentation:** Enrich this guide with your feedback

---

## 🎉 Conclusion

The interactive navigation system transforms the artificial intelligence API GOOGLE GEMINI 2.0 FLASH into an assistant capable of physically interacting with websites. This functionality opens up new possibilities for:

-   **Web task automation**
-   **Intelligent content exploration**
-   **Advanced user assistance**
-   **Complex site analysis**

**Recommended next steps:**
1.  Test the system with `python demo_interactive_navigation.py`
2.  Start with simple interactions
3.  Experiment with your own use cases
4.  Contribute to system improvements

---

*Guide updated on July 24, 2025 - Version 1.0*
