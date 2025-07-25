"""
Advanced Vision Adapter for the Google Gemini 2.0 Flash AI API
Allows Google Gemini 2.0 Flash AI to "see" and visually analyze the interior of websites
"""

import base64
import json
import logging
import requests
import io
from PIL import Image
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from datetime import datetime

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiVisualAdapter:
    """Adapter for the advanced visual capabilities of Google Gemini 2.0 Flash AI"""
    
    def __init__(self, api_key: str = None):
        """
        Initializes the Google Gemini 2.0 Flash AI vision adapter
        
        Args:
            api_key: Google Gemini 2.0 Flash AI API key (uses default key if not specified)
        """
        self.api_key = api_key or "AIzaSyDdWKdpPqgAVLet6_mchFxmG_GXnfPx2aQ"
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Configuration for image optimization
        self.max_image_size = (1024, 1024)  # Max size for the AI
        self.image_quality = 85  # JPEG quality for optimization
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_processing_time': 0
        }
        
        logger.info("🤖 Google Gemini 2.0 Flash AI Vision Adapter initialized")
    
    def encode_image_for_gemini(self, image_path: str) -> Optional[str]:
        """
        Encodes an image for the Google Gemini 2.0 Flash AI multimodal API
        
        Args:
            image_path: Path to the image to encode
            
        Returns:
            Base64 encoded image or None if error
        """
        try:
            # Open and optimize the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                    logger.info(f"📏 Image resized: {img.size}")
                
                # Save to memory
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=self.image_quality, optimize=True)
                
                # Encode to base64
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                logger.info(f"✅ Image encoded: {len(image_data)} characters")
                return image_data
                
        except Exception as e:
            logger.error(f"❌ Image encoding error {image_path}: {e}")
            return None
    
    def analyze_website_screenshot(self, 
                                 image_path: str, 
                                 analysis_prompt: str,
                                 context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes a website screenshot with Google Gemini 2.0 Flash AI Vision
        
        Args:
            image_path: Path to the screenshot
            analysis_prompt: Specific analysis prompt
            context: Additional textual context
            
        Returns:
            Analysis result with metadata
        """
        start_time = datetime.now()
        
        try:
            # Encode the image
            encoded_image = self.encode_image_for_gemini(image_path)
            if not encoded_image:
                return {
                    'success': False,
                    'error': 'Could not encode image',
                    'analysis': None
                }
            
            # Build the visual analysis prompt
            context_text = context or "General website analysis"
            visual_prompt = f"""🤖 VISUAL ANALYSIS OF A WEBSITE

**CONTEXT**: {context_text}

**ANALYSIS INSTRUCTIONS**:
{analysis_prompt}

**SPECIFIC TASKS**:
1. 📋 **Structure and Layout**: Describe the general structure, navigation, main areas
2. 🎨 **Design and UX**: Analyze colors, fonts, spacing, readability
3. 📝 **Visible Content**: Identify and summarize the main textual content
4. 🔗 **Interactive Elements**: Visible buttons, links, forms, menus
5. 📱 **Responsive Design**: Clues on mobile/desktop adaptation
6. ⚡ **Potential Issues**: Errors, broken elements, accessibility issues
7. 🎯 **Site Objective**: Determine the main purpose of the page
8. 💡 **Recommendations**: Suggestions for UX/UI improvement

**RESPONSE FORMAT**: Structure your analysis with these sections and use emojis for readability.
"""

            # Prepare the multimodal request
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [{
                    "parts": [
                        {
                            "text": visual_prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_image
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.4,  # Lower for precise analyses
                    "topK": 32,
                    "topP": 0.8,
                    "maxOutputTokens": 3000,  # Higher for detailed analyses
                }
            }
            
            # Send the request
            url = f"{self.api_url}?key={self.api_key}"
            logger.info("📤 Sending visual analysis request to Google Gemini 2.0 Flash AI...")
            
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            # Process the response
            if response.status_code == 200:
                response_data = response.json()
                
                if 'candidates' in response_data and response_data['candidates']:
                    analysis = response_data['candidates'][0]['content']['parts'][0]['text']
                    
                    # Calculate metrics
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # Update statistics
                    self.stats['images_processed'] += 1
                    self.stats['successful_analyses'] += 1
                    self.stats['total_processing_time'] += processing_time
                    
                    logger.info(f"✅ Visual analysis successful in {processing_time:.2f}s")
                    
                    return {
                        'success': True,
                        'analysis': analysis,
                        'image_path': image_path,
                        'processing_time': processing_time,
                        'image_size': os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                        'analysis_length': len(analysis),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = "No valid response from Google Gemini 2.0 Flash AI"
                    logger.error(f"❌ {error_msg}")
                    self.stats['failed_analyses'] += 1
                    
                    return {
                        'success': False,
                        'error': error_msg,
                        'analysis': None
                    }
            else:
                error_msg = f"Google Gemini 2.0 Flash AI API error: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                self.stats['failed_analyses'] += 1
                
                return {
                    'success': False,
                    'error': error_msg,
                    'analysis': None
                }
                
        except Exception as e:
            error_msg = f"Visual analysis error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.stats['failed_analyses'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'analysis': None
            }
    
    def compare_website_changes(self, 
                              image_path_before: str,
                              image_path_after: str,
                              comparison_context: str = "") -> Dict[str, Any]:
        """
        Compares two screenshots to detect changes
        
        Args:
            image_path_before: Screenshot before
            image_path_after: Screenshot after
            comparison_context: Comparison context
            
        Returns:
            Comparative analysis
        """
        try:
            # Encode both images
            encoded_before = self.encode_image_for_gemini(image_path_before)
            encoded_after = self.encode_image_for_gemini(image_path_after)
            
            if not encoded_before or not encoded_after:
                return {
                    'success': False,
                    'error': 'Could not encode one or more images',
                    'comparison': None
                }
            
            # Comparison prompt
            comparison_prompt = f"""🔍 VISUAL COMPARISON OF WEBSITES

**CONTEXT**: {comparison_context}

**INSTRUCTIONS**:
Compare these two screenshots and identify:

1. 🆚 **Visual Differences**: Changes in layout, colors, elements
2. ➕ **New Elements**: What has been added
3. ➖ **Removed Elements**: What has disappeared
4. 🔄 **Modifications**: Modified elements (text, position, style)
5. 📊 **UX Impact**: How these changes affect the user experience
6. ⚖️ **Overall Evaluation**: Are the changes positive or negative?

**FIRST IMAGE (BEFORE)**:
"""

            # Build the request with both images
            data = {
                "contents": [{
                    "parts": [
                        {"text": comparison_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_before
                            }
                        },
                        {"text": "\n\n**SECOND IMAGE (AFTER)**:"},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg", 
                                "data": encoded_after
                            }
                        },
                        {"text": "\n\nPlease now perform the detailed comparison."}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 32,
                    "topP": 0.8,
                    "maxOutputTokens": 3000,
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            url = f"{self.api_url}?key={self.api_key}"
            
            logger.info("🔍 Sending visual comparison request...")
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'candidates' in response_data and response_data['candidates']:
                    comparison = response_data['candidates'][0]['content']['parts'][0]['text']
                    
                    logger.info("✅ Visual comparison successful")
                    
                    return {
                        'success': True,
                        'comparison': comparison,
                        'image_before': image_path_before,
                        'image_after': image_path_after,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'success': False,
                'error': f"API Error: {response.status_code}",
                'comparison': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Comparison error: {str(e)}",
                'comparison': None
            }
    
    def analyze_ui_elements(self, image_path: str, element_types: List[str] = None) -> Dict[str, Any]:
        """
        Specific analysis of UI elements in a screenshot
        
        Args:
            image_path: Path to the capture
            element_types: Types of elements to analyze (buttons, forms, navigation, etc.)
            
        Returns:
            Detailed analysis of UI elements
        """
        if element_types is None:
            element_types = ['buttons', 'forms', 'navigation', 'content', 'images', 'links']
        
        elements_list = ", ".join(element_types)
        
        ui_prompt = f"""🎯 SPECIALIZED UI ELEMENTS ANALYSIS

**FOCUS ON**: {elements_list}

**DETAILED INSTRUCTIONS**:
1. 🔘 **Buttons**: Identify all buttons (CTA, navigation, action)
2. 📝 **Forms**: Fields, labels, validation, accessibility
3. 🧭 **Navigation**: Menus, breadcrumbs, navigation links
4. 📄 **Content**: Hierarchy, readability, organization
5. 🖼️ **Images**: Relevance, quality, optimization
6. 🔗 **Links**: Visibility, differentiation, call-to-action

**FOR EACH ELEMENT**:
- Position and visibility
- State (active, hover, disabled)
- Accessibility (contrast, size)
- Consistency with the design system
- Improvement recommendations

**FORMAT**: Organize by element type with 1-5 ⭐ rating
"""

        return self.analyze_website_screenshot(
            image_path=image_path,
            analysis_prompt=ui_prompt,
            context=f"Specialized UI Analysis - Focus on: {elements_list}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns adapter usage statistics
        
        Returns:
            Dictionary with statistics
        """
        avg_processing_time = (
            self.stats['total_processing_time'] / max(self.stats['images_processed'], 1)
        )
        
        success_rate = (
            self.stats['successful_analyses'] / max(self.stats['images_processed'], 1) * 100
        )
        
        return {
            'images_processed': self.stats['images_processed'],
            'successful_analyses': self.stats['successful_analyses'],
            'failed_analyses': self.stats['failed_analyses'],
            'success_rate': round(success_rate, 2),
            'average_processing_time': round(avg_processing_time, 2),
            'total_processing_time': round(self.stats['total_processing_time'], 2)
        }
    
    def reset_statistics(self):
        """Resets statistics"""
        self.stats = {
            'images_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_processing_time': 0
        }
        logger.info("📊 Statistics reset")

# Global instance for easy use
gemini_visual_adapter = None

def initialize_gemini_visual_adapter(api_key: str = None) -> GeminiVisualAdapter:
    """
    Initializes the global Google Gemini 2.0 Flash AI vision adapter
    
    Args:
        api_key: Optional API key
        
    Returns:
        Adapter instance
    """
    global gemini_visual_adapter
    
    if gemini_visual_adapter is None:
        gemini_visual_adapter = GeminiVisualAdapter(api_key)
        logger.info("🚀 Global Google Gemini 2.0 Flash AI Vision Adapter initialized")
    
    return gemini_visual_adapter

def get_gemini_visual_adapter() -> Optional[GeminiVisualAdapter]:
    """
    Returns the global vision adapter instance
    
    Returns:
        Adapter instance or None if not initialized
    """
    global gemini_visual_adapter
    return gemini_visual_adapter

if __name__ == "__main__":
    # Adapter test
    adapter = initialize_gemini_visual_adapter()
    print("🧪 Google Gemini 2.0 Flash AI Vision Adapter ready for tests")
    print(f"📊 Statistics: {adapter.get_statistics()}")
