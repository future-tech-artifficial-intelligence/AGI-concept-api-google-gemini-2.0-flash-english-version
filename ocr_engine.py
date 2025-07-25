import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from datetime import datetime
import re
from typing import Dict, Any, List, Tuple, Optional

# Tesseract OCR configuration allows artificial intelligence API GOOGLE GEMINI 2.0 FLASH to
# better analyze ancient manuscripts in Latin, for example, or documents requiring high precision like old historical documents (path to Tesseract executable)
# To be modified according to your system installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRProcessor:
    """Class to manage OCR processing of images"""
    
    def __init__(self, language: str = 'eng'): # Changed default language to English for consistency
        """
        Initializes the OCR processor
        
        Args:
            language (str): Language code for Tesseract (fra=French, eng=English)
        """
        self.language = language
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    def _check_image_format(self, image_path: str) -> bool:
        """Checks if the image format is supported"""
        _, ext = os.path.splitext(image_path.lower())
        return ext in self.supported_formats
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Image preprocessing to improve OCR accuracy
        
        Args:
            image: Image in numpy array format
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding to improve contrast
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                              cv2.THRESH_BINARY, 11, 2)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Edge enhancement
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        return eroded
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts text from an image using OCR
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing the extracted text and metadata
        """
        result = {
            "success": False,
            "error": None,
            "text": "",
            "confidence": 0,
            "timestamp": datetime.now().isoformat(),
            "language": self.language,
        }
        
        try:
            # Check image format
            if not self._check_image_format(image_path):
                result["error"] = f"Unsupported image format. Accepted formats: {', '.join(self.supported_formats)}"
                return result
                
            # Check if the file exists
            if not os.path.exists(image_path):
                result["error"] = "The image file does not exist"
                return result
                
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                result["error"] = "Unable to load image"
                return result
                
            # Preprocess the image
            processed_image = self._preprocess_image(image)
            
            # Perform OCR
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence scores
            text_parts = []
            total_confidence = 0
            word_count = 0
            
            for i in range(len(ocr_data["text"])):
                if int(ocr_data["conf"][i]) > 0:  # Ignore entries with zero confidence
                    word = ocr_data["text"][i].strip()
                    if word:
                        text_parts.append(word)
                        total_confidence += int(ocr_data["conf"][i])
                        word_count += 1
            
            # Assemble full text
            full_text = " ".join(text_parts)
            
            # Calculate average confidence
            avg_confidence = total_confidence / word_count if word_count > 0 else 0
            
            # Prepare the result
            result["success"] = True
            result["text"] = full_text
            result["confidence"] = avg_confidence
            result["word_count"] = word_count
            
        except Exception as e:
            result["error"] = f"Error during OCR: {str(e)}"
            
        return result
    
    def analyze_text_content(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the extracted text content to derive structured information
        
        Args:
            ocr_result: OCR extraction result
            
        Returns:
            Dictionary with content analysis
        """
        analysis = {
            "type": "unknown",
            "key_information": {},
            "summary": "",
            "keywords": []
        }
        
        if not ocr_result["success"] or not ocr_result["text"]:
            return analysis
            
        text = ocr_result["text"]
        
        # Document type detection
        if re.search(r'invoice|receipt|payment|total', text, re.IGNORECASE):
            analysis["type"] = "financial_document"
            
            # Amount extraction
            amounts = re.findall(r'\d+[,\.]\d+\s*€|\d+[,\.]\d+\s*EUR|\d+\s*€|\d+\s*EUR|\$\s*\d+[,\.]\d+|\d+[,\.]\d+\s*\$', text)
            if amounts:
                analysis["key_information"]["amounts"] = amounts
                
            # Date extraction
            dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}', text, re.IGNORECASE)
            if dates:
                analysis["key_information"]["dates"] = dates
                
        elif re.search(r'identity card|passport|driver\'s license|driving', text, re.IGNORECASE):
            analysis["type"] = "identity_document"
            
            # Name extraction
            names = re.findall(r'name\s*:\s*([^\n\r]+)|first name\s*:\s*([^\n\r]+)', text, re.IGNORECASE)
            if names:
                analysis["key_information"]["names"] = [n[0] or n[1] for n in names if n[0] or n[1]]
                
        elif re.search(r'article|news|journal|report', text, re.IGNORECASE):
            analysis["type"] = "article"
            
            # Extract a potential title (first line)
            lines = text.split('\n')
            if lines and len(lines[0]) > 5:
                analysis["key_information"]["title"] = lines[0]
                
        # Keyword extraction (frequently appearing words)
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Get the top 5 most frequent words as keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        analysis["keywords"] = [word for word, _ in sorted_words[:5]]
        
        # Generate a simple summary (first 100 characters or first sentence)
        first_sentence = re.split(r'[.!?]', text)[0].strip()
        analysis["summary"] = first_sentence[:100] + ("..." if len(first_sentence) > 100 else "")
        
        return analysis

def process_image_request(request_text: str) -> Tuple[bool, Optional[str]]:
    """
    Analyzes a user request to detect if it is an OCR request
    
    Args:
        request_text: User request text
        
    Returns:
        (is_ocr_request, image_path): Indicates if it's an OCR request and the image path
    """
    # Patterns to detect an OCR request
    ocr_patterns = [
        r'analyze\s+this\s+image',
        r'analyze\s+the\s+image',
        r'extract\s+(?:text)\s+(?:from|in)\s+(?:this|the)\s+image',
        r'OCR\s+(?:on|for)\s+(?:this|the)\s+image',
        r'read\s+(?:this|the)\s+image',
        r'what\s+(?:does|is|says)\s+(?:this|the)\s+image\s+(?:say|contain)',
        r'text\s+(?:in|on|from)\s+(?:this|the)\s+image'
    ]
    
    # Patterns to extract the image path
    path_patterns = [
        r'(?:image|photo|file|document)\s+(?:at|in|of|:)\s+[\'"]?([^\'"\s]+\.(jpg|jpeg|png|bmp|tiff|gif))[\'"]?',
        r'[\'"]([^\'"\s]+\.(jpg|jpeg|png|bmp|tiff|gif))[\'"]',
        r'([a-zA-Z]:\\[^\\/:*?"<>|\r\n]+\.(jpg|jpeg|png|bmp|tiff|gif))',
        r'(/[^\\/:*?"<>|\r\n]+\.(jpg|jpeg|png|bmp|tiff|gif))'
    ]
    
    # Check if the request matches an OCR request
    is_ocr_request = any(re.search(pattern, request_text, re.IGNORECASE) for pattern in ocr_patterns)
    
    # If it's an OCR request, try to extract the image path
    image_path = None
    if is_ocr_request:
        for pattern in path_patterns:
            match = re.search(pattern, request_text, re.IGNORECASE)
            if match:
                image_path = match.group(1)
                break
    
    return is_ocr_request, image_path
