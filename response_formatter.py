"""
Module to improve the structure and presentation of artificial intelligence API GOOGLE GEMINI 2.0 FLASH's text responses.
This module ensures that responses are well-structured, with properly spaced
and correctly formatted paragraphs.
"""

import re
import logging
import random

# Logger Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_markdown_symbols(text):
    """
    Removes markdown symbols (* and **) from the text.
    
    Args:
        text: The text to clean
        
    Returns:
        The text without markdown symbols
    """
    if not text:
        return text
    
    # Remove ** (bold) symbols
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove * (italic) symbols
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove isolated * that might remain
    text = re.sub(r'\*+', '', text)
    
    return text

def remove_ending_periods(text):
    """
    Removes periods at the end of sentences.
    
    Args:
        text: The text to clean
        
    Returns:
        The text without ending periods
    """
    if not text:
        return text
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for paragraph in paragraphs:
        if paragraph.strip():
            # Remove periods at the end of the paragraph
            paragraph = paragraph.rstrip()
            while paragraph.endswith('.'):
                paragraph = paragraph[:-1].rstrip()
            cleaned_paragraphs.append(paragraph)
    
    return '\n\n'.join(cleaned_paragraphs)

def format_response(text):
    """
    Improves the structure of a text response.
    
    Args:
        text: The text to format
        
    Returns:
        The formatted text with better structure
    """
    if not text:
        return text
    
    # 1. Remove markdown symbols and ending periods BEFORE any formatting
    text = remove_markdown_symbols(text)
    text = remove_ending_periods(text)
    
    # 2. Clean up superfluous newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 2. Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # If the entire text is on a single line, try to split it into logical paragraphs
    if len(paragraphs) <= 1 and len(text) > 200:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        new_paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(sentences):
            current_paragraph.append(sentence)
            
            # Create a new paragraph after 3-4 sentences or if it's a long sentence
            if (len(current_paragraph) >= 3 and i < len(sentences) - 1) or len(sentence) > 150:
                new_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add the last paragraph if there are remaining sentences
        if current_paragraph:
            new_paragraphs.append(' '.join(current_paragraph))
        
        paragraphs = new_paragraphs
    
    # 3. Improve paragraphs individually
    improved_paragraphs = []
    
    # Linking words to introduce paragraphs
    transitions = [
        "First,", "Furthermore,", "Moreover,", "In addition,", 
        "Next,", "On the other hand,", "Also,",
        "Indeed,", "Regarding"
    ]
    
    # Possible concluding phrases
    conclusion_phrases = [
        "In conclusion,", "To summarize,", "In short,", "Ultimately,",
        "To conclude,", "In summary,", "In the end,", "In brief,"
    ]
    
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        
        # Skip empty paragraphs
        if not paragraph:
            continue
            
        # For bulleted lists, keep them as is
        if re.match(r'^[-*•] ', paragraph):
            improved_paragraphs.append(paragraph)
            continue
            
        # Add appropriate transitions for middle paragraphs
        if i > 0 and i < len(paragraphs) - 1 and len(paragraphs) > 2:
            # Check if the paragraph does not already start with a transition
            if not any(paragraph.startswith(trans) for trans in transitions + conclusion_phrases):
                if random.random() < 0.6:  # 60% chance to add a transition
                    paragraph = f"{random.choice(transitions)} {paragraph}"
        
        # Add a conclusion for the last paragraph
        if i == len(paragraphs) - 1 and len(paragraphs) > 1 and len(paragraph) > 50:
            # Check if the paragraph does not already start with a concluding phrase
            if not any(paragraph.startswith(concl) for concl in conclusion_phrases):
                if random.random() < 0.7:  # 70% chance to add a conclusion
                    paragraph = f"{random.choice(conclusion_phrases)} {paragraph}"
        
        improved_paragraphs.append(paragraph)
    
    # 4. Join paragraphs with a double newline
    formatted_text = '\n\n'.join(improved_paragraphs)
    
    # 5. Add newlines before and after lists
    formatted_text = re.sub(r'([.!?])\s*\n([-*•])', r'\1\n\n\2', formatted_text)
    
    # 6. Improve the structure of overly long sentences
    sentences = re.split(r'(?<=[.!?])\s+', formatted_text)
    improved_sentences = []
    
    for sentence in sentences:
        # Add commas in very long sentences without punctuation
        if len(sentence) > 180 and sentence.count(',') < 2:
            parts = re.split(r'\s+(?:and|or|because|therefore|but|thus|then|as)\s+', sentence, flags=re.IGNORECASE)
            if len(parts) > 1:
                improved_sentence = ""
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        connector_match = re.search(r'\s+(and|or|because|therefore|but|thus|then|as)\s+', sentence[len(improved_sentence):], flags=re.IGNORECASE)
                        if connector_match:
                            improved_sentence += part + ", " + connector_match.group(0).strip() + " "
                        else:
                            improved_sentence += part + " "
                    else:
                        improved_sentence += part
                sentence = improved_sentence
        
        improved_sentences.append(sentence)
    
    formatted_text = ' '.join(improved_sentences)
    
    # 7. Finally, ensure there are two newlines between paragraphs
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
    
    return formatted_text

if __name__ == "__main__":
    # Test
    sample_text = "Artificial intelligence is a fascinating field that is evolving rapidly. Models like artificial intelligence API GOOGLE GEMINI 2.0 FLASH use deep neural networks to understand and generate natural language. These models are trained on enormous amounts of text data from the Internet. They thus learn the patterns and structures of human language. The applications are numerous: virtual assistance, content generation, machine translation, and much more. Research continues to advance with increasingly powerful architectures."
    
    print("=== ORIGINAL TEXT ===")
    print(sample_text)
    print("\n=== FORMATTED TEXT ===")
    print(format_response(sample_text))
