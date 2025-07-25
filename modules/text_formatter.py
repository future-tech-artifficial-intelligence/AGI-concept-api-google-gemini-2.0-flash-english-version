"""
Advanced formatting module to enhance the structure of Artificial Intelligence responses API GOOGLE GEMINI 2.0 FLASH
This module ensures that responses are well-structured, properly spaced,
with appropriately lengthed paragraphs and a fluent writing style.
"""

import re
import logging
import random
from typing import Dict, Any, List, Optional

# Module metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 20,  # High priority to execute after generation but before other modifications
    "description": "Enhances the structure and presentation of text responses",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_response"]
}

# Logger configuration
logger = logging.getLogger(__name__)

def remove_markdown_symbols(text: str) -> str:
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

def remove_ending_periods(text: str) -> str:
    """
    Removes periods at the end of sentences/paragraphs.
    
    Args:
        text: The text to clean
        
    Returns:
        The text without trailing periods
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

# Connectors to create longer, more fluid sentences
SENTENCE_CONNECTORS = [
    ", which means that", 
    ", especially since", 
    " since", 
    " while", 
    " so that",
    " even as",
    ", knowing that",
    " although",
    " given that",
    ", and consequently,"
]

# Connectors to introduce examples or clarifications
CLARIFICATION_CONNECTORS = [
    "In other words,",
    "To be more precise,",
    "It is important to note that",
    "It should be emphasized that",
    "To illustrate this point,",
    "Which essentially implies that",
    "Put differently,", # 'Autrement dit' can also be 'In other words,'
    "In particular,"
]

# Expressions to introduce different parts of the text
SECTION_TRANSITIONS = [
    "First of all,",
    "Next,", 
    "Moreover,",
    "Furthermore,",
    "In addition,",
    "On the other hand,",
    "To go further,",
    "To conclude,",
    "In summary,"
]

def format_paragraphs(text: str) -> str:
    """
    Ensures that the text is correctly divided into paragraphs with appropriate spacing.
    
    Args:
        text: The text to format
        
    Returns:
        The formatted text with properly spaced paragraphs
    """
    if not text:
        return text
    
    # Eliminate multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Standardize line breaks
    text = re.sub(r'(\r\n|\r|\n)+', '\n', text)
    
    # Split into paragraphs keeping existing line breaks
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If we have less than 2 paragraphs and the text is long enough,
    # try to split it into shorter paragraphs
    if len(paragraphs) < 2 and len(text) > 500:
        # Split sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into paragraphs (approx. 3-5 sentences per paragraph)
        new_paragraphs = []
        current_paragraph = []
        sentence_count = 0
        sentences_per_paragraph = random.randint(3, 5)
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            sentence_count += 1
            
            if sentence_count >= sentences_per_paragraph:
                new_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
                sentence_count = 0
                sentences_per_paragraph = random.randint(3, 5)  # Vary paragraph length
        
        # Add the last paragraph if sentences remain
        if current_paragraph:
            new_paragraphs.append(' '.join(current_paragraph))
        
        paragraphs = new_paragraphs
    
    # Join paragraphs with double line break
    formatted_text = '\n\n'.join(paragraphs)
    
    return formatted_text

def combine_short_sentences(text: str) -> str:
    """
    Combines some short sentences to create longer, more fluid sentences.
    
    Args:
        text: The text to modify
        
    Returns:
        The text with some short sentences combined
    """
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 2:
        return text
    
    result_sentences = []
    i = 0
    
    while i < len(sentences) - 1:
        current = sentences[i]
        next_sentence = sentences[i + 1]
        
        # If both sentences are short, consider combining them
        if 10 <= len(current) <= 60 and 10 <= len(next_sentence) <= 60:
            # 40% chance to combine (not all short sentences)
            if random.random() < 0.4:
                connector = random.choice(SENTENCE_CONNECTORS)
                combined = current.rstrip('.!?') + connector + ' ' + next_sentence[0].lower() + next_sentence[1:]
                result_sentences.append(combined)
                i += 2
                continue
        
        result_sentences.append(current)
        i += 1
    
    # Add the last sentence if it hasn't been combined
    if i < len(sentences):
        result_sentences.append(sentences[i])
    
    # Join the sentences
    return ' '.join(result_sentences)

def enhance_paragraph_structure(text: str) -> str:
    """
    Enhances paragraph structure by adding transitions and clarifications.
    
    Args:
        text: The text to enhance
        
    Returns:
        The text with enhanced paragraph structure
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # If we have only one paragraph, return it as is
    if len(paragraphs) <= 1:
        return text
    
    # Enhance each paragraph (except the first, to keep it natural)
    improved_paragraphs = [paragraphs[0]]
    
    for i in range(1, len(paragraphs)):
        paragraph = paragraphs[i]
        
        # Add transitions between paragraphs (50% chance)
        if random.random() < 0.5 and not paragraph.startswith(tuple(SECTION_TRANSITIONS)):
            transition = random.choice(SECTION_TRANSITIONS)
            paragraph = f"{transition} {paragraph}"
        
        improved_paragraphs.append(paragraph)
    
    # Join the paragraphs
    return '\n\n'.join(improved_paragraphs)

def add_clarifications(text: str) -> str:
    """
    Adds clarifications or examples within some paragraphs.
    
    Args:
        text: The text to enhance
        
    Returns:
        The text with added clarifications
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # If we have few paragraphs, do not modify
    if len(paragraphs) < 2:
        return text
    
    # For some long paragraphs, add a clarification
    improved_paragraphs = []
    
    for paragraph in paragraphs:
        if len(paragraph) > 200 and random.random() < 0.3:
            # Split the paragraph into approximately two parts
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            mid_point = len(sentences) // 2
            
            first_part = ' '.join(sentences[:mid_point])
            second_part = ' '.join(sentences[mid_point:])
            
            # Add a clarification
            clarification = random.choice(CLARIFICATION_CONNECTORS)
            
            improved_paragraph = f"{first_part} {clarification} {second_part}"
            improved_paragraphs.append(improved_paragraph)
        else:
            improved_paragraphs.append(paragraph)
    
    # Join the paragraphs
    return '\n\n'.join(improved_paragraphs)

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Processes response data to improve text structure.
    
    Args:
        data: The data to process
        hook: The hook called
        
    Returns:
        The data with formatted text
    """
    # Only process responses
    if hook != "process_response":
        return data
    
    try:
        # Check if text is present and of type str
        if "text" in data and isinstance(data["text"], str):
            original_text = data["text"]
            
            # IMMEDIATELY apply removal of markdown symbols and trailing periods
            cleaned_text = remove_markdown_symbols(original_text)
            cleaned_text = remove_ending_periods(cleaned_text)
            
            # Apply formatting transformations to the cleaned text
            formatted_text = format_paragraphs(cleaned_text)
            formatted_text = combine_short_sentences(formatted_text)
            formatted_text = enhance_paragraph_structure(formatted_text)
            formatted_text = add_clarifications(formatted_text)
            
            # Update the text
            data["text"] = formatted_text
            logger.info("Text reformatted with markdown symbol and trailing period removal")
        
        return data
    
    except Exception as e:
        logger.error(f"Error during text formatting: {str(e)}")
        return data

# For standalone tests
if __name__ == "__main__":
    test_text = "Here is a first short sentence. Here is a second short sentence. This paragraph is quite short. It only contains simple sentences. Artificial intelligence tends to produce this writing style. We want to improve that. We can make the text more fluid. The responses will then be more natural."
    
    print("===== ORIGINAL TEXT =====")
    print(test_text)
    
    formatted = format_paragraphs(test_text)
    print("\n===== PARAGRAPH FORMATTING =====")
    print(formatted)
    
    combined = combine_short_sentences(formatted)
    print("\n===== SHORT SENTENCE COMBINATION =====")
    print(combined)
    
    enhanced = enhance_paragraph_structure(combined)
    print("\n===== STRUCTURE ENHANCEMENT =====")
    print(enhanced)
    
    clarified = add_clarifications(enhanced)
    print("\n===== ADDING CLARIFICATIONS =====")
    print(clarified)
