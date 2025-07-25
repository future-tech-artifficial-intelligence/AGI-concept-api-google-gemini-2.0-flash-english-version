"""
Advanced planning improvement module for artificial intelligence API GOOGLE GEMINI 2.0 FLASH
"""

MODULE_METADATA = {
    'name': 'advanced_ai_planning',
    'description': 'Advanced planning system to improve the structuring of artificial intelligence API GOOGLE GEMINI 2.0 FLASH responses',
    'version': '0.1.0',
    'priority': 90,
    'hooks': ['process_request', 'process_response'],
    'dependencies': [],
    'enabled': True
}

def process(data, hook):
    """
    Main processing function for planning
    
    Args:
        data (dict): Data to be processed
        hook (str): Type of hook
    
    Returns:
        dict: Modified data
    """
    if not isinstance(data, dict):
        return data
    
    if hook == 'process_request':
        return plan_request_structure(data)
    elif hook == 'process_response':
        return structure_response(data)
    
    return data

def plan_request_structure(data):
    """Plans the processing structure of a request"""
    text = data.get('text', '')
    
    # Detect complex requests requiring planning
    complex_indicators = [
        'steps', 'plan', 'strategy', 'method', 'procedure',
        'how to', 'guide', 'tutorial'
    ]
    
    if any(indicator in text.lower() for indicator in complex_indicators):
        planning_instruction = """
        
Structure your response with:
1. A clear plan
2. Logical steps
3. Concrete examples
4. A final summary
        """
        data['text'] = text + planning_instruction
    
    return data

def structure_response(data):
    """Structures the response according to a logical plan"""
    return data
