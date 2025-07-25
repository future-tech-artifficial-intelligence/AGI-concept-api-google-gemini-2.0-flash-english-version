"""
Adapter for the advanced GOOGLE GEMINI 2.0 FLASH API artificial intelligence planning module
"""

MODULE_METADATA = {
    'name': 'planification_ia_avancee_adapter',
    'description': 'Adapter for the advanced planning system',
    'version': '0.1.0',
    'priority': 95,
    'hooks': ['process_request', 'process_response'],
    'dependencies': ['planification_ia_avancee'],
    'enabled': True
}

def process(data, hook):
    """
    Main adaptation function

    Args:
        data (dict): Data to process
        hook (str): Type of hook

    Returns:
        dict: Modified data
    """
    if not isinstance(data, dict):
        return data

    try:
        # Local import to avoid circular errors
        from . import planification_ia_avancee

        # Delegate processing to the main module
        return planification_ia_avancee.process(data, hook)
    except ImportError:
        # If the main module is not available, return unchanged data
        return data

def adapt_planning_request(data):
    """Adapts planning requests"""
    return data

def adapt_planning_response(data):
    """Adapts planning responses"""
    return data
