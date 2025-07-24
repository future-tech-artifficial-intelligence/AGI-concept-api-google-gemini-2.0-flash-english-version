
"""
Système de raisonnement causal pour améliorer la compréhension des relations cause-effet
"""

MODULE_METADATA = {
    'name': 'causal_reasoning_system',
    'description': 'Système de raisonnement causal pour analyser les relations cause-effet',
    'version': '0.1.0',
    'priority': 80,
    'hooks': ['process_request', 'process_response'],
    'dependencies': [],
    'enabled': True
}

def process(data, hook):
    """
    Fonction principale de traitement pour le raisonnement causal
    
    Args:
        data (dict): Données à traiter
        hook (str): Type de hook
    
    Returns:
        dict: Données modifiées
    """
    if not isinstance(data, dict):
        return data
    
    if hook == 'process_request':
        return analyze_causal_request(data)
    elif hook == 'process_response':
        return enhance_causal_response(data)
    
    return data

def analyze_causal_request(data):
    """Analyse les relations causales dans une requête"""
    text = data.get('text', '')
    
    # Détecter les questions causales
    causal_patterns = [
        'pourquoi', 'à cause de', 'en raison de', 'grâce à',
        'provoque', 'cause', 'entraîne', 'résulte en'
    ]
    
    if any(pattern in text.lower() for pattern in causal_patterns):
        causal_instruction = """
        
Analysez cette question en identifiant :
- Les causes potentielles
- Les effets observés
- Les mécanismes causaux
- Les facteurs intermédiaires
        """
        data['text'] = text + causal_instruction
    
    return data

def enhance_causal_response(data):
    """Améliore les explications causales dans les réponses"""
    return data
