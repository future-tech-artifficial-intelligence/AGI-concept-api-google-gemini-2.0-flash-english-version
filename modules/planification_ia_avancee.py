
"""
Module de planification avancée pour l'IA
"""

MODULE_METADATA = {
    'name': 'planification_ia_avancee',
    'description': 'Système de planification avancée pour améliorer la structuration des réponses',
    'version': '0.1.0',
    'priority': 90,
    'hooks': ['process_request', 'process_response'],
    'dependencies': [],
    'enabled': True
}

def process(data, hook):
    """
    Fonction principale de traitement pour la planification
    
    Args:
        data (dict): Données à traiter
        hook (str): Type de hook
    
    Returns:
        dict: Données modifiées
    """
    if not isinstance(data, dict):
        return data
    
    if hook == 'process_request':
        return plan_request_structure(data)
    elif hook == 'process_response':
        return structure_response(data)
    
    return data

def plan_request_structure(data):
    """Planifie la structure de traitement d'une requête"""
    text = data.get('text', '')
    
    # Détecter les requêtes complexes nécessitant une planification
    complex_indicators = [
        'étapes', 'plan', 'stratégie', 'méthode', 'procédure',
        'comment faire', 'guide', 'tutorial'
    ]
    
    if any(indicator in text.lower() for indicator in complex_indicators):
        planning_instruction = """
        
Structurez votre réponse avec :
1. Un plan clair
2. Des étapes logiques
3. Des exemples concrets
4. Une synthèse finale
        """
        data['text'] = text + planning_instruction
    
    return data

def structure_response(data):
    """Structure la réponse selon un plan logique"""
    return data
