"""
Adaptateur pour le module de planification IA avancée
"""

MODULE_METADATA = {
    'name': 'planification_ia_avancee_adapter',
    'description': 'Adaptateur pour le système de planification avancée',
    'version': '0.1.0',
    'priority': 95,
    'hooks': ['process_request', 'process_response'],
    'dependencies': ['planification_ia_avancee'],
    'enabled': True
}

def process(data, hook):
    """
    Fonction principale d'adaptation

    Args:
        data (dict): Données à traiter
        hook (str): Type de hook

    Returns:
        dict: Données modifiées
    """
    if not isinstance(data, dict):
        return data

    try:
        # Import local pour éviter les erreurs circulaires
        from . import planification_ia_avancee

        # Déléguer le traitement au module principal
        return planification_ia_avancee.process(data, hook)
    except ImportError:
        # Si le module principal n'est pas disponible, retourner les données inchangées
        return data

def adapt_planning_request(data):
    """Adapte les requêtes de planification"""
    return data

def adapt_planning_response(data):
    """Adapte les réponses de planification"""
    return data