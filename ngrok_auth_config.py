"""
Configuration d'authentification ngrok pour GeminiChat
Ce fichier contient le token d'authentification ngrok pour éviter d'avoir à le saisir manuellement.
"""

# Token d'authentification ngrok
NGROK_AUTH_TOKEN = "30EFEPCG8MXrlKyq8zHVJ3u1sPV_cv1vBoVKaaqNSEurn6Lf"

def get_auth_token():
    """
    Retourne le token d'authentification ngrok
    """
    return NGROK_AUTH_TOKEN
