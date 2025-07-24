
"""
Routes API pour la gestion des fuseaux horaires.
"""

from flask import Blueprint, request, jsonify, session
import logging
from timezone_synchronizer import get_timezone_synchronizer
import pytz

# Configuration du logger
logger = logging.getLogger(__name__)

# Créer le blueprint
timezone_bp = Blueprint('timezone', __name__)

@timezone_bp.route('/api/timezone/set', methods=['POST'])
def set_user_timezone():
    """
    Définit le fuseau horaire d'un utilisateur.
    """
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Non connecté'}), 401
        
        data = request.get_json()
        if not data or 'timezone' not in data:
            return jsonify({'error': 'Fuseau horaire requis'}), 400
        
        user_id = session['user_id']
        timezone = data['timezone']
        
        # Valider le fuseau horaire
        try:
            pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return jsonify({'error': 'Fuseau horaire invalide'}), 400
        
        # Configurer le fuseau horaire
        tz_sync = get_timezone_synchronizer()
        success = tz_sync.set_user_timezone(user_id, timezone)
        
        if success:
            logger.info(f"Fuseau horaire configuré via API pour l'utilisateur {user_id}: {timezone}")
            return jsonify({
                'message': 'Fuseau horaire configuré avec succès',
                'timezone': timezone
            })
        else:
            return jsonify({'error': 'Erreur lors de la configuration'}), 500
            
    except Exception as e:
        logger.error(f"Erreur lors de la configuration du fuseau horaire: {str(e)}")
        return jsonify({'error': 'Erreur interne'}), 500

@timezone_bp.route('/api/timezone/get', methods=['GET'])
def get_user_timezone():
    """
    Récupère le fuseau horaire d'un utilisateur.
    """
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Non connecté'}), 401
        
        user_id = session['user_id']
        tz_sync = get_timezone_synchronizer()
        
        timezone = tz_sync.get_user_timezone(user_id)
        current_time = tz_sync.get_user_current_time(user_id)
        formatted_time = tz_sync.format_time_for_user(user_id)
        
        return jsonify({
            'timezone': timezone,
            'current_time': current_time.isoformat(),
            'formatted_time': formatted_time
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du fuseau horaire: {str(e)}")
        return jsonify({'error': 'Erreur interne'}), 500

@timezone_bp.route('/api/timezone/verify', methods=['POST'])
def verify_conversation_timestamps():
    """
    Vérifie et corrige les timestamps des conversations.
    """
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Non connecté'}), 401
        
        user_id = session['user_id']
        tz_sync = get_timezone_synchronizer()
        
        report = tz_sync.verify_conversation_timestamps(user_id)
        
        logger.info(f"Vérification des timestamps effectuée pour l'utilisateur {user_id}")
        return jsonify({
            'message': 'Vérification terminée',
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la vérification: {str(e)}")
        return jsonify({'error': 'Erreur interne'}), 500

@timezone_bp.route('/api/timezone/available', methods=['GET'])
def get_available_timezones():
    """
    Retourne la liste des fuseaux horaires disponibles.
    """
    try:
        # Fuseaux horaires principaux recommandés
        main_timezones = [
            'Europe/Paris',
            'Europe/London',
            'Europe/Berlin',
            'Europe/Madrid',
            'Europe/Rome',
            'America/New_York',
            'America/Los_Angeles',
            'America/Chicago',
            'Asia/Tokyo',
            'Asia/Shanghai',
            'Australia/Sydney',
            'UTC'
        ]
        
        # Tous les fuseaux horaires disponibles
        all_timezones = sorted(pytz.all_timezones)
        
        return jsonify({
            'main_timezones': main_timezones,
            'all_timezones': all_timezones
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des fuseaux horaires: {str(e)}")
        return jsonify({'error': 'Erreur interne'}), 500
