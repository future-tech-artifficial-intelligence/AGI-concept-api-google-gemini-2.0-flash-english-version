from flask import Blueprint, request, jsonify, session
import logging
from timezone_synchronizer import get_timezone_synchronizer
import pytz

"""API Routes for managing timezones for artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""

# Logger configuration
logger = logging.getLogger(__name__)

# Create the blueprint
timezone_bp = Blueprint('timezone', __name__)

@timezone_bp.route('/api/timezone/set', methods=['POST'])
def set_user_timezone():
    """
    Sets the timezone for a user.
    """
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        
        data = request.get_json()
        if not data or 'timezone' not in data:
            return jsonify({'error': 'Timezone required'}), 400
        
        user_id = session['user_id']
        timezone = data['timezone']
        
        # Validate the timezone
        try:
            pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return jsonify({'error': 'Invalid timezone'}), 400
        
        # Configure the timezone
        tz_sync = get_timezone_synchronizer()
        success = tz_sync.set_user_timezone(user_id, timezone)
        
        if success:
            logger.info(f"Timezone configured via API for user {user_id}: {timezone}")
            return jsonify({
                'message': 'Timezone successfully configured',
                'timezone': timezone
            })
        else:
            return jsonify({'error': 'Error during configuration'}), 500
            
    except Exception as e:
        logger.error(f"Error setting timezone: {str(e)}")
        return jsonify({'error': 'Internal error'}), 500

@timezone_bp.route('/api/timezone/get', methods=['GET'])
def get_user_timezone():
    """
    Retrieves a user's timezone.
    """
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        
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
        logger.error(f"Error retrieving timezone: {str(e)}")
        return jsonify({'error': 'Internal error'}), 500

@timezone_bp.route('/api/timezone/verify', methods=['POST'])
def verify_conversation_timestamps():
    """
    Verifies and corrects conversation timestamps.
    """
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        
        user_id = session['user_id']
        tz_sync = get_timezone_synchronizer()
        
        report = tz_sync.verify_conversation_timestamps(user_id)
        
        logger.info(f"Timestamp verification performed for user {user_id}")
        return jsonify({
            'message': 'Verification complete',
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        return jsonify({'error': 'Internal error'}), 500

@timezone_bp.route('/api/timezone/available', methods=['GET'])
def get_available_timezones():
    """
    Returns the list of available timezones.
    """
    try:
        # Recommended main timezones
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
        
        # All available timezones
        all_timezones = sorted(pytz.all_timezones)
        
        return jsonify({
            'main_timezones': main_timezones,
            'all_timezones': all_timezones
        })
        
    except Exception as e:
        logger.error(f"Error retrieving timezones: {str(e)}")
        return jsonify({'error': 'Internal error'}), 500
