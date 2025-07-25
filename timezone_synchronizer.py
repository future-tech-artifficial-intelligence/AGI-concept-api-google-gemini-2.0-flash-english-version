"""
Timezone synchronization module for artificial intelligence API GOOGLE GEMINI 2.0 FLASH.
This module ensures correct and consistent timezone synchronization
across all system components.
"""

import logging
import pytz
import datetime
from typing import Dict, Optional, Any
from database import get_db_connection

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimezoneSynchronizer:
    """
    Class to manage timezone synchronization throughout the system.
    """

    def __init__(self):
        self.user_timezones_cache = {}
        self.default_timezone = "Europe/Paris"

    def set_user_timezone(self, user_id: int, timezone: str) -> bool:
        """
        Sets and saves a user's timezone.

        Args:
            user_id: User ID
            timezone: Timezone to set

        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            # Validate timezone
            pytz.timezone(timezone)

            # Update cache
            self.user_timezones_cache[user_id] = timezone

            # Save to database
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if user already exists in preferences
            cursor.execute("""
                SELECT id FROM user_preferences WHERE user_id = ?
            """, (user_id,))

            if cursor.fetchone():
                # Update existing timezone
                cursor.execute("""
                    UPDATE user_preferences
                    SET timezone = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (timezone, user_id))
            else:
                # Create new entry
                cursor.execute("""
                    INSERT INTO user_preferences (user_id, timezone, created_at, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (user_id, timezone))

            conn.commit()
            conn.close()

            logger.info(f"Timezone configured for user {user_id}: {timezone}")
            return True

        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Invalid timezone: {timezone}")
            return False
        except Exception as e:
            logger.error(f"Error configuring timezone: {str(e)}")
            return False

    def get_user_timezone(self, user_id: int) -> str:
        """
        Retrieves a user's timezone.

        Args:
            user_id: User ID

        Returns:
            User's timezone or default
        """
        # Check cache first
        if user_id in self.user_timezones_cache:
            timezone = self.user_timezones_cache[user_id]
            logger.debug(f"Timezone retrieved from cache for user {user_id}: {timezone}")
            return timezone

        # Retrieve from database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timezone FROM user_preferences WHERE user_id = ?
            """, (user_id,))

            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                timezone = result[0]
                # Cache it
                self.user_timezones_cache[user_id] = timezone
                logger.info(f"Timezone retrieved from DB for user {user_id}: {timezone}")
                return timezone

        except Exception as e:
            logger.error(f"Error retrieving timezone: {str(e)}")

        # Return default timezone
        logger.info(f"Using default timezone for user {user_id}: {self.default_timezone}")
        return self.default_timezone

    def get_user_current_time(self, user_id: int) -> datetime.datetime:
        """
        Gets the current time in the user's timezone.

        Args:
            user_id: User ID

        Returns:
            Current datetime in the user's timezone
        """
        timezone_str = self.get_user_timezone(user_id)

        try:
            tz = pytz.timezone(timezone_str)
            current_time = datetime.datetime.now(tz)

            logger.debug(f"Current time for user {user_id} ({timezone_str}): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return current_time

        except Exception as e:
            logger.error(f"Error retrieving time: {str(e)}")
            # Return default time
            tz = pytz.timezone(self.default_timezone)
            return datetime.datetime.now(tz)

    def format_time_for_user(self, user_id: int, dt: Optional[datetime.datetime] = None) -> str:
        """
        Formats time for user display.

        Args:
            user_id: User ID
            dt: Datetime to format (default: now)

        Returns:
            Formatted string with time in user's timezone
        """
        if dt is None:
            dt = self.get_user_current_time(user_id)

        # Convert to user's timezone if necessary
        user_timezone = self.get_user_timezone(user_id)

        if dt.tzinfo is None:
            # If no timezone, assume UTC and convert
            dt = pytz.utc.localize(dt)

        # Convert to user's timezone
        user_tz = pytz.timezone(user_timezone)
        dt_user = dt.astimezone(user_tz)

        return dt_user.strftime("%A %d %B %Y at %H:%M:%S (%Z)")

    def verify_conversation_timestamps(self, user_id: int) -> Dict[str, Any]:
        """
        Verifies and corrects conversation timestamps for a user.

        Args:
            user_id: User ID

        Returns:
            Verification report
        """
        report = {
            "user_id": user_id,
            "user_timezone": self.get_user_timezone(user_id),
            "corrections_made": 0,
            "conversations_checked": 0,
            "errors": []
        }

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Retrieve all user conversations
            cursor.execute("""
                SELECT id, timestamp, created_at FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
            """, (user_id,))

            conversations = cursor.fetchall()
            report["conversations_checked"] = len(conversations)

            for conv in conversations:
                conv_id, timestamp, created_at = conv

                # Check if timestamp needs correction
                try:
                    # Parse existing timestamp
                    if isinstance(timestamp, str):
                        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp

                    # Convert to user's timezone
                    user_tz = pytz.timezone(report["user_timezone"])
                    dt_corrected = dt.astimezone(user_tz)

                    # Update if necessary
                    new_timestamp = dt_corrected.isoformat()
                    if new_timestamp != timestamp:
                        cursor.execute("""
                            UPDATE conversations
                            SET timestamp = ?
                            WHERE id = ?
                        """, (new_timestamp, conv_id))
                        report["corrections_made"] += 1

                except Exception as e:
                    report["errors"].append(f"Error with conversation {conv_id}: {str(e)}")

            conn.commit()
            conn.close()

        except Exception as e:
            report["errors"].append(f"General error: {str(e)}")

        logger.info(f"Timestamp verification completed for user {user_id}: {report['corrections_made']} corrections made")
        return report

# Global instance
timezone_sync = TimezoneSynchronizer()

def get_timezone_synchronizer():
    """Returns the global instance of the timezone synchronizer."""
    return timezone_sync
