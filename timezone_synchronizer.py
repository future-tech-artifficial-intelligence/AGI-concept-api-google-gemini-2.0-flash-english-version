"""
Module de synchronisation des fuseaux horaires pour l'IA.
Ce module assure une synchronisation correcte et cohérente des fuseaux horaires
entre tous les composants du système.
"""

import logging
import pytz
import datetime
from typing import Dict, Optional, Any
from database import get_db_connection

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimezoneSynchronizer:
    """
    Classe pour gérer la synchronisation des fuseaux horaires dans tout le système.
    """

    def __init__(self):
        self.user_timezones_cache = {}
        self.default_timezone = "Europe/Paris"

    def set_user_timezone(self, user_id: int, timezone: str) -> bool:
        """
        Définit et sauvegarde le fuseau horaire d'un utilisateur.

        Args:
            user_id: ID de l'utilisateur
            timezone: Fuseau horaire à définir

        Returns:
            True si la configuration a réussi, False sinon
        """
        try:
            # Valider le fuseau horaire
            pytz.timezone(timezone)

            # Mettre à jour le cache
            self.user_timezones_cache[user_id] = timezone

            # Sauvegarder en base de données
            conn = get_db_connection()
            cursor = conn.cursor()

            # Vérifier si l'utilisateur existe déjà dans les préférences
            cursor.execute("""
                SELECT id FROM user_preferences WHERE user_id = ?
            """, (user_id,))

            if cursor.fetchone():
                # Mettre à jour le fuseau horaire existant
                cursor.execute("""
                    UPDATE user_preferences 
                    SET timezone = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (timezone, user_id))
            else:
                # Créer une nouvelle entrée
                cursor.execute("""
                    INSERT INTO user_preferences (user_id, timezone, created_at, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (user_id, timezone))

            conn.commit()
            conn.close()

            logger.info(f"Fuseau horaire configuré pour l'utilisateur {user_id}: {timezone}")
            return True

        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Fuseau horaire invalide: {timezone}")
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la configuration du fuseau horaire: {str(e)}")
            return False

    def get_user_timezone(self, user_id: int) -> str:
        """
        Récupère le fuseau horaire d'un utilisateur.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Fuseau horaire de l'utilisateur ou par défaut
        """
        # Vérifier le cache d'abord
        if user_id in self.user_timezones_cache:
            timezone = self.user_timezones_cache[user_id]
            logger.debug(f"Fuseau horaire récupéré du cache pour l'utilisateur {user_id}: {timezone}")
            return timezone

        # Récupérer depuis la base de données
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
                # Mettre en cache
                self.user_timezones_cache[user_id] = timezone
                logger.info(f"Fuseau horaire récupéré de la DB pour l'utilisateur {user_id}: {timezone}")
                return timezone

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du fuseau horaire: {str(e)}")

        # Retourner le fuseau horaire par défaut
        logger.info(f"Utilisation du fuseau horaire par défaut pour l'utilisateur {user_id}: {self.default_timezone}")
        return self.default_timezone

    def get_user_current_time(self, user_id: int) -> datetime.datetime:
        """
        Obtient l'heure actuelle dans le fuseau horaire de l'utilisateur.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Datetime actuel dans le fuseau horaire de l'utilisateur
        """
        timezone_str = self.get_user_timezone(user_id)

        try:
            tz = pytz.timezone(timezone_str)
            current_time = datetime.datetime.now(tz)

            logger.debug(f"Heure actuelle pour l'utilisateur {user_id} ({timezone_str}): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return current_time

        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'heure: {str(e)}")
            # Retourner l'heure par défaut
            tz = pytz.timezone(self.default_timezone)
            return datetime.datetime.now(tz)

    def format_time_for_user(self, user_id: int, dt: Optional[datetime.datetime] = None) -> str:
        """
        Formate l'heure pour l'affichage à l'utilisateur.

        Args:
            user_id: ID de l'utilisateur
            dt: Datetime à formater (par défaut: maintenant)

        Returns:
            Chaîne formatée avec l'heure dans le fuseau horaire de l'utilisateur
        """
        if dt is None:
            dt = self.get_user_current_time(user_id)

        # Convertir dans le fuseau horaire de l'utilisateur si nécessaire
        user_timezone = self.get_user_timezone(user_id)

        if dt.tzinfo is None:
            # Si pas de timezone, assumer UTC et convertir
            dt = pytz.utc.localize(dt)

        # Convertir dans le fuseau horaire de l'utilisateur
        user_tz = pytz.timezone(user_timezone)
        dt_user = dt.astimezone(user_tz)

        return dt_user.strftime("%A %d %B %Y à %H:%M:%S (%Z)")

    def verify_conversation_timestamps(self, user_id: int) -> Dict[str, Any]:
        """
        Vérifie et corrige les timestamps des conversations pour un utilisateur.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Rapport de vérification
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

            # Récupérer toutes les conversations de l'utilisateur
            cursor.execute("""
                SELECT id, timestamp, created_at FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
            """, (user_id,))

            conversations = cursor.fetchall()
            report["conversations_checked"] = len(conversations)

            for conv in conversations:
                conv_id, timestamp, created_at = conv

                # Vérifier si le timestamp nécessite une correction
                try:
                    # Parser le timestamp existant
                    if isinstance(timestamp, str):
                        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp

                    # Convertir dans le fuseau horaire de l'utilisateur
                    user_tz = pytz.timezone(report["user_timezone"])
                    dt_corrected = dt.astimezone(user_tz)

                    # Mettre à jour si nécessaire
                    new_timestamp = dt_corrected.isoformat()
                    if new_timestamp != timestamp:
                        cursor.execute("""
                            UPDATE conversations 
                            SET timestamp = ?
                            WHERE id = ?
                        """, (new_timestamp, conv_id))
                        report["corrections_made"] += 1

                except Exception as e:
                    report["errors"].append(f"Erreur avec la conversation {conv_id}: {str(e)}")

            conn.commit()
            conn.close()

        except Exception as e:
            report["errors"].append(f"Erreur générale: {str(e)}")

        logger.info(f"Vérification des timestamps terminée pour l'utilisateur {user_id}: {report['corrections_made']} corrections effectuées")
        return report

# Instance globale
timezone_sync = TimezoneSynchronizer()

def get_timezone_synchronizer():
    """Retourne l'instance globale du synchroniseur de fuseaux horaires."""
    return timezone_sync