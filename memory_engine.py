import os
import json
import sqlite3
import logging
import datetime
import base64
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pytz

from time_engine import should_remember_conversation, timestamp_to_readable_time_diff, MEMORY_RETENTION
from modules.text_memory_manager import TextMemoryManager  # New text memory management module

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('memory_engine')

# Paths for data storage
from database import DB_PATH
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'conversation_images')
os.makedirs(IMAGES_DIR, exist_ok=True)

class MemoryEngine:
    """
    Manages the Google Gemini 2.0 Flash AI's conversation memory, including text and images.
    """

    def __init__(self):
        """Initializes the memory engine."""
        self.db_path = DB_PATH
        self.setup_database()

        # Initialize the text memory manager
        self.text_memory_enabled = True  # Enable text file saving
        self.upload_folder_enabled = True  # Enable uploads folder

    def setup_database(self):
        """Configures the database to store conversations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table (sessions)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            session_id TEXT UNIQUE NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        # Message table in conversations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER NOT NULL,
            message_type TEXT NOT NULL,  -- 'user' or 'assistant'
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_image BOOLEAN DEFAULT FALSE,
            has_file BOOLEAN DEFAULT FALSE,
            emotional_state TEXT,
            metadata TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')

        # Table for images linked to messages
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_images (
            id INTEGER PRIMARY KEY,
            message_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            original_filename TEXT,
            thumbnail_path TEXT,
            content_type TEXT,
            description TEXT,
            metadata TEXT,
            FOREIGN KEY (message_id) REFERENCES conversation_messages (id)
        )
        ''')

        # Table for files linked to messages
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_files (
            id INTEGER PRIMARY KEY,
            message_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            FOREIGN KEY (message_id) REFERENCES conversation_messages (id),
            FOREIGN KEY (file_id) REFERENCES uploaded_files (id)
        )
        ''')

        # Table for short-term memory (recent conversations)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
            memory_level TEXT DEFAULT 'MEDIUM_TERM',
            metadata TEXT
        )
        ''')

        # Table for long-term memory (important facts, preferences, etc.)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Indexes to improve query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation ON conversation_messages (conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_message ON message_images (message_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_message ON message_files (message_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_user_id ON conversation_memory (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_session ON conversation_memory (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ltm_user_id ON long_term_memory (user_id)')

        conn.commit()
        conn.close()

    def create_conversation(self, user_id: int, title: Optional[str] = None) -> str:
        """
        Creates a new conversation for a user.

        Args:
            user_id: User ID
            title: Optional conversation title

        Returns:
            Unique session ID for the conversation
        """
        session_id = str(uuid.uuid4())
        title = title or f"Conversation on {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'INSERT INTO conversations (user_id, session_id, title) VALUES (?, ?, ?)',
            (user_id, session_id, title)
        )

        conn.commit()
        conn.close()

        logger.info(f"Created new conversation for user {user_id}, session_id: {session_id}")
        return session_id

    def get_or_create_conversation(self, user_id: int, session_id: Optional[str] = None) -> str:
        """
        Retrieves an existing conversation or creates a new one.

        Args:
            user_id: User ID
            session_id: Optional session ID

        Returns:
            Conversation session ID
        """
        if not session_id:
            return self.create_conversation(user_id)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if the conversation exists
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return session_id
        else:
            logger.warning(f"Session {session_id} not found for user {user_id}, creating new one")
            return self.create_conversation(user_id)

    def add_message(self, session_id: str, user_id: int, 
                   message_type: str, content: str,
                   image_data: Optional[str] = None, 
                   file_id: Optional[int] = None,
                   emotional_state: Optional[Dict] = None) -> int:
        """
        Adds a message to a conversation.

        Args:
            session_id: Conversation session ID
            user_id: User ID
            message_type: Message type ('user' or 'assistant')
            content: Text content of the message
            image_data: Base64 image data (optional)
            file_id: Associated file ID (optional)
            emotional_state: Assistant's emotional state (optional)

        Returns:
            ID of the created message
        """
        # Validate message type
        if message_type not in ['user', 'assistant']:
            raise ValueError(f"Invalid message type: {message_type}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Retrieve conversation ID
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )
        result = cursor.fetchone()

        if not result:
            conn.close()
            raise ValueError(f"Conversation not found: {session_id}")

        conversation_id = result[0]

        # Update last updated date
        cursor.execute(
            'UPDATE conversations SET last_updated = CURRENT_TIMESTAMP WHERE id = ?',
            (conversation_id,)
        )

        # Prepare message metadata
        has_image = bool(image_data)
        has_file = bool(file_id)

        # Prepare emotional state
        emotional_state_json = json.dumps(emotional_state) if emotional_state else None

        # Insert message
        cursor.execute(
            '''INSERT INTO conversation_messages 
               (conversation_id, message_type, content, has_image, has_file, emotional_state)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (conversation_id, message_type, content, has_image, has_file, emotional_state_json)
        )

        message_id = cursor.lastrowid

        # Process image if present
        if image_data:
            image_path = self._save_image(image_data, session_id, message_id)

            cursor.execute(
                '''INSERT INTO message_images 
                   (message_id, image_path, content_type)
                   VALUES (?, ?, ?)''',
                (message_id, image_path, 'image/jpeg')  # Assume it's a JPEG, adjust if needed
            )

        # Associate file if present
        if file_id:
            cursor.execute(
                'INSERT INTO message_files (message_id, file_id) VALUES (?, ?)',
                (message_id, file_id)
            )

        conn.commit()
        conn.close()

        # Parallel saving to text file
        if self.text_memory_enabled:
            # Retrieve title for the first save
            title = None
            if message_id == 1:  # First message of the conversation
                title_conn = sqlite3.connect(self.db_path)
                title_cursor = title_conn.cursor()
                title_cursor.execute(
                    'SELECT title FROM conversations WHERE session_id = ? AND user_id = ?',
                    (session_id, user_id)
                )
                result = title_cursor.fetchone()
                if result:
                    title = result[0]
                title_conn.close()

            # Retrieve image path if it was saved
            img_path = None
            if has_image and image_data:
                # Image already saved by _save_image, retrieve path
                img_conn = sqlite3.connect(self.db_path)
                img_cursor = img_conn.cursor()
                img_cursor.execute(
                    'SELECT image_path FROM message_images WHERE message_id = ?',
                    (message_id,)
                )
                img_result = img_cursor.fetchone()
                if img_result:
                    img_path = img_result[0]
                img_conn.close()

            # Save to text file
            self.save_to_text_file(
                user_id=user_id,
                session_id=session_id,
                message_type=message_type,
                content=content,
                image_path=img_path,
                title=title
            )

            # Also save to uploads folder if it's an image
            if has_image and image_data and self.upload_folder_enabled:
                upload_filename = f"{session_id}_{message_id}.jpg"
                self.save_uploaded_image(user_id, image_data, upload_filename)

        logger.info(f"Added {message_type} message to conversation {session_id}")
        return message_id

    def _save_image(self, image_data: str, session_id: str, message_id: int) -> str:
        """
        Saves an image to disk.

        Args:
            image_data: Image in base64 format
            session_id: Session ID
            message_id: Message ID

        Returns:
            Path of the saved image
        """
        # Create a folder for the session if necessary
        session_dir = os.path.join(IMAGES_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Generate a unique filename
        image_filename = f"msg_{message_id}_{int(datetime.datetime.now().timestamp())}.jpg"
        image_path = os.path.join(session_dir, image_filename)

        try:
            # Extract binary image data (remove the data:image/xxx;base64, prefix)
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]

            # Decode and save image
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_data))

            return os.path.relpath(image_path, os.path.dirname(os.path.abspath(__file__)))

        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return ""

    def get_conversation_history(self, session_id: str, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Retrieves conversation history.

        Args:
            session_id: Session ID
            user_id: User ID
            limit: Maximum number of messages to retrieve

        Returns:
            List of conversation messages
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # To get results as dictionaries
        cursor = conn.cursor()

        # Retrieve conversation ID
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )
        result = cursor.fetchone()

        if not result:
            conn.close()
            raise ValueError(f"Conversation not found: {session_id}")

        conversation_id = result['id']

        # Retrieve messages
        cursor.execute(
            '''SELECT id, message_type, content, timestamp, has_image, has_file, emotional_state
               FROM conversation_messages 
               WHERE conversation_id = ?
               ORDER BY timestamp DESC LIMIT ?''',
            (conversation_id, limit)
        )

        messages = []
        for row in cursor.fetchall():
            message = dict(row)

            # Transform emotional state into dictionary
            if message['emotional_state']:
                try:
                    message['emotional_state'] = json.loads(message['emotional_state'])
                except:
                    message['emotional_state'] = None

            # Retrieve image if present
            if message['has_image']:
                cursor.execute(
                    'SELECT image_path FROM message_images WHERE message_id = ?',
                    (message['id'],)
                )
                image_result = cursor.fetchone()
                if image_result:
                    message['image_path'] = image_result['image_path']

            # Retrieve file information if present
            if message['has_file']:
                cursor.execute(
                    '''SELECT f.id, f.original_filename, f.file_type, f.file_size
                       FROM message_files mf
                       JOIN uploaded_files f ON mf.file_id = f.id
                       WHERE mf.message_id = ?''',
                    (message['id'],)
                )
                file_result = cursor.fetchone()
                if file_result:
                    message['file'] = dict(file_result)

            messages.append(message)

        conn.close()

        # Reverse to get chronological order
        return messages[::-1]

    def get_user_conversations(self, user_id: int, limit: int = 20) -> List[Dict]:
        """
        Retrieves a user's conversation list.

        Args:
            user_id: User ID
            limit: Maximum number of conversations to retrieve

        Returns:
            List of conversations
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            '''SELECT id, session_id, title, created_at, last_updated
               FROM conversations
               WHERE user_id = ?
               ORDER BY last_updated DESC LIMIT ?''',
            (user_id, limit)
        )

        conversations = [dict(row) for row in cursor.fetchall()]

        # Retrieve the last message for each conversation
        for conversation in conversations:
            cursor.execute(
                '''SELECT content, message_type
                   FROM conversation_messages
                   WHERE conversation_id = ?
                   ORDER BY timestamp DESC LIMIT 1''',
                (conversation['id'],)
            )
            last_message = cursor.fetchone()
            if last_message:
                conversation['last_message'] = dict(last_message)

            # Count the number of messages
            cursor.execute(
                'SELECT COUNT(*) as count FROM conversation_messages WHERE conversation_id = ?',
                (conversation['id'],)
            )
            count_result = cursor.fetchone()
            conversation['message_count'] = count_result['count'] if count_result else 0

        conn.close()
        return conversations

    def update_conversation_title(self, session_id: str, user_id: int, title: str) -> bool:
        """
        Updates a conversation's title.

        Args:
            session_id: Session ID
            user_id: User ID
            title: New title

        Returns:
            True if update was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE conversations SET title = ? WHERE session_id = ? AND user_id = ?',
            (title, session_id, user_id)
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        # Update title in text file if feature is enabled
        if success and self.text_memory_enabled:
            file_path = os.path.join(TextMemoryManager.get_user_dir(user_id), f"{session_id}.txt")
            if os.path.exists(file_path):
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Replace the first line (title)
                    if lines and lines[0].startswith('# '):
                        lines[0] = f"# {title}\n"

                        # Rewrite the file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                except Exception as e:
                    logger.error(f"Error updating title in text file: {e}")

        return success

    def delete_conversation(self, session_id: str, user_id: int) -> bool:
        """
        Deletes a conversation and all its messages.

        Args:
            session_id: Session ID
            user_id: User ID

        Returns:
            True if deletion was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Retrieve conversation ID
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )
        result = cursor.fetchone()

        if not result:
            conn.close()
            return False

        conversation_id = result[0]

        try:
            # Delete images linked to messages
            cursor.execute(
                '''SELECT mi.image_path 
                   FROM message_images mi
                   JOIN conversation_messages cm ON mi.message_id = cm.id
                   WHERE cm.conversation_id = ?''',
                (conversation_id,)
            )

            for (image_path,) in cursor.fetchall():
                try:
                    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                except Exception as e:
                    logger.warning(f"Failed to delete image {image_path}: {str(e)}")

            # Use a transaction to ensure integrity
            cursor.execute('BEGIN TRANSACTION')

            # Delete image references
            cursor.execute(
                '''DELETE FROM message_images 
                   WHERE message_id IN (
                       SELECT id FROM conversation_messages WHERE conversation_id = ?
                   )''',
                (conversation_id,)
            )

            # Delete file references
            cursor.execute(
                '''DELETE FROM message_files 
                   WHERE message_id IN (
                       SELECT id FROM conversation_messages WHERE conversation_id = ?
                   )''',
                (conversation_id,)
            )

            # Delete messages
            cursor.execute(
                'DELETE FROM conversation_messages WHERE conversation_id = ?',
                (conversation_id,)
            )

            # Delete conversation
            cursor.execute(
                'DELETE FROM conversations WHERE id = ?',
                (conversation_id,)
            )

            cursor.execute('COMMIT')
            conn.close()

            # Clean up image folder if empty
            session_dir = os.path.join(IMAGES_DIR, session_id)
            if os.path.exists(session_dir) and not os.listdir(session_dir):
                os.rmdir(session_dir)

            # Also delete text file if feature is enabled
            if self.text_memory_enabled:
                self.delete_text_conversation(user_id, session_id)

            logger.info(f"Deleted conversation {session_id} for user {user_id}")
            return True

        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            logger.error(f"Error deleting conversation {session_id}: {str(e)}")
            return False

    def search_conversations(self, user_id: int, query: str, limit: int = 20) -> List[Dict]:
        """
        Searches a user's conversations.

        Args:
            user_id: User ID
            query: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching conversations
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Search in titles and message content
        cursor.execute(
            '''SELECT DISTINCT c.id, c.session_id, c.title, c.created_at, c.last_updated
               FROM conversations c
               JOIN conversation_messages m ON c.id = m.conversation_id
               WHERE c.user_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
               ORDER BY c.last_updated DESC LIMIT ?''',
            (user_id, f'%{query}%', f'%{query}%', limit)
        )

        conversations = [dict(row) for row in cursor.fetchall()]

        # Retrieve relevant excerpts for each conversation
        for conversation in conversations:
            cursor.execute(
                '''SELECT id, content, message_type
                   FROM conversation_messages
                   WHERE conversation_id = ? AND content LIKE ?
                   ORDER BY timestamp DESC LIMIT 3''',
                (conversation['id'], f'%{query}%')
            )

            matching_messages = [dict(row) for row in cursor.fetchall()]
            conversation['matching_messages'] = matching_messages

            # Count total number of messages
            cursor.execute(
                'SELECT COUNT(*) as count FROM conversation_messages WHERE conversation_id = ?',
                (conversation['id'],)
            )
            count_result = cursor.fetchone()
            conversation['message_count'] = count_result['count'] if count_result else 0

        conn.close()
        return conversations

    def get_context_for_gemini(self, session_id: str, user_id: int, max_messages: int = 10) -> str:
        """
        Builds a context for the Google Gemini 2.0 Flash AI from conversation history.

        Args:
            session_id: Session ID
            user_id: User ID
            max_messages: Maximum number of messages to include

        Returns:
            Context formatted for the Google Gemini 2.0 Flash AI
        """
        messages = self.get_conversation_history(session_id, user_id, max_messages)

        if not messages:
            return ""

        context = ["Here is the context from previous messages in this conversation:"]

        for msg in messages:
            if msg['message_type'] == 'user':
                prefix = "User: "
            else:
                prefix = "Google Gemini 2.0 Flash AI: "

            # Add textual content
            text = msg['content'] or ""
            if text:
                context.append(f"{prefix}{text}")

            # Mention images and files but not their content
            if msg.get('image_path'):
                context.append(f"{prefix}[shared an image]")
            if msg.get('file'):
                context.append(f"{prefix}[shared a file: {msg['file']['original_filename']}]")

        return "\n\n".join(context)

    def cleanup_old_conversations(self, days_threshold: int = 30) -> int:
        """
        Cleans up conversations inactive for a certain period.

        Args:
            days_threshold: Number of days after which a conversation is considered old

        Returns:
            Number of deleted conversations
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find inactive conversations
        cursor.execute(
            '''SELECT session_id, user_id FROM conversations
               WHERE datetime(last_updated) < datetime('now', ?)''',
            (f'-{days_threshold} days',)
        )

        old_conversations = cursor.fetchall()
        conn.close()

        # Delete each conversation
        deleted_count = 0
        for session_id, user_id in old_conversations:
            if self.delete_conversation(session_id, user_id):
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old conversations")
        return deleted_count

    def store_conversation(self, session_id: str, user_id: int, content: str, 
                          memory_level: str = "MEDIUM_TERM",
                          metadata: Dict[str, Any] = None,
                          user_timezone: Optional[str] = None) -> int:
        """
        Stores a conversation in memory.

        Args:
            session_id: Conversation session ID
            user_id: User ID
            content: Content to memorize
            memory_level: Retention level (SHORT_TERM, MEDIUM_TERM, LONG_TERM, PERMANENT)
            metadata: Additional metadata in dictionary format
            user_timezone: User's timezone (optional)

        Returns:
            ID of the created record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Generate a unique ID for this conversation
            conversation_id = str(uuid.uuid4())

            # Determine timezone to use
            if user_timezone:
                try:
                    tz = pytz.timezone(user_timezone)
                    current_time = datetime.datetime.now(tz)
                except pytz.exceptions.UnknownTimeZoneError:
                    current_time = datetime.datetime.now()
                    user_timezone = "UTC"
            else:
                current_time = datetime.datetime.now()
                user_timezone = "UTC"

            timestamp = current_time.timestamp()

            # Convert metadata to JSON if present
            metadata_json = None
            if metadata:
                metadata_json = json.dumps(metadata)

            # Prepare full metadata with detailed temporal information
            full_metadata = {
                'memory_level': memory_level,
                'timestamp': timestamp,
                'readable_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'date_complete': current_time.strftime('%A %d %B %Y'),
                'time_only': current_time.strftime('%H:%M:%S'),
                'timezone': user_timezone,
                'day_of_week': current_time.strftime('%A'),
                'month': current_time.strftime('%B'),
                'year': current_time.year,
                'content_length': len(content)
            }

            if metadata:
                full_metadata.update(metadata)

            cursor.execute('''
            INSERT INTO conversation_memory 
            (session_id, user_id, content, memory_level, metadata, timestamp) 
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (session_id, user_id, content, memory_level, json.dumps(full_metadata), timestamp))

            memory_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return memory_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing conversation in memory: {e}")
            return None

    def get_recent_conversations(self, 
                                 user_id: int, 
                                 session_id: Optional[str] = None,
                                 limit: int = 10,
                                 include_time_context: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves a user's recent conversations, considering
        the configured retention period.

        Args:
            user_id: User ID
            session_id: Optionally filter by specific session
            limit: Maximum number of conversations to retrieve
            include_time_context: Add information about elapsed time

        Returns:
            List of recent conversations
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # To access columns by name
        cursor = conn.cursor()

        # Build SQL query based on parameters
        sql = '''
        SELECT id, session_id, user_id, timestamp, content, memory_level, metadata
        FROM conversation_memory
        WHERE user_id = ?
        '''
        params = [user_id]

        if session_id:
            sql += ' AND session_id = ?'
            params.append(session_id)

        sql += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # Convert results to list of dictionaries
        result = []
        current_time = datetime.datetime.now(pytz.utc)

        for row in rows:
            # Convert timestamp string to datetime object
            timestamp_str = row['timestamp']
            if isinstance(timestamp_str, str):
                timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                timestamp = pytz.utc.localize(timestamp)
            else:
                # If it's already a datetime or another format
                timestamp = datetime.datetime.utcfromtimestamp(timestamp_str)
                timestamp = pytz.utc.localize(timestamp)

            # Check if this conversation should be remembered based on retention level
            memory_level = row['memory_level']
            if not should_remember_conversation(timestamp, memory_level):
                continue

            # Create entry for this conversation
            conversation = dict(row)

            # Deserialize JSON metadata
            if conversation['metadata']:
                try:
                    conversation['metadata'] = json.loads(conversation['metadata'])
                except:
                    conversation['metadata'] = {}

            # Add temporal context information if requested
            if include_time_context:
                conversation['time_ago'] = timestamp_to_readable_time_diff(timestamp, user_id)
                conversation['seconds_ago'] = (current_time - timestamp).total_seconds()

            result.append(conversation)

        conn.close()

        return result

    def store_text_memory(self, 
                          session_id: str, 
                          user_id: int, 
                          content: str, 
                          metadata: Dict[str, Any] = None) -> str:
        """
        Stores textual content in a file.

        Args:
            session_id: Session ID
            user_id: User ID
            content: Text content to store
            metadata: Additional metadata (optional)

        Returns:
            Path of the created text file
        """
        if not self.text_memory_enabled:
            raise RuntimeError("Text memory storage is disabled.")

        # Create a folder for the session if necessary
        session_dir = os.path.join(IMAGES_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Generate a unique filename
        file_name = f"memory_{user_id}_{int(datetime.datetime.now().timestamp())}.txt"
        file_path = os.path.join(session_dir, file_name)

        try:
            # Save content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Stored text memory for user {user_id} in file {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error storing text memory: {str(e)}")
            return ""

    def enable_text_memory(self, enabled: bool = True):
        """Enables or disables text memory."""
        self.text_memory_enabled = enabled

    def is_text_memory_enabled(self) -> bool:
        """Checks if text memory is enabled."""
        return self.text_memory_enabled

    def upload_file(self, user_id: int, file_data: bytes, file_name: str, metadata: Dict[str, Any] = None) -> int:
        """
        Uploads a file and associates it with a user.

        Args:
            user_id: User ID
            file_data: File data
            file_name: Original file name
            metadata: Additional metadata (optional)

        Returns:
            ID of the uploaded file
        """
        if not self.upload_folder_enabled:
            raise RuntimeError("File upload is disabled.")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert file into uploaded files table
        cursor.execute('''
        INSERT INTO uploaded_files (user_id, original_filename, file_type, file_size, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, file_name, 'text/plain', len(file_data), json.dumps(metadata) if metadata else None))

        file_id = cursor.lastrowid

        # Save file to disk
        user_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads', str(user_id))
        os.makedirs(user_dir, exist_ok=True)

        file_path = os.path.join(user_dir, f"{file_id}_{file_name}")
        with open(file_path, 'wb') as f:
            f.write(file_data)

        conn.commit()
        conn.close()

        logger.info(f"Uploaded file {file_name} for user {user_id}, file_id: {file_id}")
        return file_id

    def enable_upload_folder(self, enabled: bool = True):
        """Enables or disables the uploads folder."""
        self.upload_folder_enabled = enabled

    def is_upload_folder_enabled(self) -> bool:
        """Checks if the uploads folder is enabled."""
        return self.upload_folder_enabled

    def search_files(self, user_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for files uploaded by a user.

        Args:
            user_id: User ID
            query: Text to search for in metadata
            limit: Maximum number of results

        Returns:
            List of matching files
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            '''SELECT id, original_filename, file_type, file_size, created_at, metadata
               FROM uploaded_files
               WHERE user_id = ? AND (original_filename LIKE ? OR metadata LIKE ?)
               ORDER BY created_at DESC LIMIT ?''',
            (user_id, f'%{query}%', f'%{query}%', limit)
        )

        files = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return files

    def get_file_metadata(self, file_id: int) -> Dict[str, Any]:
        """
        Retrieves file metadata.

        Args:
            file_id: File ID

        Returns:
            Dictionary of file metadata
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            'SELECT id, original_filename, file_type, file_size, created_at, metadata FROM uploaded_files WHERE id = ?',
            (file_id,)
        )

        result = cursor.fetchone()
        conn.close()

        return dict(result) if result else {}

    def delete_file(self, file_id: int) -> bool:
        """
        Deletes an uploaded file.

        Args:
            file_id: File ID

        Returns:
            True if deletion was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Retrieve file information
        cursor.execute(
            'SELECT original_filename, user_id FROM uploaded_files WHERE id = ?',
            (file_id,)
        )
        result = cursor.fetchone()

        if not result:
            conn.close()
            return False

        original_filename, user_id = result

        try:
            # Delete file from disk
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads', str(user_id), f"{file_id}_{original_filename}")
            if os.path.exists(file_path):
                os.remove(file_path)

            # Delete record from database
            cursor.execute('DELETE FROM uploaded_files WHERE id = ?', (file_id,))
            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False
        finally:
            conn.close()

    def clear_uploads(self, user_id: int) -> int:
        """
        Deletes all files uploaded by a user.

        Args:
            user_id: User ID

        Returns:
            Number of files deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Retrieve user's files
        cursor.execute(
            'SELECT id, original_filename FROM uploaded_files WHERE user_id = ?',
            (user_id,)
        )
        files = cursor.fetchall()

        # Delete files from disk
        deleted_count = 0
        for file_id, original_filename in files:
            try:
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads', str(user_id), f"{file_id}_{original_filename}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {original_filename}: {str(e)}")

        # Delete records from database
        cursor.execute('DELETE FROM uploaded_files WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()

        logger.info(f"Cleared {deleted_count} uploaded files for user {user_id}")
        return deleted_count

    def search_memories(self, 
                       user_id: int, 
                       query: str, 
                       search_long_term: bool = True,
                       search_conversations: bool = True,
                       limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Searches a user's memories.

        Args:
            user_id: User ID
            query: Text to search for
            search_long_term: Search in long-term memories
            search_conversations: Search in conversations
            limit: Maximum number of results per category

        Returns:
            Dictionary with results by category
        """
        results = {}

        if search_conversations:
            # Search in conversations with a simple text search
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
            SELECT id, session_id, user_id, timestamp, content, memory_level, metadata
            FROM conversation_memory
            WHERE user_id = ? AND content LIKE ?
            ORDER BY timestamp DESC LIMIT ?
            ''', (user_id, f'%{query}%', limit))

            rows = cursor.fetchall()

            # Process results as in get_recent_conversations
            conversations = []
            current_time = datetime.datetime.now(pytz.utc)

            for row in rows:
                # Convert timestamp
                timestamp_str = row['timestamp']
                if isinstance(timestamp_str, str):
                    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    timestamp = pytz.utc.localize(timestamp)
                else:
                    timestamp = datetime.datetime.utcfromtimestamp(timestamp_str)
                    timestamp = pytz.utc.localize(timestamp)

                # Check retention
                memory_level = row['memory_level']
                if not should_remember_conversation(timestamp, memory_level):
                    continue

                conversation = dict(row)

                # Deserialize metadata
                if conversation['metadata']:
                    try:
                        conversation['metadata'] = json.loads(conversation['metadata'])
                    except:
                        conversation['metadata'] = {}

                conversation['time_ago'] = timestamp_to_readable_time_diff(timestamp, user_id)
                conversation['seconds_ago'] = (current_time - timestamp).total_seconds()

                conversations.append(conversation)

            conn.close()
            results['conversations'] = conversations

        if search_long_term:
            # Search in long-term memories
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
            SELECT id, user_id, category, content, importance, created_at, last_accessed
            FROM long_term_memory
            WHERE user_id = ? AND content LIKE ?
            ORDER BY importance DESC, last_accessed DESC LIMIT ?
            ''', (user_id, f'%{query}%', limit))

            rows = cursor.fetchall()

            # Update last accessed date
            if rows:
                memory_ids = [row['id'] for row in rows]
                placeholders = ','.join(['?'] * len(memory_ids))
                cursor.execute(f'''
                UPDATE long_term_memory
                SET last_accessed = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                ''', memory_ids)
                conn.commit()

            long_term = [dict(row) for row in rows]
            conn.close()
            results['long_term'] = long_term

        return results

    def forget_conversation(self, memory_id: int) -> bool:
        """
        Deletes a specific conversation from memory.

        Args:
            memory_id: ID of the record to delete

        Returns:
            True if deletion successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM conversation_memory WHERE id = ?', (memory_id,))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return success

    def get_memory_context(self, 
                           user_id: int, 
                           session_id: Optional[str] = None,
                           max_conversations: int = 5,
                           max_long_term: int = 3,
                           format_as_text: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Generates a memory context for use in Google Gemini 2.0 Flash AI interactions.

        Args:
            user_id: User ID
            session_id: Optional session ID to focus on a conversation
            max_conversations: Max number of recent conversations to include
            max_long_term: Max number of long-term memories to include
            format_as_text: Format the result as text or return a dictionary

        Returns:
            Memory context in text or dictionary format
        """
        # Retrieve recent conversations
        recent = self.get_recent_conversations(
            user_id=user_id,
            session_id=session_id,
            limit=max_conversations,
            include_time_context=True
        )

        # Retrieve most important long-term memories
        long_term = self.get_long_term_memories(
            user_id=user_id,
            limit=max_long_term
        )

        if format_as_text:
            # Format as text for the Google Gemini 2.0 Flash AI
            context = "Recent conversation memory:\n"

            if recent:
                for i, conv in enumerate(recent):
                    # Format entry with full temporal context
                    time_context = ""
                    if conv['metadata']:
                        try:
                            metadata = json.loads(conv['metadata'])
                        except (TypeError, json.JSONDecodeError):
                            metadata = {}

                        if 'readable_time' in metadata:
                            readable_time = metadata['readable_time']
                            time_ago = timestamp_to_readable_time_diff(metadata.get('timestamp', 0), user_id)
                            date_complete = metadata.get('date_complete', '')
                            timezone_info = metadata.get('timezone', 'UTC')

                            if date_complete:
                                time_context = f" ({time_ago} - {date_complete} at {metadata.get('time_only', '')} ({timezone_info}))"
                            else:
                                time_context = f" ({time_ago} - {readable_time})"
                    context += f"{i+1}. {conv['content']}{time_context}\n"
            else:
                context += "No recent conversations memorized.\n"

            context += "\nImportant information:\n"

            if long_term:
                for i, memory in enumerate(long_term):
                    context += f"{i+1}. {memory['category']}: {memory['content']}\n"
            else:
                context += "No important information memorized.\n"

            return context
        else:
            # Return a structured dictionary
            return {
                "recent_conversations": recent,
                "long_term_memories": long_term
            }

    def update_memory_importance(self, memory_id: int, importance: int) -> bool:
        """
        Updates the importance of a long-term memory.

        Args:
            memory_id: Memory ID
            importance: New importance value (1-10)

        Returns:
            True if update successful, False otherwise
        """
        importance = max(1, min(10, importance))  # Limit between 1 and 10

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        UPDATE long_term_memory
        SET importance = ?
        WHERE id = ?
        ''', (importance, memory_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def clear_expired_memories(self) -> int:
        """
        Deletes expired memories based on their retention level.

        Returns:
            Number of deleted memories
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate cutoff timestamps for each level
        now = datetime.datetime.now()
        short_term_cutoff = now - datetime.timedelta(minutes=MEMORY_RETENTION["SHORT_TERM"])
        medium_term_cutoff = now - datetime.timedelta(minutes=MEMORY_RETENTION["MEDIUM_TERM"])
        long_term_cutoff = now - datetime.timedelta(minutes=MEMORY_RETENTION["LONG_TERM"])

        # Format as SQLite string
        short_term_cutoff_str = short_term_cutoff.strftime('%Y-%m-%d %H:%M:%S')
        medium_term_cutoff_str = medium_term_cutoff.strftime('%Y-%m-%d %H:%M:%S')
        long_term_cutoff_str = long_term_cutoff.strftime('%Y-%m-%d %H:%M:%S')

        # Delete expired memories based on their level
        cursor.execute('''
        DELETE FROM conversation_memory 
        WHERE (memory_level = 'SHORT_TERM' AND timestamp < ?) 
           OR (memory_level = 'MEDIUM_TERM' AND timestamp < ?)
           OR (memory_level = 'LONG_TERM' AND timestamp < ?)
        ''', (short_term_cutoff_str, medium_term_cutoff_str, long_term_cutoff_str))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    def save_to_text_file(self, 
                      user_id: int,
                      session_id: str,
                      message_type: str,
                      content: str,
                      image_path: Optional[str] = None,
                      title: Optional[str] = None) -> bool:
        """
        Saves a message to a text file in addition to the database.

        Args:
            user_id: User ID
            session_id: Session ID
            message_type: Message type ('user' or 'assistant')
            content: Message content
            image_path: Image path (optional)
            title: Conversation title (optional)

        Returns:
            True if message was saved successfully
        """
        if not self.text_memory_enabled:
            return False

        return TextMemoryManager.save_message(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=content,
            image_path=image_path,
            title=title
        )

    def get_text_conversation(self, user_id: int, session_id: str) -> Optional[str]:
        """
        Retrieves a conversation from a text file.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Conversation content or None if file does not exist
        """
        if not self.text_memory_enabled:
            return None

        return TextMemoryManager.read_conversation(user_id, session_id)

    def list_text_conversations(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Lists all textual conversations for a user.

        Args:
            user_id: User ID

        Returns:
            List of user's textual conversations
        """
        if not self.text_memory_enabled:
            return []

        return TextMemoryManager.list_conversations(user_id)

    def search_text_conversations(self, user_id: int, query: str) -> List[Dict[str, Any]]:
        """
        Searches a user's textual conversations.

        Args:
            user_id: User ID
            query: Text to search for

        Returns:
            List of textual conversations containing the query
        """
        if not self.text_memory_enabled:
            return []

        return TextMemoryManager.search_conversations(user_id, query)

    def delete_text_conversation(self, user_id: int, session_id: str) -> bool:
        """
        Deletes a textual conversation.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            True if deletion was successful
        """
        if not self.text_memory_enabled:
            return False

        return TextMemoryManager.delete_conversation(user_id, session_id)

    def save_uploaded_image(self, user_id: int, image_data: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Saves an uploaded image.

        Args:
            user_id: User ID
            image_data: Base64 image data
            filename: File name (optional)

        Returns:
            Relative path of the saved image or None if error
        """
        if not self.upload_folder_enabled:
            return None

        return TextMemoryManager.save_uploaded_image(user_id, image_data, filename)

    def list_uploaded_images(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Lists images uploaded by a user.

        Args:
            user_id: User ID

        Returns:
            List of uploaded images
        """
        if not self.upload_folder_enabled:
            return []

        return TextMemoryManager.list_uploaded_images(user_id)
