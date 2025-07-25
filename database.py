import sqlite3
import hashlib
import os
import json
from datetime import datetime

DB_PATH = 'gemini_chat.db'

def get_db_connection():
    """Returns a connection to the database with a timeout"""
    return sqlite3.connect(DB_PATH, timeout=20.0)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create chat history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Create emotional state table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotional_state (
        id INTEGER PRIMARY KEY,
        base_state TEXT NOT NULL,
        joy REAL DEFAULT 0.5,
        sadness REAL DEFAULT 0.5,
        anger REAL DEFAULT 0.5,
        fear REAL DEFAULT 0.5,
        surprise REAL DEFAULT 0.5,
        disgust REAL DEFAULT 0.5,
        trust REAL DEFAULT 0.5,
        anticipation REAL DEFAULT 0.5,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create table for long-term memory (consciousness)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS consciousness_memory (
        id INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        knowledge TEXT NOT NULL,
        importance REAL DEFAULT 0.5,
        usage_count INTEGER DEFAULT 1,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create table for user interactions (context)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_interaction_context (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        context_data TEXT NOT NULL,
        session_id TEXT NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Create user preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            timezone TEXT DEFAULT 'Europe/Paris',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Check if the default emotional state already exists
    cursor.execute("SELECT COUNT(*) FROM emotional_state")
    if cursor.fetchone()[0] == 0:
        # Insert the default emotional state (neutral)
        cursor.execute('''
        INSERT INTO emotional_state (base_state, joy, sadness, anger, fear, surprise, disgust, trust, anticipation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('neutral', 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

    conn.commit()
    conn.close()

def hash_password(password):
    # Simple hash for the example - use bcrypt in production
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    conn = None
    try:
        # Add a 20-second timeout and EXCLUSIVE mode to handle blocking
        conn = sqlite3.connect(DB_PATH, timeout=20.0, isolation_level="EXCLUSIVE")
        cursor = conn.cursor()

        hashed_password = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, hashed_password, email)
        )

        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # User already exists
        return False
    except sqlite3.OperationalError as e:
        # Log error for debugging
        print(f"SQLite Operational Error: {e}")
        return False
    finally:
        # Ensure the connection is closed even if an error occurs
        if conn:
            conn.close()

def validate_login(username, password):
    """
    Validates a user's login credentials.

    Args:
        username: Username to check
        password: Password to check (unhashed)

    Returns:
        True if credentials are valid, False otherwise
    """
    conn = None
    try:
        # Add a timeout and read-only mode for this operation
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        cursor = conn.cursor()

        hashed_password = hash_password(password)
        cursor.execute(
            "SELECT id FROM users WHERE username = ? AND password = ?",
            (username, hashed_password)
        )

        user = cursor.fetchone()
        return user is not None
    except Exception as e:
        print(f"Error during login validation: {str(e)}")
        return False
    finally:
        # Ensure the connection is closed even if an error occurs
        if conn:
            conn.close()

def get_emotional_state():
    """Retrieves the AI's current emotional state"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM emotional_state ORDER BY last_updated DESC LIMIT 1")

    state = cursor.fetchone()
    conn.close()

    if state:
        return {
            'id': state[0],
            'base_state': state[1],
            'joy': state[2],
            'sadness': state[3],
            'anger': state[4],
            'fear': state[5],
            'surprise': state[6],
            'disgust': state[7],
            'trust': state[8],
            'anticipation': state[9],
            'last_updated': state[10]
        }
    return None

def update_emotional_state(emotions):
    """Updates the AI's emotional state"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Determine the base state from dominant emotions
    emotions_values = {
        'joy': emotions.get('joy', 0.5),
        'sadness': emotions.get('sadness', 0.5),
        'anger': emotions.get('anger', 0.5),
        'fear': emotions.get('fear', 0.5),
        'surprise': emotions.get('surprise', 0.5),
        'disgust': emotions.get('disgust', 0.5),
        'trust': emotions.get('trust', 0.5),
        'anticipation': emotions.get('anticipation', 0.5)
    }

    # Find the dominant emotion
    dominant_emotion = max(emotions_values, key=emotions_values.get)

    # Calculate the base state
    if emotions_values[dominant_emotion] > 0.7:
        base_state = dominant_emotion
    elif sum(emotions_values.values()) / len(emotions_values) < 0.4:
        base_state = 'calm'
    else:
        base_state = 'neutral'

    cursor.execute('''
    INSERT INTO emotional_state (base_state, joy, sadness, anger, fear, surprise, disgust, trust, anticipation)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        base_state,
        emotions_values['joy'],
        emotions_values['sadness'],
        emotions_values['anger'],
        emotions_values['fear'],
        emotions_values['surprise'],
        emotions_values['disgust'],
        emotions_values['trust'],
        emotions_values['anticipation']
    ))

    conn.commit()
    conn.close()

def store_memory(topic, knowledge, importance=0.5):
    """Stores information in the AI's memory"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if the topic already exists
    cursor.execute("SELECT id, knowledge, importance, usage_count FROM consciousness_memory WHERE topic = ?", (topic,))
    existing = cursor.fetchone()

    if existing:
        try:
            # Update existing knowledge (with robust error handling)
            try:
                if isinstance(existing[1], str):
                    merged_knowledge = json.loads(existing[1])
                else:
                    merged_knowledge = existing[1]

                # Ensure merged_knowledge is indeed a dictionary
                if not isinstance(merged_knowledge, dict):
                    merged_knowledge = {"data": merged_knowledge}
            except (json.JSONDecodeError, TypeError):
                merged_knowledge = {"error_recovery": True}

            # Prepare new_knowledge with additional protection
            try:
                if isinstance(knowledge, str):
                    try:
                        new_knowledge = json.loads(knowledge)
                    except (json.JSONDecodeError, TypeError):
                        new_knowledge = {"text": knowledge}
                else:
                    new_knowledge = knowledge

                # Ensure new_knowledge is indeed a dictionary
                if not isinstance(new_knowledge, dict):
                    new_knowledge = {"data": new_knowledge}
            except Exception:
                new_knowledge = {"error_recovery": True}

            # Merge knowledge with maximum protection
            for key, value in new_knowledge.items():
                if key in merged_knowledge:
                    # Triple check types before using update()
                    if isinstance(merged_knowledge[key], dict) and isinstance(value, dict):
                        # Both are dictionaries, secure merge
                        merged_knowledge[key] = merged_knowledge[key].copy()  # Copy to avoid in-place modifications
                        merged_knowledge[key].update(value)
                    else:
                        # If one of them is not a dictionary, simply replace
                        merged_knowledge[key] = value
                else:
                    merged_knowledge[key] = value
        except Exception as e:
            # In case of critical error, create a new object rather than failing
            import logging
            logging.error(f"Error merging knowledge: {str(e)}")
            merged_knowledge = {"error_recovery": True, "text": str(knowledge) if knowledge else ""}

        # Increase importance and usage count
        new_importance = min(1.0, existing[2] + 0.1)
        new_count = existing[3] + 1

        cursor.execute('''
        UPDATE consciousness_memory 
        SET knowledge = ?, importance = ?, usage_count = ?, last_accessed = CURRENT_TIMESTAMP 
        WHERE id = ?
        ''', (json.dumps(merged_knowledge), new_importance, new_count, existing[0]))
    else:
        # Create a new entry
        cursor.execute('''
        INSERT INTO consciousness_memory (topic, knowledge, importance)
        VALUES (?, ?, ?)
        ''', (topic, knowledge if isinstance(knowledge, str) else json.dumps(knowledge), importance))

    conn.commit()
    conn.close()

def get_memories(topic=None, min_importance=0.0, limit=5):
    """Retrieves memories from the AI's memory"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if topic:
        # Search by topic
        cursor.execute('''
        SELECT topic, knowledge, importance FROM consciousness_memory 
        WHERE topic LIKE ? AND importance >= ? 
        ORDER BY importance DESC, last_accessed DESC LIMIT ?
        ''', (f"%{topic}%", min_importance, limit))
    else:
        # Retrieve the most important memories
        cursor.execute('''
        SELECT topic, knowledge, importance FROM consciousness_memory 
        WHERE importance >= ? 
        ORDER BY importance DESC, last_accessed DESC LIMIT ?
        ''', (min_importance, limit))

    memories = cursor.fetchall()
    conn.close()

    return [{'topic': mem[0], 'knowledge': json.loads(mem[1]), 'importance': mem[2]} for mem in memories]

def update_user_context(user_id, context_data, session_id):
    """Updates or creates the user's interaction context"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if a context already exists for this user and session
    cursor.execute("SELECT id FROM user_interaction_context WHERE user_id = ? AND session_id = ?", 
                   (user_id, session_id))
    existing = cursor.fetchone()

    if existing:
        cursor.execute('''
        UPDATE user_interaction_context 
        SET context_data = ?, last_updated = CURRENT_TIMESTAMP 
        WHERE id = ?
        ''', (json.dumps(context_data), existing[0]))
    else:
        cursor.execute('''
        INSERT INTO user_interaction_context (user_id, context_data, session_id)
        VALUES (?, ?, ?)
        ''', (user_id, json.dumps(context_data), session_id))

    conn.commit()
    conn.close()

def get_user_context(user_id, session_id):
    """Retrieves the user's interaction context"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    SELECT context_data FROM user_interaction_context 
    WHERE user_id = ? AND session_id = ? 
    ORDER BY last_updated DESC LIMIT 1
    ''', (user_id, session_id))

    context = cursor.fetchone()
    conn.close()

    return json.loads(context[0]) if context else {}
