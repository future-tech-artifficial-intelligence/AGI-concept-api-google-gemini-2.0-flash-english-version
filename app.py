import os
import socket
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_compress import Compress
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
import uuid
import logging
from memory_engine import MemoryEngine  # Import the memory manager
import base64
import pathlib
from typing import Optional, Dict, Any, List
import shutil
from ai_api_manager import get_ai_api_manager  # New import for the API manager

# Import blueprints for API configuration
from api_config_routes import api_config_bp
from api_keys_routes import api_keys_bp
from timezone_api_routes import timezone_bp

import logging
import sys
import os
from flask import Flask, render_template, request, jsonify, session
import uuid
from pathlib import Path

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intelligent Searx system initialization
# NOTE: This function is defined again later (line 144) and will be overridden.
# The actual active initialization logic will be from the later definition.
def initialize_searx_system():
    """Initializes the Searx system when the application starts"""
    try:
        logger.info("🚀 STARTING SEARX SYSTEM AUTOMATICALLY")
        logger.info("=" * 60)
        
        from searx_interface import get_searx_interface
        from port_manager import get_port_manager
        
        # Step 1: Check Docker
        logger.info("� Checking Docker...")
        import subprocess
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"✅ Docker available: {result.stdout.strip()}")
                
                # Check if Docker daemon is active
                result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    logger.warning("⚠️ Docker daemon not active - attempting to start...")
                    try:
                        # Try to start Docker Desktop
                        subprocess.Popen([
                            "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"
                        ], shell=True)
                        logger.info("🔄 Docker Desktop starting...")
                        
                        # Wait a bit
                        import time
                        time.sleep(10)
                    except Exception as e:
                        logger.warning(f"⚠️ Unable to start Docker automatically: {e}")
            else:
                logger.warning("⚠️ Docker not available")
        except Exception as e:
            logger.warning(f"⚠️ Docker check error: {e}")
        
        # Step 2: Initialize port manager
        logger.info("🔧 Initializing port manager...")
        port_manager = get_port_manager()
        
        # Step 3: Check if Searx already exists
        current_url = port_manager.get_current_searx_url()
        
        if current_url:
            logger.info(f"✅ Searx detected on: {current_url}")
            # Check if it's actually working
            searx_interface = get_searx_interface()
            if searx_interface.check_health():
                logger.info("✅ Existing Searx is operational")
                return True
            else:
                logger.info("� Searx detected but not functional - restarting...")
                port_manager.stop_all_searx_containers()
        
        # Step 4: Intelligent Searx startup
        logger.info("�🚀 Intelligent Searx startup...")
        success, url = port_manager.start_searx_smart()
        
        if success:
            logger.info(f"✅ Searx started successfully on: {url}")
            
            # Wait for Searx to be fully ready
            logger.info("⏳ Waiting for full initialization...")
            import time
            max_wait = 30  # 30 seconds max
            wait_time = 0
            
            searx_interface = get_searx_interface()
            while wait_time < max_wait:
                if searx_interface.check_health():
                    logger.info("✅ Searx fully operational!")
                    break
                time.sleep(2)
                wait_time += 2
                logger.info(f"⏳ Waiting... ({wait_time}/{max_wait}s)")
            
            if wait_time >= max_wait:
                logger.warning("⚠️ Searx takes longer than expected to start")
                return False
            
            # Quick search test
            logger.info("🧪 Quick system test...")
            try:
                results = searx_interface.search("test", max_results=1)
                if results:
                    logger.info("✅ Search test successful!")
                else:
                    logger.info("⚠️ Empty search test (normal on first start)")
            except Exception as e:
                logger.warning(f"⚠️ Search test failed: {e}")
            
            return True
        else:
            logger.warning("⚠️ Searx failed to start - degraded mode activated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Critical error during Searx initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

# Platform detection
try:
    from platform_detector import get_platform_detector
    platform_detector = get_platform_detector()
    IS_TERMUX = platform_detector.platform_info.get('is_termux', False)
    PLATFORM_TYPE = platform_detector.platform_info.get('platform_type', 'unknown')
    
    if IS_TERMUX:
        logger.info(f"🤖 Termux platform detected - Android mode activated")
        logger.info(platform_detector.get_platform_summary())
    else:
        logger.info(f"🖥️  {PLATFORM_TYPE} platform detected")
except ImportError:
    IS_TERMUX = 'TERMUX_VERSION' in os.environ
    PLATFORM_TYPE = 'termux' if IS_TERMUX else 'standard'
    logger.warning("⚠️ Platform detector not available, using basic detection")

# Automatic dependency installation on startup
try:
    from auto_installer import AutoInstaller
    installer = AutoInstaller()
    
    # Check for missing modules
    missing_report = installer.generate_missing_modules_report()
    if "❌" in missing_report:
        logger.info("🔧 Auto-installer detected missing modules but disabled for Nix/Poetry environment")
        logger.info("Modules are managed by pyproject.toml")
    else:
        logger.info("✅ All dependencies are installed")
except Exception as e:
    logger.warning(f"⚠️ Unable to run auto-installer: {str(e)}")
    logger.warning("Some modules may not be available")

# Check aiohttp availability before importing web scraping
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("⚠️ aiohttp module not available")

# Import autonomous web scraping system (deferred loading for faster startup)
WEB_SCRAPING_AVAILABLE = False
SEARX_AVAILABLE = False

def load_web_scraping_modules():
    """Loads web scraping modules in a deferred manner"""
    global WEB_SCRAPING_AVAILABLE
    try:
        if AIOHTTP_AVAILABLE:
            # Background loading to avoid timeouts
            import threading
            def load_modules():
                try:
                    from autonomous_web_scraper import start_autonomous_web_learning, get_autonomous_learning_status
                    from web_learning_integration import trigger_autonomous_learning, force_web_learning_session
                    global WEB_SCRAPING_AVAILABLE
                    WEB_SCRAPING_AVAILABLE = True
                    logger.info("✅ Autonomous web scraping system loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Deferred web scraping loading failed: {e}")
            
            threading.Thread(target=load_modules, daemon=True).start()
            return True
        else:
            logger.warning("⚠️ Web scraping disabled - aiohttp module missing")
    except ImportError as e:
        logger.warning(f"⚠️ Web scraping not available: {e}")
    return False

def initialize_searx_system(): # This definition overrides the previous one.
    """Initializes the Searx search system with visual capture"""
    global SEARX_AVAILABLE
    try:
        from searx_manager import initialize_searx
        import threading
        
        def init_searx():
            try:
                global SEARX_AVAILABLE
                SEARX_AVAILABLE = initialize_searx()
                if SEARX_AVAILABLE:
                    logger.info("✅ Searx search system initialized successfully")
                    
                    # Also initialize the visual capture system
                    try:
                        from searx_visual_capture import get_searx_visual_capture
                        visual_capture = get_searx_visual_capture()
                        logger.info("📸 Searx visual capture system initialized")
                    except ImportError:
                        logger.warning("⚠️ Visual capture not available (missing dependencies)")
                    except Exception as e:
                        logger.warning(f"⚠️ Visual capture not available: {e}")
                else:
                    logger.warning("⚠️ Searx initialization failed")
            except Exception as e:
                logger.error(f"❌ Error during Searx initialization: {e}")
        
        # Initialize Searx in the background
        threading.Thread(target=init_searx, daemon=True).start()
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Searx module not available: {e}")
        return False

# Base configuration
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'gemini_chat_secret_key') # Internal key name, not an API reference
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year in seconds for caching

# Initialize Searx system on startup
logger.info("🎯 STARTING APPLICATION WITH INTEGRATED SEARX")
logger.info("=" * 70)

searx_available = initialize_searx_system()

if searx_available:
    logger.info("� APPLICATION STARTED WITH FULL SEARX SYSTEM")
    logger.info("🔍 Autonomous searches enabled for AI")
    logger.info("📸 Visual analysis available")
else:
    logger.info("⚠️ APPLICATION STARTED IN DEGRADED MODE")
    logger.info("💡 Start Docker and restart to enable Searx")

logger.info("=" * 70)

# Configuration for uploads and memory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
CONVERSATIONS_DIR = os.path.join(DATA_DIR, 'conversations_text')
IMAGES_DIR = os.path.join(DATA_DIR, 'conversation_images')

# Register API routes blueprints
app.register_blueprint(api_config_bp)
app.register_blueprint(api_keys_bp)
app.register_blueprint(timezone_bp)

# Register advanced web navigation API
try:
    from web_navigation_api import register_web_navigation_api, initialize_web_navigation_api
    register_web_navigation_api(app)
    
    # Initialize the API with the Searx interface if available
    try:
        from searx_interface import get_searx_interface
        searx_interface = get_searx_interface()
        initialize_web_navigation_api(searx_interface)
        logger.info("✅ Advanced Web Navigation API registered with Searx")
    except:
        initialize_web_navigation_api(None)
        logger.info("✅ Advanced Web Navigation API registered without Searx")
        
except ImportError as e:
    logger.warning(f"⚠️ Web Navigation API not available: {str(e)}")
except Exception as e:
    logger.error(f"❌ Error during Web Navigation API registration: {str(e)}")

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOADS_DIR, CONVERSATIONS_DIR, IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure max file size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
# Allowed extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Optimization - Response compression
compress = Compress()
compress.init_app(app)

# Proxy support like ngrok
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Increase log level to avoid displaying messages
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gemini_chat.db') # Internal file name, not an API reference

def init_db():
    """Initializes the SQLite database with necessary tables"""
    from database import init_db as db_init_db

    # Call init_db function from database.py
    db_init_db()

    # Then, initialize app.py specific tables if necessary
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Conversation sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_sessions (
        session_id TEXT PRIMARY KEY,
        user_id INTEGER,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        message_type TEXT NOT NULL,
        content TEXT NOT NULL,
        emotional_state TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
    )
    ''')

    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Memory engine initialization (optimized for deployment)
memory_engine = None

def init_memory_engine():
    """Initializes the memory engine in a deferred manner"""
    global memory_engine
    if memory_engine is None:
        memory_engine = MemoryEngine()
        memory_engine.enable_text_memory(True)  # Enable text memory
        memory_engine.enable_upload_folder(True)  # Enable upload folder
        logger.info("✅ Memory engine initialized")
    return memory_engine

# Health check route for deployments
@app.route('/health')
def health_check():
    return {'status': 'ok', 'timestamp': datetime.now().isoformat()}, 200

# Main routes
@app.route('/')
def index():
    return render_template('index-modern.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Use the validate_login function from database.py
        from database import validate_login

        if validate_login(username, password):
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!')
            return redirect(url_for('chat_page'))  # Redirect to chat instead of index
        else:
            flash('Incorrect username or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        email = request.form.get('email', username + '@example.com')  # Use a default value if not provided

        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')

        # Use the register_user function from database.py
        from database import register_user

        if register_user(username, password, email):
            flash('Registration successful! You can now log in.')
            return redirect(url_for('login'))
        else:
            flash('Username or email already in use')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out')
    return redirect(url_for('index'))

@app.route('/chat')
def chat_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template('chat.html')

@app.route('/api-settings')
def api_settings():
    """AI API configuration page."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template('api_settings.html')

# API Routes
@app.route('/api/chat', methods=['POST'])
def chat_api():
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401

    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    image_data = data.get('image', None)  # Retrieve image data if present

    # Minimal log to avoid displaying message content
    logger.info(f"Message received from user (length: {len(user_message)} characters)")
    logger.info(f"Image present: {'Yes' if image_data else 'No'}")

    # Retrieve user ID (or use a default ID)
    user_id = session.get('user_id', 1)

    # Handle image if present
    image_path = None
    if image_data:
        try:
            # Ensure the user's upload folder exists
            user_upload_dir = os.path.join(UPLOADS_DIR, str(user_id))
            os.makedirs(user_upload_dir, exist_ok=True)
            logger.info(f"Upload folder created/checked: {user_upload_dir}")

            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"upload_{timestamp}.jpg"

            # Save the image to the uploads folder via the TextMemoryManager
            # Directly use the TextMemoryManager module to avoid potential issues
            from modules.text_memory_manager import TextMemoryManager

            image_path = TextMemoryManager.save_uploaded_image(user_id, image_data, filename)

            if image_path:
                logger.info(f"Image saved successfully: {image_path}")

                # Check if the file actually exists
                full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)
                if os.path.exists(full_path):
                    logger.info(f"Image file exists: {full_path}")
                else:
                    logger.error(f"Image file not found: {full_path}")
            else:
                logger.warning("Image save failed: path not returned")
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")

    # Use the API manager to get a response
    api_manager = get_ai_api_manager()
    try:
        # Check that the image is in an acceptable format if present
        if image_data and not image_data.startswith('data:image/'):
            logger.warning("Incorrect image format, adding missing prefix")
            # Try to add a MIME prefix if missing
            image_data = 'data:image/jpeg;base64,' + image_data.split(',')[-1]

        # Pass user ID and session ID to access previous conversations
        api_result = api_manager.get_response(
            user_message, 
            image_data=image_data,
            user_id=user_id,
            session_id=session_id
        )
        ai_response = api_result['response']

        # In case of an error in the response, display an appropriate message
        if api_result.get('status') == 'error':
            logger.error(f"API Error: {api_result.get('error', 'Unknown error')}")
            ai_response = f"Sorry, an error occurred while communicating with the API. Please try again or contact the system administrator."

        emotional_state = api_result.get('emotional_state', {'base_state': 'neutral', 'intensity': 0.5})
        api_used = api_manager.get_current_api_name()
    except Exception as e:
        logger.error(f"Exception during artificial intelligence API GOOGLE GEMINI 2.0 FLASH API call: {str(e)}")
        ai_response = "Sorry, an error occurred while processing your message. Please try again."
        emotional_state = {'base_state': 'concerned', 'intensity': 0.8}

    # Minimal log for the response
    logger.info(f"Response generated (length: {len(ai_response)} characters)")

    # Save conversation to database
    conn = None
    try:
        # Add timeout and locking protection
        conn = sqlite3.connect(DB_PATH, timeout=20.0, isolation_level="EXCLUSIVE")
        cursor = conn.cursor()

        # Check if session already exists
        cursor.execute("SELECT 1 FROM conversation_sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            # Create new conversation session
            cursor.execute(
                "INSERT INTO conversation_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
                (session_id, user_id, f"Conversation from {datetime.now().strftime('%d/%m/%Y')}")
            )

        # Save user message
        cursor.execute(
            "INSERT INTO messages (session_id, message_type, content) VALUES (?, ?, ?)",
            (session_id, "user", user_message)
        )

        # Save AI response
        cursor.execute(
            "INSERT INTO messages (session_id, message_type, content) VALUES (?, ?, ?)",
            (session_id, "bot", ai_response)
        )

        # Update last modified timestamp
        cursor.execute(
            "UPDATE conversation_sessions SET last_updated = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,)
        )

        conn.commit()
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite error saving conversation: {str(e)}")
        # Do not stop execution, response can still be returned
        # even if saving fails
    finally:
        if conn:
            conn.close()

    # Also save to the textual memory system
    try:
        # Initialize memory engine if necessary
        mem_engine = init_memory_engine()
        
        # User message
        mem_engine.save_to_text_file(
            user_id=user_id,
            session_id=session_id,
            message_type="user",
            content=user_message,
            image_path=image_path,
            title=f"Conversation from {datetime.now().strftime('%d/%m/%Y')}"
        )

        # Assistant response
        mem_engine.save_to_text_file(
            user_id=user_id,
            session_id=session_id,
            message_type="assistant",
            content=ai_response
        )
    except Exception as e:
        logger.error(f"Error saving messages to text files: {str(e)}")
        # Conversation is already saved in the database, so continue

    # Create response
    response = {
        'response': ai_response,
        'session_id': session_id,
        'emotional_state': emotional_state
    }

    return jsonify(response)

@app.route('/api/conversations')
def get_conversations():
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Retrieve all user conversations
        cursor.execute("""
            SELECT s.session_id, s.title, s.created_at, s.last_updated,
                COUNT(m.id) as message_count,
                (SELECT content FROM messages 
                    WHERE session_id = s.session_id 
                    ORDER BY created_at DESC LIMIT 1) as last_message
            FROM conversation_sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.last_updated DESC
        """)

        conversations = [dict(row) for row in cursor.fetchall()]
        return jsonify({'conversations': conversations})
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite error retrieving conversations: {str(e)}")
        return jsonify({'error': 'Database error, please try again'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serves uploaded files."""
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/conversation_images/<path:filename>')
def serve_conversation_image(filename):
    """Serves images associated with conversations."""
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Handles image upload."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401

    # Check that a file has been sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    file = request.files['file']

    # Check that the file has a name
    if file.filename == '':
        return jsonify({'error': 'Invalid file name'}), 400

    # Check that the file is an allowed image
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unauthorized file format'}), 400

    # Retrieve user ID
    user_id = session.get('user_id', 1)

    # Create a secure filename
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    safe_filename = f"{timestamp}_{filename}"

    # Destination file path
    user_upload_dir = os.path.join(UPLOADS_DIR, str(user_id))
    os.makedirs(user_upload_dir, exist_ok=True)
    file_path = os.path.join(user_upload_dir, safe_filename)

    # Save the file
    file.save(file_path)

    # Convert the image to base64 for memorization
    with open(file_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')

    # Save the image with the memory engine
    mem_engine = init_memory_engine()
    image_path = mem_engine.save_uploaded_image(user_id, img_data, safe_filename)

    # Return the result
    return jsonify({
        'success': True,
        'filename': safe_filename,
        'path': image_path
    })

@app.route('/api/api-key-test', methods=['POST'])
def test_api_key():
    """Tests an API key"""
    try:
        data = request.get_json()
        api_type = data.get('api_type')
        api_key = data.get('api_key')

        if not api_type or not api_key:
            return jsonify({'success': False, 'error': 'API type and key required'})

        # Test according to API type
        if api_type == 'gemini': # This is an internal type identifier, not the full public API name.
            from gemini_api_adapter import test_gemini_api_key
            result = test_gemini_api_key(api_key)
        else:
            return jsonify({'success': False, 'error': f'API type {api_type} not supported'})

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error testing API key: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trigger-web-search', methods=['POST'])
def trigger_web_search():
    """Triggers an autonomous web search"""
    try:
        # Load modules if not already done
        if not WEB_SCRAPING_AVAILABLE:
            if not load_web_scraping_modules():
                return jsonify({'success': False, 'error': 'Web scraping not available'})

        data = request.get_json()
        query = data.get('query', 'artificial intelligence')

        from web_learning_integration import force_web_learning_session
        result = force_web_learning_session()

        if result.get("forced") and result.get("session_result", {}).get("success"):
            session_result = result["session_result"]
            return jsonify({
                'success': True,
                'message': f'Web search performed successfully',
                'pages_processed': session_result.get('pages_processed', 0),
                'files_created': len(session_result.get('files_created', [])),
                'domain_focus': session_result.get('domain_focus', query)
            })
        else:
            return jsonify({'success': False, 'error': 'Web search session failed'})

    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Checks if the file extension is allowed
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_local_ip():
    """Gets the local IP address of the machine."""
    try:
        # Create a socket and connect to an external address to get the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        # In case of error, return localhost
        return "127.0.0.1"

# Optimization - Static resource caching
@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        if request.path.startswith('/static'):
            # Static files cached for a long time
            response.headers['Cache-Control'] = 'public, max-age=31536000'
        else:
            # Dynamic pages - no cache
            response.headers['Cache-Control'] = 'no-store'
    return response

# Main entry point
if __name__ == '__main__':
    # Use port for deployment
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Allow external access
    
    # Get local IP address for display
    local_ip = get_local_ip()

    # Production configuration
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Initialize background systems
    logger.info("🚀 Initializing advanced systems...")
    
    # Load web scraping
    load_web_scraping_modules()
    
    # Initialize Searx
    initialize_searx_system()
    
    # Display server URL in logs with local IP
    logger.info(f"Starting server on http://{local_ip}:{port}")
    logger.info(f"Local access: http://localhost:{port}")
    logger.info(f"Network access: http://{local_ip}:{port}")
    logger.info(f"Searx Interface: http://localhost:8080 (once initialized)")
    logger.info(f"📸 Visual capture: Enabled (if dependencies available)")
    logger.info(f"💡 AI test with vision: 'Search and show me information about...'")
    
    # Start server with optimized deployment configuration
    app.run(
        host=host, 
        port=port, 
        debug=False, 
        threaded=True,
        use_reloader=False,
        use_debugger=False
    )
