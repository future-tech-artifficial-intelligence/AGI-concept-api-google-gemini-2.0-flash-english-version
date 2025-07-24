
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_compress import Compress
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
import uuid
import logging
from memory_engine import MemoryEngine  # Importer le gestionnaire de mémoire
import base64
import pathlib
from typing import Optional, Dict, Any, List
import shutil
from ai_api_manager import get_ai_api_manager  # Nouveau import pour le gestionnaire d'API

# Importer les blueprints pour la configuration des API
from api_config_routes import api_config_bp
from api_keys_routes import api_keys_bp
from timezone_api_routes import timezone_bp

# Configuration des logs - DOIT être défini AVANT les imports qui l'utilisent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('gemini-chat')

# Installation automatique des dépendances au démarrage
try:
    from auto_installer import AutoInstaller
    installer = AutoInstaller()
    
    # Vérifier s'il y a des modules manquants
    missing_report = installer.generate_missing_modules_report()
    if "❌" in missing_report:
        logger.info("🔧 Auto-installer détecté des modules manquants mais désactivé pour l'environnement Nix/Poetry")
        logger.info("Les modules sont gérés par pyproject.toml")
    else:
        logger.info("✅ Toutes les dépendances sont installées")
except Exception as e:
    logger.warning(f"⚠️ Impossible d'exécuter l'auto-installer: {str(e)}")
    logger.warning("Certains modules peuvent ne pas être disponibles")

# Vérifier la disponibilité d'aiohttp avant d'importer le web scraping
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("⚠️ Module aiohttp non disponible")

# Import du système de web scraping autonome (chargement différé pour accélérer le démarrage)
WEB_SCRAPING_AVAILABLE = False

def load_web_scraping_modules():
    """Charge les modules de web scraping de manière différée"""
    global WEB_SCRAPING_AVAILABLE
    try:
        if AIOHTTP_AVAILABLE:
            # Chargement en arrière-plan pour éviter les timeouts
            import threading
            def load_modules():
                try:
                    from autonomous_web_scraper import start_autonomous_web_learning, get_autonomous_learning_status
                    from web_learning_integration import trigger_autonomous_learning, force_web_learning_session
                    global WEB_SCRAPING_AVAILABLE
                    WEB_SCRAPING_AVAILABLE = True
                    logger.info("✅ Système de web scraping autonome chargé avec succès")
                except Exception as e:
                    logger.warning(f"⚠️ Chargement web scraping différé échoué: {e}")
            
            threading.Thread(target=load_modules, daemon=True).start()
            return True
        else:
            logger.warning("⚠️ Web scraping désactivé - module aiohttp manquant")
    except ImportError as e:
        logger.warning(f"⚠️ Web scraping non disponible: {e}")
    return False

# Configuration de base
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'gemini_chat_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 an en secondes pour mise en cache

# Configuration pour les uploads et mémorisation
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
CONVERSATIONS_DIR = os.path.join(DATA_DIR, 'conversations_text')
IMAGES_DIR = os.path.join(DATA_DIR, 'conversation_images')

# Enregistrer les blueprints des routes API
app.register_blueprint(api_config_bp)
app.register_blueprint(api_keys_bp)
app.register_blueprint(timezone_bp)

# Créer les répertoires s'ils n'existent pas
for directory in [DATA_DIR, UPLOADS_DIR, CONVERSATIONS_DIR, IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configurer la taille maximale des fichiers (10Mo)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
# Extensions autorisées pour les images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Optimisation - Compression des réponses
compress = Compress()
compress.init_app(app)

# Support des proxys comme ngrok
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Augmenter le niveau de log pour éviter d'afficher les messages
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Configuration de la base de données
DB_PATH = os.path.join(PROJECT_ROOT, 'gemini_chat.db')

def init_db():
    """Initialiser la base de données SQLite avec les tables nécessaires"""
    from database import init_db as db_init_db

    # Appeler la fonction init_db de database.py
    db_init_db()

    # Ensuite, initialiser les tables spécifiques à app.py si nécessaire
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table des utilisateurs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Table des sessions de conversation
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

    # Table des messages
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

# Initialiser la base de données au démarrage
init_db()

# Initialisation du moteur de mémoire (optimisée pour le déploiement)
memory_engine = None

def init_memory_engine():
    """Initialise le moteur de mémoire de manière différée"""
    global memory_engine
    if memory_engine is None:
        memory_engine = MemoryEngine()
        memory_engine.enable_text_memory(True)  # Activer la mémoire textuelle
        memory_engine.enable_upload_folder(True)  # Activer le dossier d'uploads
        logger.info("✅ Moteur de mémoire initialisé")
    return memory_engine

# Route de health check pour les déploiements
@app.route('/health')
def health_check():
    return {'status': 'ok', 'timestamp': datetime.now().isoformat()}, 200

# Routes principales
@app.route('/')
def index():
    return render_template('index-modern.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Utiliser la fonction validate_login de database.py
        from database import validate_login

        if validate_login(username, password):
            session['logged_in'] = True
            session['username'] = username
            flash('Connexion réussie !')
            return redirect(url_for('chat_page'))  # Redirection vers le chat au lieu de index
        else:
            flash('Nom d\'utilisateur ou mot de passe incorrect')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        email = request.form.get('email', username + '@example.com')  # Utiliser une valeur par défaut si non fourni

        if password != confirm_password:
            flash('Les mots de passe ne correspondent pas')
            return render_template('register.html')

        # Utiliser la fonction register_user de database.py
        from database import register_user

        if register_user(username, password, email):
            flash('Inscription réussie ! Vous pouvez maintenant vous connecter.')
            return redirect(url_for('login'))
        else:
            flash('Nom d\'utilisateur ou email déjà utilisé')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Vous avez été déconnecté')
    return redirect(url_for('index'))

@app.route('/chat')
def chat_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template('chat.html')

@app.route('/api-settings')
def api_settings():
    """Page de configuration des API d'IA."""
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
    image_data = data.get('image', None)  # Récupérer les données d'image si présentes

    # Log minimal pour éviter d'afficher le contenu des messages
    logger.info(f"Message reçu de l'utilisateur (longueur: {len(user_message)} caractères)")
    logger.info(f"Image présente: {'Oui' if image_data else 'Non'}")

    # Récupérer l'ID utilisateur (ou utiliser un ID par défaut)
    user_id = session.get('user_id', 1)

    # Gérer l'image si présente
    image_path = None
    if image_data:
        try:
            # S'assurer que le dossier d'uploads pour l'utilisateur existe
            user_upload_dir = os.path.join(UPLOADS_DIR, str(user_id))
            os.makedirs(user_upload_dir, exist_ok=True)
            logger.info(f"Dossier d'uploads créé/vérifié: {user_upload_dir}")

            # Générer un nom de fichier unique avec timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"upload_{timestamp}.jpg"

            # Sauvegarder l'image dans le dossier d'uploads via le TextMemoryManager
            # Utiliser directement le module TextMemoryManager pour éviter les problèmes potentiels
            from modules.text_memory_manager import TextMemoryManager

            image_path = TextMemoryManager.save_uploaded_image(user_id, image_data, filename)

            if image_path:
                logger.info(f"Image sauvegardée avec succès: {image_path}")

                # Vérifier si le fichier existe réellement
                full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)
                if os.path.exists(full_path):
                    logger.info(f"Le fichier image existe: {full_path}")
                else:
                    logger.error(f"Fichier image non trouvé: {full_path}")
            else:
                logger.warning("Échec de la sauvegarde de l'image: chemin non retourné")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'image: {str(e)}")

    # Utiliser le gestionnaire d'API pour obtenir une réponse
    api_manager = get_ai_api_manager()
    try:
        # Vérifier que l'image est dans un format acceptable si présente
        if image_data and not image_data.startswith('data:image/'):
            logger.warning("Format d'image incorrect, ajout du préfixe manquant")
            # Essayer d'ajouter un préfixe MIME si manquant
            image_data = 'data:image/jpeg;base64,' + image_data.split(',')[-1]

        # Passer l'ID utilisateur et l'ID de session pour accéder aux conversations précédentes
        api_result = api_manager.get_response(
            user_message, 
            image_data=image_data,
            user_id=user_id,
            session_id=session_id
        )
        ai_response = api_result['response']

        # En cas d'erreur dans la réponse, afficher un message approprié
        if api_result.get('status') == 'error':
            logger.error(f"Erreur API: {api_result.get('error', 'Erreur inconnue')}")
            ai_response = f"Désolé, une erreur s'est produite lors de la communication avec l'API. Veuillez réessayer ou contacter l'administrateur système."

        emotional_state = api_result.get('emotional_state', {'base_state': 'neutral', 'intensity': 0.5})
        api_used = api_manager.get_current_api_name()
    except Exception as e:
        logger.error(f"Exception lors de l'appel à l'API Gemini: {str(e)}")
        ai_response = "Désolé, une erreur s'est produite lors du traitement de votre message. Veuillez réessayer."
        emotional_state = {'base_state': 'concerned', 'intensity': 0.8}

    # Log minimal pour la réponse
    logger.info(f"Réponse générée (longueur: {len(ai_response)} caractères)")

    # Enregistrer la conversation en base de données
    conn = None
    try:
        # Ajouter un timeout et protection contre le verrouillage
        conn = sqlite3.connect(DB_PATH, timeout=20.0, isolation_level="EXCLUSIVE")
        cursor = conn.cursor()

        # Vérifier si la session existe déjà
        cursor.execute("SELECT 1 FROM conversation_sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            # Créer nouvelle session de conversation
            cursor.execute(
                "INSERT INTO conversation_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
                (session_id, user_id, f"Conversation du {datetime.now().strftime('%d/%m/%Y')}")
            )

        # Enregistrer le message de l'utilisateur
        cursor.execute(
            "INSERT INTO messages (session_id, message_type, content) VALUES (?, ?, ?)",
            (session_id, "user", user_message)
        )

        # Enregistrer la réponse de l'IA
        cursor.execute(
            "INSERT INTO messages (session_id, message_type, content) VALUES (?, ?, ?)",
            (session_id, "bot", ai_response)
        )

        # Mettre à jour le timestamp de dernière modification
        cursor.execute(
            "UPDATE conversation_sessions SET last_updated = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,)
        )

        conn.commit()
    except sqlite3.OperationalError as e:
        logger.error(f"Erreur SQLite lors de l'enregistrement de la conversation: {str(e)}")
        # Ne pas arrêter l'exécution, la réponse peut toujours être retournée
        # même si l'enregistrement échoue
    finally:
        if conn:
            conn.close()

    # Sauvegarder également dans le système de mémoire textuelle
    try:
        # Initialiser le moteur de mémoire si nécessaire
        mem_engine = init_memory_engine()
        
        # Message de l'utilisateur
        mem_engine.save_to_text_file(
            user_id=user_id,
            session_id=session_id,
            message_type="user",
            content=user_message,
            image_path=image_path,
            title=f"Conversation du {datetime.now().strftime('%d/%m/%Y')}"
        )

        # Réponse de l'assistant
        mem_engine.save_to_text_file(
            user_id=user_id,
            session_id=session_id,
            message_type="assistant",
            content=ai_response
        )
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des messages dans les fichiers texte: {str(e)}")
        # La conversation est déjà sauvegardée en base de données, donc continuer

    # Créer la réponse
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

        # Récupérer toutes les conversations de l'utilisateur
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
        logger.error(f"Erreur SQLite lors de la récupération des conversations: {str(e)}")
        return jsonify({'error': 'Erreur de base de données, veuillez réessayer'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Sert les fichiers téléchargés."""
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/conversation_images/<path:filename>')
def serve_conversation_image(filename):
    """Sert les images associées aux conversations."""
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Gère l'upload d'une image."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401

    # Vérifier qu'un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400

    file = request.files['file']

    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier invalide'}), 400

    # Vérifier que le fichier est une image autorisée
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format de fichier non autorisé'}), 400

    # Récupérer l'ID utilisateur
    user_id = session.get('user_id', 1)

    # Créer un nom de fichier sécurisé
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    safe_filename = f"{timestamp}_{filename}"

    # Chemin du fichier destination
    user_upload_dir = os.path.join(UPLOADS_DIR, str(user_id))
    os.makedirs(user_upload_dir, exist_ok=True)
    file_path = os.path.join(user_upload_dir, safe_filename)

    # Sauvegarder le fichier
    file.save(file_path)

    # Convertir l'image en base64 pour la mémoriser
    with open(file_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')

    # Enregistrer l'image avec le moteur de mémoire
    mem_engine = init_memory_engine()
    image_path = mem_engine.save_uploaded_image(user_id, img_data, safe_filename)

    # Retourner le résultat
    return jsonify({
        'success': True,
        'filename': safe_filename,
        'path': image_path
    })

@app.route('/api/api-key-test', methods=['POST'])
def test_api_key():
    """Teste une clé API"""
    try:
        data = request.get_json()
        api_type = data.get('api_type')
        api_key = data.get('api_key')

        if not api_type or not api_key:
            return jsonify({'success': False, 'error': 'Type d\'API et clé requis'})

        # Tester selon le type d'API
        if api_type == 'gemini':
            from gemini_api_adapter import test_gemini_api_key
            result = test_gemini_api_key(api_key)
        else:
            return jsonify({'success': False, 'error': f'Type d\'API {api_type} non supporté'})

        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur lors du test de clé API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trigger-web-search', methods=['POST'])
def trigger_web_search():
    """Déclenche une recherche web autonome"""
    try:
        # Charger les modules si pas encore fait
        if not WEB_SCRAPING_AVAILABLE:
            if not load_web_scraping_modules():
                return jsonify({'success': False, 'error': 'Web scraping non disponible'})

        data = request.get_json()
        query = data.get('query', 'intelligence artificielle')

        from web_learning_integration import force_web_learning_session
        result = force_web_learning_session()

        if result.get("forced") and result.get("session_result", {}).get("success"):
            session_result = result["session_result"]
            return jsonify({
                'success': True,
                'message': f'Recherche web effectuée avec succès',
                'pages_processed': session_result.get('pages_processed', 0),
                'files_created': len(session_result.get('files_created', [])),
                'domain_focus': session_result.get('domain_focus', query)
            })
        else:
            return jsonify({'success': False, 'error': 'Échec de la session de recherche web'})

    except Exception as e:
        logger.error(f"Erreur lors de la recherche web: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Vérifie que l'extension du fichier est autorisée
def allowed_file(filename):
    """Vérifie que l'extension du fichier est autorisée."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Optimisation - Cache statique pour ressources
@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        if request.path.startswith('/static'):
            # Fichiers statiques mis en cache longtemps
            response.headers['Cache-Control'] = 'public, max-age=31536000'
        else:
            # Pages dynamiques - pas de cache
            response.headers['Cache-Control'] = 'no-store'
    return response

# Point d'entrée principal
if __name__ == '__main__':
    # Utiliser le port pour Replit deployment
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Permettre l'accès externe

    # Configuration pour la production
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Afficher l'URL du serveur dans les logs
    logger.info(f"Démarrage du serveur sur http://{host}:{port}")
    
    # Démarrer le serveur avec configuration optimisée pour le déploiement
    app.run(
        host=host, 
        port=port, 
        debug=False, 
        threaded=True,
        use_reloader=False,
        use_debugger=False
    )
