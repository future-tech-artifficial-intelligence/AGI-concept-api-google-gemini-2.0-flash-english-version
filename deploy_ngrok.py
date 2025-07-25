import os
import sys
import time
import signal
import logging
import subprocess
import threading
import webbrowser
import json
import tempfile
import requests
import atexit
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from importlib import reload
import app  # The main application

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ngrok-deployer')

# Global variables
ngrok_process = None
flask_process = None
observer = None
ngrok_url = None
ngrok_config_file = os.path.join(tempfile.gettempdir(), 'ngrok_config.yml')
current_directory = os.path.dirname(os.path.abspath(__file__))

# ngrok token - imported from the configuration file
from ngrok_auth_config import get_auth_token
NGROK_AUTH_TOKEN = get_auth_token()

# Class to manage file monitoring
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, restart_function):
        self.restart_function = restart_function
        self.last_modified = time.time()
        self.cooldown = 2  # Cooldown time in seconds

    def on_modified(self, event):
        # Ignore hidden files and directories
        if event.src_path.startswith(".") or not event.src_path.endswith(('.py', '.html', '.css', '.js')):
            return

        current_time = time.time()
        if current_time - self.last_modified > self.cooldown:
            self.last_modified = current_time
            logger.info(f"Change detected in {event.src_path}")
            self.restart_function()

def create_ngrok_config():
    """Creates a temporary ngrok configuration file"""
    import yaml

    config = {
        "version": "2",
        "authtoken": NGROK_AUTH_TOKEN,
        "web_addr": "localhost:4040",
        "tunnels": {
            "geminichat": {
                "proto": "http",
                "addr": "4004",
                "inspect": True
            }
        }
    }

    # Create the YAML configuration file
    config_file = os.path.join(tempfile.gettempdir(), 'ngrok_config.yml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_file

def setup_ngrok_auth():
    """Configures ngrok authentication with the token"""
    try:
        # Check if ngrok is already authenticated
        auth_process = subprocess.run(
            ['ngrok', 'config', 'check'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # If the token is not configured or is different, add it
        if b"authtoken" not in auth_process.stdout and b"authtoken" not in auth_process.stderr:
            logger.info("🔑 Configuring ngrok token...")
            auth_result = subprocess.run(
                ['ngrok', 'config', 'add-authtoken', NGROK_AUTH_TOKEN],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if auth_result.returncode == 0:
                logger.info("✅ ngrok token configured successfully")
            else:
                logger.error(f"❌ Error configuring ngrok token: {auth_result.stderr.decode('utf-8')}")
        else:
            logger.info("✅ ngrok is already authenticated")

    except Exception as e:
        logger.error(f"❌ Error configuring ngrok token: {str(e)}")

def start_ngrok():
    """Starts ngrok with optimized configuration"""
    global ngrok_process, ngrok_url

    # First, ensure authentication is properly configured
    setup_ngrok_auth()

    config_path = create_ngrok_config()

    try:
        # Check if ngrok is installed
        if sys.platform == 'win32':
            ngrok_cmd = ['ngrok', 'http', '--config', config_path, '4004']
        else:
            ngrok_cmd = ['ngrok', 'http', '--config', config_path, '4004']

        ngrok_process = subprocess.Popen(
            ngrok_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for ngrok to be ready and retrieve the URL
        time.sleep(3)  # Give ngrok time to start

        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            tunnels = response.json()['tunnels']
            for tunnel in tunnels:
                if tunnel['proto'] == 'https':
                    ngrok_url = tunnel['public_url']
                    break

            if ngrok_url:
                logger.info(f"🚀 Site successfully deployed to: {ngrok_url}")
                # Open the browser to the URL
                webbrowser.open(ngrok_url)
            else:
                logger.error("❌ Failed to retrieve ngrok URL")
        except Exception as e:
            logger.error(f"❌ Error retrieving ngrok URL: {str(e)}")

    except FileNotFoundError:
        logger.error("❌ ngrok is not installed or not in PATH")
        logger.info("📢 Install ngrok from https://ngrok.com/download and add it to your PATH")
        sys.exit(1)

def start_flask():
    """Starts the Flask server with optimizations"""
    global flask_process

    # Environment variables for optimizations
    env = os.environ.copy()
    env['FLASK_ENV'] = 'production'
    env['FLASK_APP'] = 'app.py'

    # Configure to use production mode but with reloading enabled
    flask_cmd = [sys.executable, '-m', 'flask', 'run', '--host=0.0.0.0', '--port=4004', '--reload']

    logger.info("🌐 Starting Flask server...")
    flask_process = subprocess.Popen(
        flask_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Thread to display Flask logs
    def log_output(pipe, prefix):
        for line in iter(pipe.readline, b''):
            try:
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                if decoded_line:  # Ignore empty lines
                    logger.info(f"{prefix} {decoded_line}")
            except Exception as e:
                logger.error(f"Error decoding Flask logs: {str(e)}")

    threading.Thread(target=log_output, args=(flask_process.stdout, "📊 Flask:"), daemon=True).start()
    threading.Thread(target=log_output, args=(flask_process.stderr, "⚠️ Flask:"), daemon=True).start()

def restart_server():
    """Restarts the Flask server"""
    global flask_process

    logger.info("🔄 Restarting Flask server...")

    # Stop the existing process
    if flask_process:
        if sys.platform == 'win32':
            # For Windows, use taskkill
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(flask_process.pid)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # For Unix
            os.kill(flask_process.pid, signal.SIGTERM)
            flask_process.wait()

    # Reload the application module
    try:
        reload(app)
        logger.info("📚 app.py module reloaded")
    except Exception as e:
        logger.error(f"❌ Error reloading module: {str(e)}")

    # Restart Flask
    start_flask()
    logger.info("✅ Flask server restarted")

def start_file_watcher():
    """Starts file monitoring for hot reloading"""
    global observer

    event_handler = FileChangeHandler(restart_server)
    observer = Observer()

    # Observe the main folder and subfolders
    paths_to_watch = [
        current_directory,
        os.path.join(current_directory, 'templates'),
        os.path.join(current_directory, 'static')
    ]

    for path in paths_to_watch:
        if os.path.exists(path):
            observer.schedule(event_handler, path, recursive=True)

    observer.start()
    logger.info("👁️ File monitoring activated for hot reloading")

def cleanup():
    """Cleans up processes on exit"""
    logger.info("🧹 Cleaning up processes...")

    if ngrok_process:
        ngrok_process.terminate()
        ngrok_process.wait()

    if flask_process:
        if sys.platform == 'win32':
            # For Windows
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(flask_process.pid)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # For Unix
            os.kill(flask_process.pid, signal.SIGTERM)
            flask_process.wait()

    if observer:
        observer.stop()
        observer.join()

    # Delete the temporary configuration file
    if os.path.exists(ngrok_config_file):
        try:
            os.remove(ngrok_config_file)
        except:
            pass

def check_requirements():
    """Checks and installs dependencies if necessary"""
    required_packages = [
        'flask', 'flask_compress', 'watchdog', 'requests', 'pyyaml'
    ]

    logger.info("🔍 Checking dependencies...")

    # Check installed packages
    try:
        import importlib.util
        missing = []

        for pkg in required_packages:
            # Special cases for certain packages
            module_name = pkg
            if pkg == 'pyyaml':
                module_name = 'yaml'
            elif pkg == 'flask_compress':
                module_name = 'flask_compress'

            spec = importlib.util.find_spec(module_name)
            if spec is None:
                missing.append(pkg)

        if missing:
            logger.info(f"📦 Installing missing packages: {', '.join(missing)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
            logger.info("✅ Packages installed successfully")
    except Exception as e:
        logger.error(f"❌ Error checking/installing packages: {str(e)}")
        # Continue even on error
        pass

def main():
    """Main function"""
    logger.info("🚀 Starting GeminiChat deployment with ngrok")

    # Check and install dependencies
    check_requirements()

    # Register cleanup function
    atexit.register(cleanup)

    try:
        # Start Flask
        start_flask()
        time.sleep(2)  # Wait for Flask to start

        # Start ngrok with authentication
        start_ngrok()

        # Start file monitoring
        start_file_watcher()

        logger.info("✅ System deployed and running")
        logger.info("🌐 ngrok administration interface: http://localhost:4040")

        # Show how to stop the server
        logger.info("💡 Press Ctrl+C to stop the server")

        # Keep the script running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("👋 Shutting down server...")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")

    finally:
        cleanup()

if __name__ == "__main__":
    main()
