#!/usr/bin/env python3
"""
Script to stop all ngrok sessions and restart cleanly
"""

import subprocess
import sys
import time
import webbrowser
import os
import zipfile
import urllib.request

def download_ngrok():
    """Download and install ngrok for Windows"""
    ngrok_dir = os.path.join(os.path.expanduser("~"), ".ngrok")
    ngrok_exe = os.path.join(ngrok_dir, "ngrok.exe")
    
    if os.path.exists(ngrok_exe):
        print("✅ ngrok already installed")
        return ngrok_exe
    
    print("📦 Downloading ngrok...")
    os.makedirs(ngrok_dir, exist_ok=True)
    
    # URL for Windows 64-bit
    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
    zip_path = os.path.join(ngrok_dir, "ngrok.zip")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete")
        
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ngrok_dir)
        
        os.remove(zip_path)
        print("✅ Installation complete")
        return ngrok_exe
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

def kill_all_ngrok():
    """Kill all ngrok processes and clean up"""
    print("🧹 Performing a clean ngrok shutdown...")
    
    # Method 1: Kill via taskkill (Windows)
    try:
        result = subprocess.run(['taskkill', '/F', '/IM', 'ngrok.exe'], 
                              capture_output=True, text=True)
        if "SUCCÈS" in result.stdout or "SUCCESS" in result.stdout:
            print("✅ ngrok processes stopped")
        else:
            print("ℹ️ No ngrok processes found")
    except:
        print("ℹ️ taskkill command failed")
    
    # Method 2: Via pyngrok
    try:
        import pyngrok
        from pyngrok import ngrok
        ngrok.kill()
        print("✅ pyngrok sessions closed")
    except ImportError:
        print("ℹ️ pyngrok not installed")
    except Exception as e:
        print(f"ℹ️ pyngrok cleanup error: {e}")
    
    # Wait a bit
    time.sleep(2)
    print("✅ Cleanup complete")

def start_ngrok_fresh():
    """Start ngrok cleanly"""
    print("🚀 Starting ngrok...")
    
    # Check if Flask is running
    import requests
    try:
        response = requests.get('http://localhost:5000', timeout=3)
        if response.status_code == 200:
            print("✅ Flask is running on port 5000")
        else:
            print(f"⚠️ Flask responded with code: {response.status_code}")
    except Exception as e:
        print(f"❌ Flask is not running: {e}")
        print("💡 First run: python app.py")
        return False
    
    # Download ngrok if necessary
    ngrok_path = download_ngrok()
    if not ngrok_path:
        print("❌ Could not install ngrok")
        return False
    
    # Install pyngrok if necessary
    try:
        import pyngrok
    except ImportError:
        print("📦 Installing pyngrok...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyngrok'])
        import pyngrok
    
    from pyngrok import ngrok, conf
    
    # Configure the path to ngrok
    conf.get_default().ngrok_path = ngrok_path
    
    # Token configuration
    token = "30EFEPCG8MXrlKyq8zHVJ3u1sPV_cv1vBoVKaaqNSEurn6Lf"
    conf.get_default().auth_token = token
    
    try:
        # Start the tunnel
        print("🌐 Creating tunnel on port 5000...")
        tunnel = ngrok.connect(5000, "http")
        url = tunnel.public_url
        
        print(f"\n🎉 SUCCESS! Your site is accessible at:")
        print(f"🌍 {url}")
        print(f"📊 ngrok interface: http://localhost:4040")
        print("\n💡 Keep this window open to maintain the tunnel")
        print("🛑 Press Ctrl+C to stop\n")
        
        # Open the browser
        webbrowser.open(url)
        
        # Maintain the tunnel
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Stopping the tunnel...")
            ngrok.kill()
            
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        return False
    
    return True

def main():
    print("🔄 ngrok clean restart script")
    print("=" * 50)
    
    # Step 1: Clean up
    kill_all_ngrok()
    
    # Step 2: Restart
    start_ngrok_fresh()

if __name__ == "__main__":
    main()
