#!/usr/bin/env python3
"""
Ultra-simplified script for ngrok with automatic download
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
    import os
    import zipfile
    import urllib.request
    
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

def main():
    print("🚀 Installing and starting ngrok...")
    
    # Download ngrok if necessary
    ngrok_path = download_ngrok()
    if not ngrok_path:
        print("❌ Could not install ngrok")
        return
    
    # Install pyngrok
    try:
        import pyngrok
        print("✅ pyngrok already installed")
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
    print("✅ Token configured")
    
    try:
        # Start the tunnel
        print("🚀 Starting ngrok tunnel on port 5000...")
        tunnel = ngrok.connect(5000, "http")
        url = tunnel.public_url
        
        print(f"\n🌍 YOUR SITE IS ACCESSIBLE AT: {url}")
        print(f"📊 ngrok interface: http://localhost:4040")
        print("💡 Press Ctrl+C to stop\n")
        
        # Open the browser
        webbrowser.open(url)
        
        # Maintain the tunnel
        input("Press Enter to stop the tunnel...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("👋 Stopping the tunnel...")
        ngrok.kill()

if __name__ == "__main__":
    main()
