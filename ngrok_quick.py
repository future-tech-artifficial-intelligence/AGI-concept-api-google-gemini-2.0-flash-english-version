#!/usr/bin/env python3
"""
Script ultra-simplifié pour ngrok avec téléchargement automatique
"""

import subprocess
import sys
import time
import webbrowser
import os
import zipfile
import urllib.request

def download_ngrok():
    """Télécharger et installer ngrok pour Windows"""
    import os
    import zipfile
    import urllib.request
    
    ngrok_dir = os.path.join(os.path.expanduser("~"), ".ngrok")
    ngrok_exe = os.path.join(ngrok_dir, "ngrok.exe")
    
    if os.path.exists(ngrok_exe):
        print("✅ ngrok déjà installé")
        return ngrok_exe
    
    print("� Téléchargement de ngrok...")
    os.makedirs(ngrok_dir, exist_ok=True)
    
    # URL pour Windows 64-bit
    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
    zip_path = os.path.join(ngrok_dir, "ngrok.zip")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Téléchargement terminé")
        
        print("📦 Extraction...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ngrok_dir)
        
        os.remove(zip_path)
        print("✅ Installation terminée")
        return ngrok_exe
        
    except Exception as e:
        print(f"❌ Erreur de téléchargement: {e}")
        return None

def main():
    print("�🚀 Installation et démarrage de ngrok...")
    
    # Télécharger ngrok si nécessaire
    ngrok_path = download_ngrok()
    if not ngrok_path:
        print("❌ Impossible d'installer ngrok")
        return
    
    # Installer pyngrok
    try:
        import pyngrok
        print("✅ pyngrok déjà installé")
    except ImportError:
        print("📦 Installation de pyngrok...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyngrok'])
        import pyngrok
    
    from pyngrok import ngrok, conf
    
    # Configurer le chemin vers ngrok
    conf.get_default().ngrok_path = ngrok_path
    
    # Configuration du token
    token = "30EFEPCG8MXrlKyq8zHVJ3u1sPV_cv1vBoVKaaqNSEurn6Lf"
    conf.get_default().auth_token = token
    print("✅ Token configuré")
    
    try:
        # Démarrer le tunnel
        print("🚀 Démarrage du tunnel ngrok sur le port 5000...")
        tunnel = ngrok.connect(5000, "http")
        url = tunnel.public_url
        
        print(f"\n🌍 VOTRE SITE EST ACCESSIBLE SUR : {url}")
        print(f"📊 Interface ngrok : http://localhost:4040")
        print("💡 Appuyez sur Ctrl+C pour arrêter\n")
        
        # Ouvrir le navigateur
        webbrowser.open(url)
        
        # Maintenir le tunnel
        input("Appuyez sur Entrée pour arrêter le tunnel...")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
    finally:
        print("👋 Arrêt du tunnel...")
        ngrok.kill()

if __name__ == "__main__":
    main()
