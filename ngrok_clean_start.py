#!/usr/bin/env python3
"""
Script pour arrêter toutes les sessions ngrok et relancer proprement
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
    ngrok_dir = os.path.join(os.path.expanduser("~"), ".ngrok")
    ngrok_exe = os.path.join(ngrok_dir, "ngrok.exe")
    
    if os.path.exists(ngrok_exe):
        print("✅ ngrok déjà installé")
        return ngrok_exe
    
    print("📦 Téléchargement de ngrok...")
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

def kill_all_ngrok():
    """Tuer tous les processus ngrok et nettoyer"""
    print("🧹 Nettoyage complet de ngrok...")
    
    # Méthode 1: Tuer via taskkill (Windows)
    try:
        result = subprocess.run(['taskkill', '/F', '/IM', 'ngrok.exe'], 
                              capture_output=True, text=True)
        if "SUCCÈS" in result.stdout or "SUCCESS" in result.stdout:
            print("✅ Processus ngrok arrêtés")
        else:
            print("ℹ️ Aucun processus ngrok trouvé")
    except:
        print("ℹ️ Commande taskkill échouée")
    
    # Méthode 2: Via pyngrok
    try:
        import pyngrok
        from pyngrok import ngrok
        ngrok.kill()
        print("✅ Sessions pyngrok fermées")
    except ImportError:
        print("ℹ️ pyngrok non installé")
    except Exception as e:
        print(f"ℹ️ Nettoyage pyngrok: {e}")
    
    # Attendre un peu
    time.sleep(2)
    print("✅ Nettoyage terminé")

def start_ngrok_fresh():
    """Démarrer ngrok proprement"""
    print("🚀 Démarrage de ngrok...")
    
    # Vérifier que Flask fonctionne
    import requests
    try:
        response = requests.get('http://localhost:5000', timeout=3)
        if response.status_code == 200:
            print("✅ Flask fonctionne sur le port 5000")
        else:
            print(f"⚠️ Flask répond avec le code: {response.status_code}")
    except Exception as e:
        print(f"❌ Flask ne fonctionne pas: {e}")
        print("💡 Lancez d'abord: python app.py")
        return False
    
    # Télécharger ngrok si nécessaire
    ngrok_path = download_ngrok()
    if not ngrok_path:
        print("❌ Impossible d'installer ngrok")
        return False
    
    # Installer pyngrok si nécessaire
    try:
        import pyngrok
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
    
    try:
        # Démarrer le tunnel
        print("🌐 Création du tunnel sur le port 5000...")
        tunnel = ngrok.connect(5000, "http")
        url = tunnel.public_url
        
        print(f"\n🎉 SUCCÈS ! Votre site est accessible sur :")
        print(f"🌍 {url}")
        print(f"📊 Interface ngrok : http://localhost:4040")
        print("\n💡 Gardez cette fenêtre ouverte pour maintenir le tunnel")
        print("🛑 Appuyez sur Ctrl+C pour arrêter\n")
        
        # Ouvrir le navigateur
        webbrowser.open(url)
        
        # Maintenir le tunnel
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Arrêt du tunnel...")
            ngrok.kill()
            
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        return False
    
    return True

def main():
    print("🔄 Script de redémarrage propre de ngrok")
    print("=" * 50)
    
    # Étape 1: Nettoyer
    kill_all_ngrok()
    
    # Étape 2: Redémarrer
    start_ngrok_fresh()

if __name__ == "__main__":
    main()
