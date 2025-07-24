#!/usr/bin/env python3
"""
Script simplifié pour lancer ngrok avec Flask
"""

import os
import sys
import time
import subprocess
import webbrowser
import requests
from ngrok_auth_config import get_auth_token

def install_pyngrok():
    """Installer pyngrok si pas disponible"""
    try:
        import pyngrok
    except ImportError:
        print("📦 Installation de pyngrok...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyngrok'])
        import pyngrok

def start_with_pyngrok():
    """Démarrer ngrok avec pyngrok (version Python)"""
    try:
        from pyngrok import ngrok, conf
        
        # Configuration du token
        token = get_auth_token()
        if token and token.strip():
            conf.get_default().auth_token = token.strip()
            print(f"✅ Token ngrok configuré")
        
        # Démarrer le tunnel
        print("🚀 Démarrage du tunnel ngrok...")
        tunnel = ngrok.connect(4004, "http")
        print(f"🌍 Votre site est accessible sur: {tunnel.public_url}")
        
        # Ouvrir le navigateur
        webbrowser.open(tunnel.public_url)
        
        # Afficher les informations
        print("📊 Interface d'administration ngrok: http://localhost:4040")
        print("💡 Appuyez sur Ctrl+C pour arrêter le tunnel")
        
        # Maintenir le tunnel ouvert
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Arrêt du tunnel ngrok...")
            ngrok.kill()
            
    except Exception as e:
        print(f"❌ Erreur avec pyngrok: {str(e)}")
        return False
    return True

def start_with_command():
    """Démarrer ngrok avec la ligne de commande"""
    try:
        # Tester si ngrok est disponible
        result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ngrok n'est pas installé ou pas dans le PATH")
            return False
            
        print(f"✅ ngrok détecté: {result.stdout.strip()}")
        
        # Configurer le token
        token = get_auth_token()
        if token and token.strip():
            auth_result = subprocess.run(['ngrok', 'config', 'add-authtoken', token.strip()], 
                                       capture_output=True, text=True)
            if auth_result.returncode == 0:
                print("✅ Token configuré")
            else:
                print(f"⚠️ Avertissement token: {auth_result.stderr}")
        
        # Lancer ngrok
        print("🚀 Démarrage de ngrok...")
        process = subprocess.Popen(['ngrok', 'http', '5000'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Attendre que ngrok démarre
        time.sleep(3)
        
        # Récupérer l'URL
        try:
            response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
            tunnels = response.json()['tunnels']
            for tunnel in tunnels:
                if tunnel['proto'] == 'https':
                    url = tunnel['public_url']
                    print(f"🌍 Votre site est accessible sur: {url}")
                    webbrowser.open(url)
                    break
        except Exception as e:
            print(f"⚠️ Impossible de récupérer l'URL automatiquement: {e}")
            print("🌍 Vérifiez http://localhost:4040 pour voir votre URL")
        
        print("📊 Interface d'administration ngrok: http://localhost:4040")
        print("💡 Appuyez sur Ctrl+C pour arrêter")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n👋 Arrêt de ngrok...")
            process.terminate()
            process.wait()
            
        return True
        
    except FileNotFoundError:
        print("❌ ngrok n'est pas installé")
        return False
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return False

def main():
    """Fonction principale"""
    print("🚀 Démarrage de ngrok pour GeminiChat")
    print("🔍 Vérification que Flask fonctionne sur le port 5000...")
    
    # Vérifier que Flask fonctionne
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✅ Flask fonctionne correctement sur le port 5000")
        else:
            print(f"⚠️ Flask répond avec le code: {response.status_code}")
    except Exception as e:
        print(f"❌ Flask ne semble pas fonctionner sur le port 5000: {e}")
        print("💡 Lancez d'abord votre application Flask")
        return
    
    # Essayer avec la commande ngrok en premier
    print("\n🔄 Tentative avec ngrok en ligne de commande...")
    if start_with_command():
        return
    
    # Sinon essayer avec pyngrok
    print("\n🔄 Tentative avec pyngrok (version Python)...")
    install_pyngrok()
    if start_with_pyngrok():
        return
    
    print("\n❌ Impossible de démarrer ngrok")
    print("💡 Essayez d'installer ngrok depuis: https://ngrok.com/download")

if __name__ == "__main__":
    main()
