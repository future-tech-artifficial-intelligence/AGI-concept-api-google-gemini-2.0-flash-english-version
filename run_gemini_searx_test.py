"""
Launch script for artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx interaction tests
"""

import os
import sys
import time
import requests
from pathlib import Path

def check_prerequisites():
    """Checks that all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Verify that we are in the correct directory
    required_files = [
        'app.py',
        'searx_interface.py', 
        'interactive_web_navigator.py',
        'test_gemini_searx_interaction.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files are present")
    
    # Check that the Flask app is running
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ Flask application accessible on localhost:5000")
        else:
            print(f"⚠️  Flask application responds with code: {response.status_code}")
    except Exception as e:
        print(f"❌ Flask application not accessible: {str(e)}")
        print("💡 Make sure 'python app.py' is running")
        return False
    
    # Check Searx (optional as it can be integrated into the app)
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print("✅ Searx accessible on localhost:8080")
        else:
            print(f"⚠️  Searx responds with code: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Searx not accessible on localhost:8080: {str(e)}")
        print("💡 Searx can be integrated into the Flask application")
    
    return True

def main():
    """Main function"""
    print("🧪 Testing artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Interaction Capabilities")
    print("=" * 55)
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please:")
        print("1. Ensure you are in the correct directory")
        print("2. Start the application with: python app.py")
        print("3. Optionally start Searx")
        return
    
    # Waiting for services to stabilize
    print("\n⏳ Waiting for services to stabilize...")
    time.sleep(3)
    
    print("🚀 Launching artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx interaction test...\n")
    
    # Launch the test
    try:
        import subprocess
        result = subprocess.run([sys.executable, "test_gemini_searx_interaction.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Tests completed successfully!")
        else:
            print(f"\n⚠️  Tests completed with errors (code: {result.returncode})")
            
    except Exception as e:
        print(f"\n❌ Error launching tests: {str(e)}")
    
    print("📁 Consult the 'test_results_searx_interaction' folder for detailed reports")

if __name__ == "__main__":
    main()
