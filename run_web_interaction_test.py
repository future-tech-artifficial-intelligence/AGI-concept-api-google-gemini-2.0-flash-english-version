"""
Quick launch script to test artificial intelligence API GOOGLE GEMINI 2.0 FLASH's web interaction capabilities
Usage: python run_web_interaction_test.py
"""

import asyncio
import sys
import os
from pathlib import Path

def main():
    print("🌐 Testing artificial intelligence API GOOGLE GEMINI 2.0 FLASH's Web Interaction Capabilities")
    print("=" * 60)
    
    # Check that we are in the correct directory
    current_dir = Path.cwd()
    required_files = [
        'test_gemini_web_interaction.py',
        'interactive_web_navigator.py',
        'gemini_api_adapter.py',
        'ai_api_config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("Make sure you are in the project root directory.")
        return 1
    
    print("✅ All required files are present")
    print("🚀 Launching web interaction test...")
    print()
    
    try:
        # Import and launch the test
        from test_gemini_web_interaction import main as test_main
        asyncio.run(test_main())
        
        print()
        print("🎉 Tests completed successfully!")
        print("📁 Consult the 'test_results_web_interaction' folder for detailed reports")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print("Check :")
        print("1. That your artificial intelligence API GOOGLE GEMINI 2.0 FLASH API key is configured in ai_api_config.json")
        print("2. That Chrome/Chromium is installed for Selenium")
        print("3. That all dependencies are installed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
