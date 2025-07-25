#!/usr/bin/env python3
"""
Automatic Launcher: Searx + Application
Automatically launches Searx then starts python app.py
"""

import sys
import os
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AutoLauncher')

def main():
    """Automatically launches Searx then the application"""
    
    print("\n🚀 AUTOMATIC LAUNCHER - SEARX + APPLICATION")
    print("=" * 60)
    
    try:
        # Step 1: Start Searx automatically
        logger.info("🔧 Phase 1: Automatic Searx startup...")
        
        try:
            from searx_auto_starter import SearxAutoStarter
            auto_starter = SearxAutoStarter()
            searx_ready = auto_starter.start_complete_system()
            
            if searx_ready:
                logger.info("✅ Searx ready!")
            else:
                logger.info("⚠️ Searx in degraded mode")
                
        except Exception as e:
            logger.warning(f"⚠️ Searx Error: {e}")
            logger.info("🔄 Continuing without Searx")
        
        # Step 2: Launch the application
        logger.info("🌐 Phase 2: Launching the application...")
        
        print("\n" + "=" * 60)
        print("🎉 STARTING THE APPLICATION")
        print("=" * 60)
        
        # Import and launch the Flask application
        from app import app
        
        print("🌐 Web Interface: http://localhost:5000")
        print("🤖 artificial intelligence API GOOGLE GEMINI 2.0 FLASH with autonomous searches")
        print("📸 Visual analysis available")
        print("🔧 Intelligent port management")
        print("=" * 60)
        print("💡 To stop: Ctrl+C")
        print()
        
        # Launch the application
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Shutdown requested by user")
        
        # Clean up Searx
        try:
            from port_manager import get_port_manager
            pm = get_port_manager()
            pm.stop_all_searx_containers()
            logger.info("✅ Searx cleanup completed")
        except:
            pass
        
        print("👋 Application stopped cleanly!")
        
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
