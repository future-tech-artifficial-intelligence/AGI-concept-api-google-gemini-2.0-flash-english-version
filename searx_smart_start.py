#!/usr/bin/env python3
"""
Searx Smart Start Script with Automatic Port Management
Automatically resolves port conflicts and configures Searx optimally
"""

import sys
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('searx_smart_start.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('SearxSmartStart')

def main():
    """Main smart start function"""
    
    print("🚀 SEARX SMART START")
    print("=" * 50)
    
    try:
        # Import necessary modules
        from port_manager import PortManager
        from searx_interface import SearxInterface
        
        # Initialize the port manager
        logger.info("🔧 Initializing port manager...")
        port_manager = PortManager()
        
        # Check current status
        current_url = port_manager.get_current_searx_url()
        if current_url:
            print(f"✅ Searx already active on: {current_url}")
            
            # Test connectivity
            searx = SearxInterface(current_url)
            if searx.check_health():
                print("🟢 Searx is running correctly")
                print(f"🌐 Interface accessible: {current_url}")
                return True
            else:
                print("⚠️  Searx detected but not functional, restarting...")
                port_manager.stop_all_searx_containers()
        
        # Smart start
        print("🔍 Analyzing available ports...")
        success, url = port_manager.start_searx_smart()
        
        if not success:
            print("❌ Smart start failed")
            print("\n🔧 Possible solutions:")
            print("1. Run: python free_port_8080.py")
            print("2. Use: docker-compose -f docker-compose.searx-alt.yml up -d")
            print("3. Restart your computer to free up all ports")
            return False
        
        # Final verification
        print(f"⏳ Verifying Searx on {url}...")
        searx = SearxInterface(url)
        
        # Wait for Searx to be ready
        max_attempts = 12
        for attempt in range(max_attempts):
            if searx.check_health():
                print(f"✅ Searx operational on: {url}")
                break
            
            print(f"⏳ Attempt {attempt + 1}/{max_attempts} - Waiting for initialization...")
            time.sleep(5)
        else:
            print("❌ Searx not responding after 60 seconds")
            return False
        
        # Search test
        print("\n🧪 Search test...")
        results = searx.search("test artificial intelligence", max_results=3)
        
        if results:
            print(f"✅ Test successful: {len(results)} results found")
            print(f"   First result: {results[0].title}")
        else:
            print("⚠️  Search test failed")
        
        # Final information
        print("\n" + "=" * 50)
        print("🎉 SEARX STARTED SUCCESSFULLY!")
        print(f"🌐 URL: {url}")
        print(f"📊 Management Interface: {url}")
        print("🔍 Ready for autonomous searches")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        print(f"❌ Unexpected error: {e}")
        return False

def show_status():
    """Displays the current status of Searx"""
    try:
        from port_manager import PortManager
        from searx_interface import SearxInterface
        
        pm = PortManager()
        
        print("📊 SEARX STATUS")
        print("=" * 30)
        
        # Check saved configuration
        config = pm.load_port_config()
        if config:
            print(f"📁 Configuration: {config['compose_file']}")
            print(f"🔌 Port: {config['searx_port']}")
            print(f"🌐 URL: {config['url']}")
            print(f"📅 Configured: {time.ctime(config['timestamp'])}")
        else:
            print("❌ No configuration saved")
        
        # Test current ports
        print("\n🔍 Port scan:")
        for port in [8080, 8081, 8082, 8083]:
            status = "🟢 Available" if pm.is_port_available(port) else "🔴 Occupied"
            print(f"   Port {port}: {status}")
            
            if not pm.is_port_available(port):
                process = pm.get_process_using_port(port)
                if process:
                    print(f"      → {process['name']} (PID: {process['pid']})")
        
        # Connectivity test
        current_url = pm.get_current_searx_url()
        if current_url:
            print(f"\n🌐 Connectivity test: {current_url}")
            searx = SearxInterface(current_url)
            if searx.check_health():
                print("✅ Searx is running correctly")
            else:
                print("❌ Searx is not responding")
        else:
            print("\n❌ No Searx instance detected")
            
    except Exception as e:
        print(f"❌ Error during check: {e}")

def stop_all():
    """Stops all Searx instances"""
    try:
        from port_manager import PortManager
        
        pm = PortManager()
        success = pm.stop_all_searx_containers()
        
        if success:
            print("✅ All Searx instances stopped")
        else:
            print("❌ Error during shutdown")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            show_status()
        elif command == "stop":
            stop_all()
        elif command == "start":
            main()
        else:
            print("Available commands:")
            print("  python searx_smart_start.py start   # Starts Searx")
            print("  python searx_smart_start.py status  # Displays status")
            print("  python searx_smart_start.py stop    # Stops all")
    else:
        # Default start
        main()
