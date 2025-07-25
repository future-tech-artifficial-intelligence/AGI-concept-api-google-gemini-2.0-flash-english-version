"""
Test script for the enhancement modules system.
This script allows testing dynamic module loading and their application.
"""

import json
import sys
import time
from module_manager import get_module_manager, ModuleInfo

def print_module_info(module_info: ModuleInfo):
    """Displays information about a module."""
    print(f"Module: {module_info.name} (v{module_info.version})")
    print(f"  Type: {module_info.module_type}")
    print(f"  Enabled: {module_info.enabled}")
    print(f"  Priority: {module_info.priority}")
    print(f"  Description: {module_info.description}")
    print(f"  Path: {module_info.path}")
    print(f"  Hooks: {', '.join(module_info.hooks) if module_info.hooks else 'None'}")
    print(f"  Has processor: {'Yes' if module_info.processor else 'No'}")
    print(f"  Dependencies: {', '.join(module_info.dependencies) if module_info.dependencies else 'None'}")
    print(f"  Last modified: {time.ctime(module_info.last_modified)}")
    if hasattr(module_info, 'available_functions'):
        print(f"  Available functions: {', '.join(module_info.available_functions.keys())}")
    print()

def test_module_loading():
    """Tests module loading."""
    print("=== TESTING MODULE LOADING ===")

    manager = get_module_manager()
    manager.start()

    # Display all loaded modules
    print("\nAll loaded modules:")
    modules = manager.registry.get_all_enabled_modules()
    if not modules:
        print("  No modules loaded.")
    else:
        for module in modules:
            print_module_info(module)

    # Check modules with auto-generated processors
    auto_generated_modules = [m for m in modules if m.module_type == "auto_generated"]
    if auto_generated_modules:
        print("\nModules with auto-generated processors:")
        for module in auto_generated_modules:
            print(f"  - {module.name}")

    # Test module reloading
    if modules:
        module_name = modules[0].name
        print(f"\nReloading module: {module_name}")
        reloaded = manager.loader.reload_module(module_name)
        if reloaded:
            print(f"  Module {module_name} successfully reloaded.")
        else:
            print(f"  Failed to reload module {module_name}.")

    return manager

def test_module_processing(manager):
    """Tests processing with modules."""
    print("\n=== TESTING MODULE PROCESSING ===")

    # Example requests to test
    test_requests = [
        {
            "contents": [{
                "parts": [{
                    "text": "Why is the sky blue?"
                }]
            }],
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        },
        {
            "contents": [{
                "parts": [{
                    "text": "Can you give me a plan to organize a birthday party?"
                }]
            }],
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        }
    ]

    for i, request in enumerate(test_requests):
        print(f"\nProcessing request #{i+1}:")
        print(f"  Original: {request['contents'][0]['parts'][0]['text']}")

        # Process with modules
        processed = manager.process_with_modules(request, "process_request")

        print(f"  Processed: {processed['contents'][0]['parts'][0]['text']}")

        # Check if the request was identified as requiring planning
        if 'requires_planning' in processed:
            print(f"  Planning required: {processed['requires_planning']}")
            print(f"  Planning keywords: {processed.get('planning_keywords_detected', [])}")
        print()

def test_get_module_info(manager):
    """Tests retrieving module information."""
    print("\n=== TESTING MODULE INFO API ===")

    # Get information about all modules
    info = manager.get_module_info()
    print(f"Total modules: {len(info['modules'])}")

    # Get information about a specific module
    modules = manager.registry.get_all_enabled_modules()
    if modules:
        module_name = modules[0].name
        module_info = manager.get_module_info(module_name)
        print(f"\nDetails for module '{module_name}':")
        for key, value in module_info.items():
            print(f"  {key}: {value}")

def main():
    """Main test function."""
    print("Module System Test\n")

    manager = test_module_loading()
    test_module_processing(manager)
    test_get_module_info(manager)

    print("\nTest completed. Press Ctrl+C to exit.")
    try:
        # Keep the program running to monitor file changes
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
        print("\nTest stopped.")

if __name__ == "__main__":
    main()
