"""
Module template to demonstrate how to create a module compatible with the system.
This module provides an example that can be copied to create new modules.
"""

# Module metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 100,
    "description": "Example module template",
    "version": "0.1.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

def process(data: dict, hook: str) -> dict:
    """
    Main function called by the module manager.
    
    Args:
        data: The data to process (structure depending on the hook)
        hook: The hook being called (e.g., 'process_request', 'process_response')
        
    Returns:
        The potentially modified data
    """
    # Here, we do nothing, it's just an example
    print(f"Module template called with hook: {hook}")
    return data

# You can also define specific handlers for each hook
def handle_process_request(data):
    """Specific handler for the process_request hook"""
    return data

def handle_process_response(data):
    """Specific handler for the process_response hook"""
    return data

# Other useful functions/classes...
def utility_function():
    """Any utility function"""
    return "Utility function called"

# Example of a class compatible with the system
class ModuleTemplateProcessor:
    """Class that implements processing for this module"""
    
    def __init__(self):
        """Initialization"""
        self.config = {"some_setting": True}
    
    def process(self, data, hook):
        """
        Main processing method (alternative to the process function)
        This method will be used if the process function is not defined at the module level
        """
        print(f"Class-based process called for hook: {hook}")
        return data
    
    def handle_process_request(self, data):
        """Specific handler for the process_request hook"""
        return data
