import os
import sys
import importlib
import importlib.util
import inspect
import time
import logging
import json
from typing import Dict, List, Any, Callable, Optional, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent

# Path Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(PROJECT_ROOT, 'modules')
MODULE_REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'module_registry.json')

# Create modules directory if it doesn't exist
if not os.path.exists(MODULES_DIR):
    os.makedirs(MODULES_DIR)
    # Create an __init__.py file so Python recognizes the folder as a package
    with open(os.path.join(MODULES_DIR, '__init__.py'), 'w') as f:
        f.write('# Module package initialization\n')

# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'module_manager.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ModuleManager')

class ModuleInfo:
    """Class to store information about an enhancement module."""
    
    def __init__(
        self, 
        name: str, 
        path: str,
        module_obj: Any = None,
        enabled: bool = True,
        priority: int = 100,
        description: str = "",
        version: str = "0.1",
        dependencies: List[str] = None,
        hooks: List[str] = None,
        processor: Callable = None,
        creation_time: float = None,
        last_modified: float = None,
        module_type: str = "standard"
    ):
        self.name = name
        self.path = path
        self.module_obj = module_obj
        self.enabled = enabled
        self.priority = priority  # Lower = higher priority
        self.description = description
        self.version = version
        self.dependencies = dependencies or []
        self.hooks = hooks or []
        self.processor = processor
        self.creation_time = creation_time or time.time()
        self.last_modified = last_modified or time.time()
        self.error = None
        self.module_type = module_type  # 'standard', 'class_based', 'function_based', 'auto_generated'
        self.available_functions = {}  # Dictionary of available functions in this module
    
    def to_dict(self) -> Dict:
        """Converts the object to a dictionary for serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "enabled": self.enabled,
            "priority": self.priority,
            "description": self.description,
            "version": self.version,
            "dependencies": self.dependencies,
            "hooks": self.hooks,
            "creation_time": self.creation_time,
            "last_modified": self.last_modified,
            "module_type": self.module_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModuleInfo':
        """Creates an instance from a dictionary."""
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 100),
            description=data.get("description", ""),
            version=data.get("version", "0.1"),
            dependencies=data.get("dependencies", []),
            hooks=data.get("hooks", []),
            creation_time=data.get("creation_time"),
            last_modified=data.get("last_modified"),
            module_type=data.get("module_type", "standard")
        )

class ModuleRegistry:
    """Manages the registry of enhancement modules."""
    
    def __init__(self, registry_path: str = MODULE_REGISTRY_PATH):
        self.registry_path = registry_path
        self.modules: Dict[str, ModuleInfo] = {}
        self.load_registry()
    
    def load_registry(self) -> None:
        """Loads the module registry from a JSON file."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for module_name, module_data in data.items():
                        self.modules[module_name] = ModuleInfo.from_dict(module_data)
                logger.info(f"Module registry loaded with {len(self.modules)} modules")
            except Exception as e:
                logger.error(f"Error loading module registry: {e}")
        else:
            logger.info("No module registry found, starting with empty registry")
    
    def save_registry(self) -> None:
        """Saves the module registry to a JSON file."""
        try:
            serialized = {name: module.to_dict() for name, module in self.modules.items()}
            with open(self.registry_path, 'w') as f:
                json.dump(serialized, f, indent=4)
            logger.info(f"Module registry saved with {len(self.modules)} modules")
        except Exception as e:
            logger.error(f"Error saving module registry: {e}")
    
    def register_module(self, module_info: ModuleInfo) -> bool:
        """Adds a module to the registry."""
        try:
            self.modules[module_info.name] = module_info
            self.save_registry()
            logger.info(f"Module '{module_info.name}' registered successfully")
            return True
        except Exception as e:
            logger.error(f"Error registering module '{module_info.name}': {e}")
            return False
    
    def unregister_module(self, module_name: str) -> bool:
        """Removes a module from the registry."""
        if module_name in self.modules:
            del self.modules[module_name]
            self.save_registry()
            logger.info(f"Module '{module_name}' unregistered")
            return True
        else:
            logger.warning(f"Attempted to unregister non-existent module '{module_name}'")
            return False
    
    def get_module(self, module_name: str) -> Optional[ModuleInfo]:
        """Retrieves a module by its name."""
        return self.modules.get(module_name)
    
    def get_modules_by_hook(self, hook_name: str) -> List[ModuleInfo]:
        """Retrieves all modules that implement a specific hook."""
        modules = [
            m for m in self.modules.values() 
            if hook_name in (m.hooks or []) and m.enabled and m.processor is not None
        ]
        # Sort by priority (smaller number = higher priority)
        return sorted(modules, key=lambda m: m.priority)
    
    def get_all_enabled_modules(self) -> List[ModuleInfo]:
        """Retrieves all enabled modules."""
        return sorted(
            [m for m in self.modules.values() if m.enabled],
            key=lambda m: m.priority
        )

class ModuleAdapter:
    """
    Adapts different types of modules to integrate into the system.
    Generates processors for modules that do not have one.
    """
    
    @staticmethod
    def create_generic_processor(module_obj):
        """
        Creates a generic processor for a module that does not have a process() function.
        
        Args:
            module_obj: The Python module object
            
        Returns:
            An automatically generated process() function
        """
        # Find useful functions in the module
        functions = {}
        classes = {}
        
        for name, obj in inspect.getmembers(module_obj):
            # Ignore private/special attributes and imports
            if name.startswith('_') or name == 'process':
                continue
            
            # Collect useful functions
            if inspect.isfunction(obj):
                functions[name] = obj
            # Collect useful classes
            elif inspect.isclass(obj) and obj.__module__ == module_obj.__name__:
                classes[name] = obj
        
        # Function that attempts to use module functionalities
        def generic_process(data, hook):
            """
            Generic processor that tries to use module functionalities
            based on input data and the hook.
            """
            result = data.copy()
            
            # Log hook and data information
            logger.debug(f"Generic processor for {module_obj.__name__}: hook={hook}, data_keys={list(data.keys()) if isinstance(data, dict) else 'non-dict'}")
            
            # Try to invoke useful functions based on hook and data
            try:
                # If the module has a hook-specific function
                hook_fn_name = f"handle_{hook}"
                if hook_fn_name in functions:
                    logger.debug(f"Using specific hook handler {hook_fn_name}")
                    functions[hook_fn_name](result)
                    return result
                
                # If the module has a class with the same name as the module
                module_name = module_obj.__name__.split('.')[-1]
                if module_name in classes:
                    cls = classes[module_name]
                    instance = cls()
                    
                    # Look for relevant methods
                    if hasattr(instance, hook_fn_name):
                        logger.debug(f"Using class method {hook_fn_name}")
                        getattr(instance, hook_fn_name)(result)
                        return result
                    
                    if hasattr(instance, "process"):
                        logger.debug(f"Using class method process")
                        instance.process(result, hook)
                        return result
            except Exception as e:
                logger.warning(f"Error in generic processor for {module_obj.__name__}: {e}")
            
            # Simply return unchanged data if no appropriate method is found
            return result
        
        # Attach metadata to our function
        generic_process.__module__ = module_obj.__name__
        generic_process.__name__ = "generic_process"
        generic_process.__doc__ = f"Auto-generated processor for module {module_obj.__name__}"
        
        return generic_process

    @staticmethod
    def detect_module_type(module_obj) -> str:
        """
        Detects the module type based on its structure.
        
        Args:
            module_obj: The Python module object
            
        Returns:
            The module type ('standard', 'class_based', 'function_based')
        """
        # Look for a standard process function
        if hasattr(module_obj, 'process') and callable(getattr(module_obj, 'process')):
            return "standard"
        
        # Look for a class with a process method
        classes = {name: obj for name, obj in inspect.getmembers(module_obj, inspect.isclass)
                  if obj.__module__ == module_obj.__name__}
        
        for cls_name, cls in classes.items():
            if hasattr(cls, 'process') and callable(getattr(cls, 'process')):
                return "class_based"
        
        # Look for useful functions
        functions = {name: obj for name, obj in inspect.getmembers(module_obj, inspect.isfunction)
                    if not name.startswith('_')}
        
        if functions:
            return "function_based"
        
        return "unknown"
    
    @staticmethod
    def get_module_hooks(module_obj) -> List[str]:
        """
        Automatically detects hooks supported by a module.
        
        Args:
            module_obj: The Python module object
            
        Returns:
            List of supported hooks
        """
        hooks = set()
        default_hooks = ["process_request", "process_response"]
        
        # Look for handle_X type functions
        for name, obj in inspect.getmembers(module_obj):
            if callable(obj) and name.startswith('handle_'):
                hook = name[7:]  # Remove 'handle_' prefix
                hooks.add(hook)
        
        # Look within classes
        classes = {name: obj for name, obj in inspect.getmembers(module_obj, inspect.isclass)
                  if obj.__module__ == module_obj.__name__}
        
        for cls_name, cls in classes.items():
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if name.startswith('handle_'):
                    hook = name[7:]
                    hooks.add(hook)
        
        # If MODULE_METADATA exists and contains hooks
        if hasattr(module_obj, 'MODULE_METADATA') and isinstance(module_obj.MODULE_METADATA, dict):
            if 'hooks' in module_obj.MODULE_METADATA and isinstance(module_obj.MODULE_METADATA['hooks'], list):
                hooks.update(module_obj.MODULE_METADATA['hooks'])
        
        # Add default hooks if no hooks were found
        if not hooks:
            hooks.update(default_hooks)
        
        return list(hooks)

class ModuleLoader:
    """Dynamically loads Python modules."""
    
    def __init__(self, modules_dir: str = MODULES_DIR, registry: ModuleRegistry = None):
        self.modules_dir = modules_dir
        self.registry = registry or ModuleRegistry()
    
    def load_module(self, module_path: str) -> Optional[ModuleInfo]:
        """
        Loads a Python module from its path and extracts its metadata.
        """
        try:
            if not module_path.endswith('.py') or os.path.basename(module_path).startswith('__'):
                return None
            
            module_name = os.path.basename(module_path).replace('.py', '')
            module_rel_path = os.path.relpath(module_path, PROJECT_ROOT)
            
            # Check if the module is already loaded and up-to-date
            existing_module = self.registry.get_module(module_name)
            if existing_module and existing_module.path == module_rel_path:
                last_modified = os.path.getmtime(module_path)
                if existing_module.last_modified >= last_modified:
                    # The module is already up-to-date
                    logger.debug(f"Module {module_name} is already up to date")
                    return existing_module
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                logger.error(f"Failed to create spec for {module_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract module metadata
            metadata = getattr(module, 'MODULE_METADATA', {})
            
            # Determine module type and proceed accordingly
            module_type = ModuleAdapter.detect_module_type(module)
            
            # Get the appropriate processing function
            process_fn = None
            
            if module_type == "standard":
                process_fn = getattr(module, 'process', None)
            elif module_type == "class_based":
                # Look for a class with a process method
                classes = {name: obj for name, obj in inspect.getmembers(module, inspect.isclass)
                          if obj.__module__ == module.__name__}
                
                for cls_name, cls in classes.items():
                    # Prefer a class with the same name as the module
                    if cls_name.lower() == module_name.lower() and hasattr(cls, 'process'):
                        instance = cls()
                        process_fn = instance.process
                        break
                
                # If no preferred class is found, take the first one with a process method
                if not process_fn:
                    for cls_name, cls in classes.items():
                        if hasattr(cls, 'process'):
                            instance = cls()
                            process_fn = instance.process
                            break
            
            # If no process function is found, generate a generic one
            if not process_fn or not callable(process_fn):
                process_fn = ModuleAdapter.create_generic_processor(module)
                module_type = "auto_generated"
                logger.info(f"Created generic processor for module {module_name}")
            
            # Determine supported hooks
            hooks = metadata.get('hooks', ModuleAdapter.get_module_hooks(module))
            
            # Create ModuleInfo object
            module_info = ModuleInfo(
                name=module_name,
                path=module_rel_path,
                module_obj=module,
                enabled=metadata.get('enabled', True),
                priority=metadata.get('priority', 100),
                description=metadata.get('description', ''),
                version=metadata.get('version', '0.1'),
                dependencies=metadata.get('dependencies', []),
                hooks=hooks,
                processor=process_fn,
                creation_time=existing_module.creation_time if existing_module else time.time(),
                last_modified=os.path.getmtime(module_path),
                module_type=module_type
            )
            
            # Collect available functions
            module_info.available_functions = {
                name: obj for name, obj in inspect.getmembers(module, callable)
                if not name.startswith('_') and obj.__module__ == module.__name__
            }
            
            # Register in the registry
            self.registry.register_module(module_info)
            
            logger.info(f"Successfully loaded module {module_name} (v{module_info.version}) as {module_type} module")
            return module_info
            
        except Exception as e:
            logger.error(f"Error loading module {module_path}: {str(e)}")
            
            # Create error module if possible
            try:
                module_name = os.path.basename(module_path).replace('.py', '')
                module_rel_path = os.path.relpath(module_path, PROJECT_ROOT)
                error_module = ModuleInfo(
                    name=module_name,
                    path=module_rel_path,
                    enabled=False,
                    description=f"Error: {str(e)}",
                    last_modified=os.path.getmtime(module_path)
                )
                error_module.error = str(e)
                self.registry.register_module(error_module)
            except:
                pass
                
            return None
    
    def load_all_modules(self) -> Dict[str, ModuleInfo]:
        """
        Loads all Python modules present in the modules directory.
        """
        loaded_modules = {}
        for root, _, files in os.walk(self.modules_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_path = os.path.join(root, file)
                    module_info = self.load_module(module_path)
                    if module_info:
                        loaded_modules[module_info.name] = module_info
                        
                        # Log module type information
                        if module_info.module_type == "auto_generated":
                            logger.debug(f"Auto-generated processor used for module {module_info.name}")
        
        logger.info(f"Loaded {len(loaded_modules)} modules")
        return loaded_modules
    
    def reload_module(self, module_name: str) -> Optional[ModuleInfo]:
        """
        Reloads a specific module by its name.
        """
        module_info = self.registry.get_module(module_name)
        if not module_info:
            logger.warning(f"Attempted to reload non-existent module '{module_name}'")
            return None
        
        module_path = os.path.join(PROJECT_ROOT, module_info.path)
        
        # Clear module cache
        if module_info.name in sys.modules:
            del sys.modules[module_info.name]
        
        return self.load_module(module_path)

class ModuleFileWatcher(FileSystemEventHandler):
    """Monitors changes in the modules directory."""
    
    def __init__(self, loader: ModuleLoader):
        super().__init__()
        self.loader = loader
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"New module detected: {event.src_path}")
            self.loader.load_module(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"Module modified: {event.src_path}")
            self.loader.load_module(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            module_name = os.path.basename(event.src_path).replace('.py', '')
            logger.info(f"Module deleted: {module_name}")
            self.loader.registry.unregister_module(module_name)

class ModuleManager:
    """
    Manages loading, watching, and executing enhancement modules.
    """
    
    def __init__(self):
        self.registry = ModuleRegistry()
        self.loader = ModuleLoader(registry=self.registry)
        self.file_watcher = ModuleFileWatcher(self.loader)
        self.observer = Observer()
        self.started = False
    
    def start(self):
        """Starts the module manager."""
        if self.started:
            return
        
        # Initially load all modules
        self.loader.load_all_modules()
        
        # Start file monitoring
        self.observer.schedule(self.file_watcher, MODULES_DIR, recursive=True)
        self.observer.start()
        
        self.started = True
        logger.info("Module manager started")
    
    def stop(self):
        """Stops the module manager."""
        if not self.started:
            return
        
        self.observer.stop()
        self.observer.join()
        
        self.started = False
        logger.info("Module manager stopped")
    
    def process_with_modules(self, 
                           request_data: Dict[str, Any], 
                           hook: str = 'process_request') -> Dict[str, Any]:
        """
        Processes a request using all registered modules for a specific hook.
        Always guarantees a valid dictionary return.
        """
        try:
            # Log input type for debugging
            logger.debug(f"process_with_modules input type for {hook}: {type(request_data)}")
            
            modules = self.registry.get_modules_by_hook(hook)
            
            # TYPE PROTECTION: Ensure input data is a dictionary
            if not isinstance(request_data, dict):
                logger.error(f"Invalid request_data type: {type(request_data)}, expected dict")
                
                # Convert to dictionary based on type
                if isinstance(request_data, str):
                    result = {"text": request_data}
                elif request_data is None:
                    result = {"text": ""}  # Default value for None
                else:
                    # Try to convert to dictionary if possible
                    try:
                        result = dict(request_data)  # Tries to convert to dict if it's iterable
                    except (TypeError, ValueError):
                        result = {"data": str(request_data)}  # Safe fallback
            else:
                # Create a deep copy to avoid side effects
                import copy
                
                # Use deepcopy for a true deep copy
                try:
                    result = copy.deepcopy(request_data)
                except Exception as e:
                    # If deepcopy fails, do a manual copy
                    logger.warning(f"Deepcopy failed: {str(e)}, falling back to manual copy")
                    result = {}
                    # Manual copy for primary keys
                    for key, value in request_data.items():
                        if isinstance(value, str):
                            result[key] = value
                        elif isinstance(value, dict):
                            # Secure recursive copy of dictionaries
                            try:
                                result[key] = copy.deepcopy(value)
                            except:
                                # If deepcopy fails, do a simple copy
                                result[key] = value.copy() if hasattr(value, 'copy') else value
                        elif isinstance(value, (list, tuple, set)):
                            # Recursive copy for collections
                            try:
                                result[key] = copy.deepcopy(value)
                            except:
                                # If deepcopy fails, do a simple copy
                                result[key] = value.copy() if hasattr(value, 'copy') else list(value)
                        else:
                            # For immutable or non-copyable types, assign directly
                            result[key] = value
            
            # Process by each module
            for module_info in modules:
                try:
                    if not module_info.processor:
                        logger.debug(f"Module {module_info.name} has no processor, skipping")
                        continue
                    
                    # Check that result is indeed a dictionary before sending to the module
                    if not isinstance(result, dict):
                        logger.warning(f"Result became non-dict before module {module_info.name}: {type(result)}")
                        result = {"text": str(result) if result is not None else ""}
                    
                    # Apply module processing with input protection
                    try:
                        processed_result = module_info.processor(result.copy() if hasattr(result, 'copy') else result, hook)
                    except Exception as e:
                        logger.error(f"Module {module_info.name} processor raised exception: {str(e)}")
                        # Continue with the current result without modifying
                        continue
                    
                    # RESULT VALIDATION: Ensure the result is still a dictionary
                    if processed_result is not None:
                        if isinstance(processed_result, dict):
                            result = processed_result
                        else:
                            logger.warning(f"Module {module_info.name} returned non-dict result: {type(processed_result)}")
                            # If the module returns a string, put it in the 'text' field
                            if isinstance(processed_result, str):
                                # Preserve other keys from the previous result
                                prev_result = result.copy() if hasattr(result, 'copy') else {}
                                prev_result["text"] = processed_result
                                result = prev_result
                            else:
                                # For other types, store them in a generic field
                                # while preserving the previous dictionary
                                prev_result = result.copy() if hasattr(result, 'copy') else {}
                                prev_result["processed_data"] = processed_result
                                result = prev_result
                except Exception as e:
                    logger.error(f"Error processing with module {module_info.name}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Continue with other modules even if one fails
            
            # FINAL VALIDATION: Ensure the result is a valid dictionary
            if not isinstance(result, dict):
                logger.error(f"Final result is not a dict: {type(result)}. Converting to dict.")
                return {"text": str(result) if result is not None else ""}
            
            # Ensure the dictionary contains at least one key
            if len(result) == 0:
                logger.warning("Result dictionary is empty, adding default text key")
                result["text"] = ""
                
            return result
            
        except Exception as e:
            logger.error(f"Critical error in process_with_modules: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a minimal dictionary in case of critical error
            return {"text": request_data.get("text", "") if isinstance(request_data, dict) else str(request_data) if request_data is not None else ""}
        
        return result
    
    def get_module_info(self, module_name: str = None) -> Dict[str, Any]:
        """
        Returns information about loaded modules.
        
        Args:
            module_name: Name of the specific module or None for all modules
            
        Returns:
            Information about the module(s)
        """
        if module_name:
            module = self.registry.get_module(module_name)
            if not module:
                return {"error": f"Module {module_name} not found"}
            
            return {
                "name": module.name,
                "description": module.description,
                "version": module.version,
                "enabled": module.enabled,
                "hooks": module.hooks,
                "type": module.module_type,
                "has_processor": module.processor is not None,
                "available_functions": list(getattr(module, "available_functions", {}).keys()),
                "error": module.error
            }
        else:
            modules_info = []
            for name, module in self.registry.modules.items():
                modules_info.append({
                    "name": module.name,
                    "description": module.description,
                    "version": module.version,
                    "enabled": module.enabled,
                    "type": module.module_type,
                    "hooks": module.hooks,
                    "has_processor": module.processor is not None,
                    "error": module.error
                })
            return {"modules": modules_info}

# Singleton for module management
_module_manager = None

def get_module_manager() -> ModuleManager:
    """Retrieves the singleton instance of the module manager."""
    global _module_manager
    if _module_manager is None:
        _module_manager = ModuleManager()
    return _module_manager
