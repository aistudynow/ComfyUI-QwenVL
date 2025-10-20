# __init__.py for aistudynow ComfyUI Nodes
# Dynamically loads all node scripts in this folder.
# Author: aistudynow

import importlib.util
import os
import sys

# Get current directory
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def load_modules_from_directory(directory):
    """Auto-load all .py modules in the directory except __init__.py."""
    for file in os.listdir(directory):
        if not file.endswith(".py"):
            continue
        file_path = os.path.join(directory, file)
        module_name = os.path.splitext(file)[0]

        # Skip __init__.py itself
        if module_name == "__init__":
            continue

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

            print(f"[aistudynow] Loaded module: {module_name}")

        except Exception as e:
            print(f"[aistudynow] Error loading module {module_name}: {e}")

# Load all modules in the current directory
load_modules_from_directory(current_dir)

# Sort nodes alphabetically by display name for cleaner UI
NODE_CLASS_MAPPINGS = dict(
    sorted(NODE_CLASS_MAPPINGS.items(), key=lambda x: NODE_DISPLAY_NAME_MAPPINGS.get(x[0], x[0]))
)
NODE_DISPLAY_NAME_MAPPINGS = dict(sorted(NODE_DISPLAY_NAME_MAPPINGS.items(), key=lambda x: x[1]))

# Web directory for custom JS / UI assets
WEB_DIRECTORY = "./web"

def load_javascript(web_directory):
    """Register any JavaScript files for UI integration."""
    js_files = []
    js_path = os.path.join(web_directory, "refreshNode.js")
    if os.path.exists(js_path):
        js_files.append({"path": "refreshNode.js"})
    return js_files

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "load_javascript",
]
