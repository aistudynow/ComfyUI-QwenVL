# __init__.py for aistudynow ComfyUI Nodes
import importlib.util
import os
import sys
from typing import Any, Dict

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

def load_modules_from_directory(directory):
    for file in os.listdir(directory):
        if not file.endswith(".py"):
            continue
        module_name = os.path.splitext(file)[0]
        if module_name == "__init__":
            continue

        file_path = os.path.join(directory, file)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                print(f"[aistudynow] Error loading module {module_name}: Spec or loader is None")
                continue
            module = importlib.util.module_from_spec(spec)  # type: ignore
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore

            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

            print(f"[aistudynow] Loaded module: {module_name}")
        except Exception as e:
            print(f"[aistudynow] Error loading module {module_name}: {e}")

load_modules_from_directory(current_dir)

NODE_CLASS_MAPPINGS = dict(
    sorted(NODE_CLASS_MAPPINGS.items(), key=lambda x: str(NODE_DISPLAY_NAME_MAPPINGS.get(x[0], x[0])))
)
NODE_DISPLAY_NAME_MAPPINGS = dict(sorted(NODE_DISPLAY_NAME_MAPPINGS.items(), key=lambda x: str(x[1])))

WEB_DIRECTORY = "./web"

def load_javascript(web_directory):
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
