import sys
import os

# Get the parent directory of the custom node
custom_node_root = os.path.dirname(os.path.abspath(__file__))

# Set an environment variable to store the custom node's root directory
os.environ["DREAMWALTZ_ROOT_FOLDER"] = custom_node_root

# Debug print statement to display the root folder path
print(f"DREAMWALTZ_ROOT_FOLDER is set to: {os.environ['DREAMWALTZ_ROOT_FOLDER']}")

# Add the custom node's root to sys.path if it's not already present
if custom_node_root not in sys.path:
    sys.path.append(custom_node_root)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
