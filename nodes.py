# ComfyUI custom node template
#
# Copy this template and paste it into a new file at ComfyUi/custom_nodes/RENAMEME_NODE_CLASS_NAME.py
# Rename each of the items that start with 'RENAMEME_' (including the file name)
#
# For more information on each of these fields, see the comments of the example_node.py inside of ComfyUI:
# https://github.com/comfyanonymous/ComfyUI/blob/master/custom_nodes/example_node.py.example
#
# Names
#
# RENAMEME_NODE_CLASS_NAME - The name of your custom node class. Name it using CamelCase.
# RENAMEME_NODE_NAME - The name of your custom node as it is represented to ComfyUI.
#   This can be anything, but it's simplest to use the same thing as RENAMEME_NODE_CLASS_NAME.
# RENAMEME_NODE_CATEGORY - The category of your custom node class.
#   When creating a new node in the ComfyUI web interface using the context menu in the following way:
#   Double Click > RENAMEME_NODE_CATEGORY > RENAMEME_DISPLAYED_NODE_NAME
# RENAMEME_DISPLAYED_NODE_NAME - The name of your custom node as it is displayed in the ComfyUI web interface.
#
# Inputs and Outputs
#
# RENAMEME_INPUT_NAME - The name of your node input.
#   This will be used in the web interface of ComfyUI as well as in the run function.
# RENAMEME_INPUT_TYPE - The type of your node input.
# RENAMEME_OUTPUT_NAME - The name of your node output.
#   This willbe used in the web interface of ComfyUI.
# RENAMEME_OUTPUT_TYPE - The type of your node output.


class RENAMEME_NODE_CLASS_NAME:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "RENAMEME_INPUT_NAME": ("RENAMEME_INPUT_TYPE",),
            },
        }

    RETURN_TYPES = ("RENAMEME_OUTPUT_TYPE",)
    RETURN_NAMES = ("RENAMEME_OUTPUT_NAME",)

    FUNCTION = "run"

    CATEGORY = "RENAMEME_NODE_CATEGORY"

    def run(self, RENAMEME_INPUT_NAME):
        return (RENAMEME_OUTPUT_NAME,)


NODE_CLASS_MAPPINGS = {"RENAMEME_NODE_NAME": RENAMEME_NODE_CLASS_NAME}

NODE_DISPLAY_NAME_MAPPINGS = {"RENAMEME_NODE_NAME": "RENAMEME_DISPLAYED_NODE_NAME"}
