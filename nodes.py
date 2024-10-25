import os
from pathlib import Path
import subprocess
import sys
import re
import torch
from PIL import Image
import numpy as np


class DreamWaltzGStageOneTrainer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
                "iterations": (
                    "INT",
                    {"default": 10000, "min": 1000, "max": 100000, "step": 1000},
                ),
                "train_resolution": (
                    ["64,128,256", "64,128", "128,256", "256"],
                    {"default": "64,128,256"},
                ),
                "background_mode": (["gray", "white", "black"], {"default": "gray"}),
                "body_parts": (["hands", "face", "full"], {"default": "hands"}),
                "scene_type": (["canonical", "dynamic"], {"default": "canonical"}),
                "enable_fp16": ("BOOLEAN", {"default": True}),
                "enable_progressive_grid": ("BOOLEAN", {"default": True}),
                "enable_sigma_guidance": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("checkpoints_folder_path", "images")
    FUNCTION = "init_training"
    CATEGORY = "DreamWaltzG"

    def sanitize_prompt(self, prompt):
        # Remove leading/trailing whitespace
        prompt = prompt.strip()
        # Replace multiple spaces with single underscore
        prompt = re.sub(r"\s+", "_", prompt)
        # Remove any special characters except underscores
        prompt = re.sub(r"[^a-zA-Z0-9_]", "", prompt)
        return prompt

    def init_training(
        self,
        prompt,
        iterations,
        train_resolution,
        background_mode,
        body_parts,
        scene_type,
        enable_fp16,
        enable_progressive_grid,
        enable_sigma_guidance,
    ):
        sanitized_prompt = self.sanitize_prompt(prompt)

        root_folder = Path(os.environ.get("DREAMWALTZ_ROOT_FOLDER", "."))

        exp_name = f"{sanitized_prompt}/nerf_64_256_{iterations}k"
        exp_full_path = root_folder / "outputs" / exp_name

        checkpoints_folder_path = exp_full_path / "checkpoints"

        results_folder_path = exp_full_path / "results" / "1024x1024" / "image"

        # Ensure the checkpoints directory exists
        os.makedirs(str(checkpoints_folder_path), exist_ok=True)

        # Define the path to main.py and hardcoded checkpoint path
        main_script_path = "main.py"
        checkpoint_path = "external/human_templates/instant-ngp/adult_neutral"

        # Build command with user parameters
        cmd = [
            "python",
            main_script_path,
            "--guide.text",
            prompt,
            "--log.exp_name",
            exp_name,
            "--optim.ckpt",
            checkpoint_path,
            "--predefined_body_parts",
            body_parts,
            "--stage",
            "nerf",
            "--nerf.bg_mode",
            background_mode,
            "--optim.fp16",
            str(enable_fp16).lower(),
            "--optim.iters",
            str(iterations),
            "--prompt.scene",
            scene_type,
            "--data.train_w",
            train_resolution,
            "--data.train_h",
            train_resolution,
            "--data.progressive_grid",
            str(enable_progressive_grid).lower(),
            "--use_sigma_guidance",
            str(enable_sigma_guidance).lower(),
        ]

        print(f"Original prompt: {prompt}")
        print(f"Sanitized prompt: {sanitized_prompt}")
        print(f"Executing command from directory: {root_folder}")
        print(f"Executing command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(root_folder),
            )

            # Read and print output in real-time
            while True:
                output = process.stdout.readline()
                if output:
                    print(output.strip(), flush=True)
                    sys.stdout.flush()

                if process.poll() is not None:
                    break

            # Get the return code
            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            raise Exception(f"Training script failed with error code {e.returncode}")

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

        images = []
        if os.path.exists(results_folder_path):
            image_files = sorted(
                [
                    f
                    for f in os.listdir(results_folder_path)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            for img_file in image_files:
                img_path = results_folder_path / img_file
                img = Image.open(img_path).convert("RGB")
                images.append(
                    np.array(img, dtype=np.float32) / 255.0
                )  # Normalized to [0, 1]

        images_tensor = torch.from_numpy(np.stack(images)) if images else None

        print(str(exp_full_path))
        return (str(exp_full_path), images_tensor)


class DreamWaltzGStageTwoTrainer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_one_checkpoint_path": (
                    "STRING",
                    {"default": ""},
                ),  # Output from stage one
                "prompt": ("STRING", {"default": ""}),  # Should match stage one prompt
                "iterations": (
                    "INT",
                    {"default": 5000, "min": 1000, "max": 10000, "step": 1000},
                ),
                "body_parts": (["hands", "face", "full"], {"default": "hands"}),
                "scene_type": (["canonical", "dynamic"], {"default": "canonical"}),
                "learn_hand_betas": ("BOOLEAN", {"default": True}),
                "lbs_weight_smooth": ("BOOLEAN", {"default": True}),
                "background_color": (
                    "STRING",
                    {"default": "0.5,0.5,0.5", "placeholder": "r,g,b values (0-1)"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("checkpoints_folder_path", "images")
    FUNCTION = "train_3dgs"
    CATEGORY = "DreamWaltzG"

    def validate_background_color(self, bg_color_str):
        try:
            # Split the string by comma and convert to floats
            values = [float(x.strip()) for x in bg_color_str.split(",")]
            if len(values) != 3:
                raise ValueError("Background color must have exactly 3 values")
            if not all(0 <= v <= 1 for v in values):
                raise ValueError("Background color values must be between 0 and 1")
            return values
        except Exception as e:
            raise ValueError(f"Invalid background color format: {str(e)}")

    def sanitize_prompt(self, prompt):
        # Remove leading/trailing whitespace
        prompt = prompt.strip()
        # Replace multiple spaces with single underscore
        prompt = re.sub(r"\s+", "_", prompt)
        # Remove any special characters except underscores
        prompt = re.sub(r"[^a-zA-Z0-9_]", "", prompt)
        return prompt

    def train_3dgs(
        self,
        stage_one_checkpoint_path,
        prompt,
        iterations,
        body_parts,
        scene_type,
        learn_hand_betas,
        lbs_weight_smooth,
        background_color,
    ):
        # Validate inputs
        if not os.path.exists(stage_one_checkpoint_path):
            raise ValueError(
                f"Stage one checkpoint path does not exist: {stage_one_checkpoint_path}"
            )

        sanitized_prompt = self.sanitize_prompt(prompt)
        bg_color_values = self.validate_background_color(background_color)

        # Get the root folder from environment variable or use default
        root_folder = Path(os.environ.get("DREAMWALTZ_ROOT_FOLDER", "."))

        # Construct the new experiment name by appending 3dgs suffix
        exp_name = f"{stage_one_checkpoint_path}-3dgs,cnl,{iterations//1000}k"
        exp_full_path = Path(exp_name)

        # Create checkpoints and results folders
        checkpoints_folder_path = exp_full_path / "checkpoints"
        results_folder_path = exp_full_path / "results" / "1024x1024" / "image"

        os.makedirs(str(checkpoints_folder_path), exist_ok=True)

        # Define the path to main.py
        main_script_path = "main.py"

        # Build command with user parameters
        cmd = [
            "python",
            main_script_path,
            "--guide.text",
            prompt,
            "--log.exp_name",
            exp_name,
            "--render.from_nerf",
            str(stage_one_checkpoint_path),
            "--predefined_body_parts",
            body_parts,
            "--stage",
            "gs",
            "--optim.iters",
            str(iterations),
            "--prompt.scene",
            scene_type,
            "--render.learn_hand_betas",
            str(learn_hand_betas).lower(),
            "--render.lbs_weight_smooth",
            str(lbs_weight_smooth).lower(),
            "--render.bg_color",
            f"[{bg_color_values[0]},{bg_color_values[1]},{bg_color_values[2]}]",
        ]

        print(f"Original prompt: {prompt}")
        print(f"Sanitized prompt: {sanitized_prompt}")
        print(f"Stage one checkpoint path: {stage_one_checkpoint_path}")
        print(f"Executing command from directory: {root_folder}")
        print(f"Executing command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(root_folder),
            )

            # Read and print output in real-time
            while True:
                output = process.stdout.readline()
                if output:
                    print(output.strip(), flush=True)
                    sys.stdout.flush()

                if process.poll() is not None:
                    break

            # Get the return code
            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            raise Exception(f"Training script failed with error code {e.returncode}")

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

        # Load and process result images
        images = []
        if os.path.exists(results_folder_path):
            image_files = sorted(
                [
                    f
                    for f in os.listdir(results_folder_path)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            for img_file in image_files:
                img_path = results_folder_path / img_file
                img = Image.open(img_path).convert("RGB")
                images.append(
                    np.array(img, dtype=np.float32) / 255.0
                )  # Normalized to [0, 1]

        images_tensor = torch.from_numpy(np.stack(images)) if images else None

        print(f"Training completed. Results saved to: {str(exp_full_path)}")
        return (str(exp_full_path), images_tensor)


# Update the mappings for the node
NODE_CLASS_MAPPINGS = {
    "DreamWaltzGStageOneTrainer": DreamWaltzGStageOneTrainer,
    "DreamWaltzGStageTwoTrainer": DreamWaltzGStageTwoTrainer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamWaltzGStageOneTrainer": "DreamWaltzG Stage One Trainer",
    "DreamWaltzGStageTwoTrainer": "DreamWaltzG Stage Two Trainer",
}
