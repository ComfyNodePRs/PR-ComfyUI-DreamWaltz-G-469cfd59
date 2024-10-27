import os
from pathlib import Path
import subprocess
import sys
import re
import torch
from PIL import Image
import numpy as np


import sys
import os
import re
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
import torch


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
                    ["64,64", "64,128,256", "512,512"],
                    {"default": "64,64"},
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
        prompt = prompt.strip()
        prompt = re.sub(r"\s+", "_", prompt)
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

        train_resolution = train_resolution.split(",")

        if len(train_resolution) == 3:
            exp_name = f"{sanitized_prompt}/nerf_{train_resolution[0]}_{train_resolution[1]}_{train_resolution[2]}_{iterations // 1000 }k"
        else:
            exp_name = f"{sanitized_prompt}/nerf_{train_resolution[0]}_{train_resolution[1]}_{iterations // 1000 }k"

        exp_full_path = root_folder / "outputs" / exp_name
        checkpoints_folder_path = exp_full_path / "checkpoints"
        results_folder_path = exp_full_path / "results" / "1024x1024" / "image"

        # Ensure the checkpoints directory exists
        os.makedirs(str(checkpoints_folder_path), exist_ok=True)

        # Find the latest checkpoint in the experiment's output directory
        checkpoint_path = None
        checkpoint_files = sorted(checkpoints_folder_path.glob("*.pth"), reverse=True)
        if checkpoint_files:
            checkpoint_path = str(checkpoint_files[0])  # Use the most recent checkpoint
        else:
            # Default to template checkpoint if no previous checkpoint exists
            checkpoint_path = (
                "external/human_templates/instant-ngp/adult_neutral/step_005000.pth"
            )

        # # Build command with user parameters
        # cmd = [
        #     "python",
        #     "main.py",
        #     "--guide.text",
        #     prompt,
        #     "--log.exp_name",
        #     exp_name,
        #     "--optim.ckpt",
        #     checkpoint_path,
        #     "--optim.resume",
        #     "True",  # Ensure it resumes if checkpoint exists
        #     "--predefined_body_parts",
        #     body_parts,
        #     "--stage",
        #     "nerf",
        #     "--nerf.bg_mode",
        #     background_mode,
        #     "--optim.fp16",
        #     str(enable_fp16).lower(),
        #     "--optim.iters",
        #     str(iterations),
        #     "--prompt.scene",
        #     scene_type,
        #     "--data.train_w",
        #     train_resolution,
        #     "--data.train_h",
        #     train_resolution,
        #     "--data.progressive_grid",
        #     str(enable_progressive_grid).lower(),
        #     "--use_sigma_guidance",
        #     str(enable_sigma_guidance).lower(),
        # ]

        # print(f"Executing command: {' '.join(cmd)}")

        # try:
        #     process = subprocess.Popen(
        #         cmd,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.STDOUT,
        #         text=True,
        #         bufsize=1,
        #         universal_newlines=True,
        #         cwd=str(root_folder),
        #     )

        #     while True:
        #         output = process.stdout.readline()
        #         if output:
        #             print(output.strip(), flush=True)
        #             sys.stdout.flush()

        #         if process.poll() is not None:
        #             break

        #     return_code = process.wait()

        #     if return_code != 0:
        #         raise subprocess.CalledProcessError(return_code, cmd)

        # except subprocess.CalledProcessError as e:
        #     print(f"Error executing command: {e}")
        #     raise Exception(f"Training script failed with error code {e.returncode}")

        # except Exception as e:
        #     print(f"Unexpected error: {e}")
        #     raise

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
                images.append(np.array(img, dtype=np.float32) / 255.0)

        images_tensor = torch.from_numpy(np.stack(images)) if images else None

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
        # Convert stage one checkpoint path to Path object and resolve to absolute path
        stage_one_checkpoint_path = Path(stage_one_checkpoint_path).resolve()

        # Validate inputs
        if not stage_one_checkpoint_path.exists():
            raise ValueError(
                f"Stage one checkpoint path does not exist: {stage_one_checkpoint_path}"
            )

        sanitized_prompt = self.sanitize_prompt(prompt)
        bg_color_values = self.validate_background_color(background_color)

        # Get the root folder from environment variable or use default
        root_folder = Path(os.environ.get("DREAMWALTZ_ROOT_FOLDER", ".")).resolve()

        # Construct the new experiment name by appending 3dgs suffix
        exp_name = (
            root_folder
            / "outputs"
            / sanitized_prompt
            / f"3dgs_cnl_{iterations // 1000}k"
        )
        exp_checkpoint_full_path = exp_name / "checkpoints"

        # Previous Nerf checkpoint
        nerf_checkpoint_path = stage_one_checkpoint_path / "checkpoints"

        # Define the path to main.py relative to root_folder
        main_script_path = root_folder / "main.py"

        cmd = [
            "python",
            str(main_script_path),  # Convert Path to str
            "--guide.text",
            prompt,
            "--log.exp_name",
            str(exp_name),  # Convert Path to str
            "--render.from_nerf",
            str(nerf_checkpoint_path),  # Convert Path to str
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
                    print(output.strip(), flush=True)  # Flushes each line immediately

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
        results_folder_path = None

        # Load and process result images
        images = []
        if results_folder_path.exists():
            image_files = sorted(
                [
                    f
                    for f in results_folder_path.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
            )
            for img_file in image_files:
                img = Image.open(img_file).convert("RGB")
                images.append(
                    np.array(img, dtype=np.float32) / 255.0
                )  # Normalized to [0, 1]

        images_tensor = torch.from_numpy(np.stack(images)) if images else None

        print(f"Training completed. Results saved to: {exp_checkpoint_full_path}")
        return (str(exp_checkpoint_full_path), images_tensor)


# Update the mappings for the node
NODE_CLASS_MAPPINGS = {
    "DreamWaltzGStageOneTrainer": DreamWaltzGStageOneTrainer,
    "DreamWaltzGStageTwoTrainer": DreamWaltzGStageTwoTrainer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamWaltzGStageOneTrainer": "DreamWaltzG Stage One Trainer",
    "DreamWaltzGStageTwoTrainer": "DreamWaltzG Stage Two Trainer",
}
