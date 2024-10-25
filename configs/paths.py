from pathlib import Path
import os

# Fetch the custom node root folder from the environment variable
root_folder = os.environ.get("DREAMWALTZ_ROOT_FOLDER", ".")
root_path = Path(root_folder)  # Convert the root folder to a Path object

# External models
HUMAN_TEMPLATES = root_path / "external" / "human_templates"

# Datasets
AIST_ROOT = root_path / "datasets" / "AIST++"
MOTIONX_ROOT = root_path / "datasets" / "Motion-X"
MOTIONX_REENACT_ROOT = root_path / "datasets" / "Motion-X-ReEnact"
PW3D_ROOT = root_path / "datasets" / "3DPW"
TALKSHOW_ROOT = root_path / "datasets" / "TalkShow"

# Datasets (Not used)
AMASS_ROOT = root_path / "datasets" / "AMASS"
HYBRIK_ROOT = root_path / "datasets" / "HybrIK"
