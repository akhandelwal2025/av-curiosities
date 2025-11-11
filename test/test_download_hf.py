from huggingface_hub import hf_hub_download
import os
import zipfile

HF_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

sensor_type = "lidar"
sensor_name = "lidar_top_360fov"
chunk_counter = 0
zip_root = f"{sensor_type}/{sensor_name}"
zip_to_download = f"{sensor_name}.chunk_{chunk_counter:04d}.zip"
zip_path = hf_hub_download(
    repo_id = HF_REPO_ID,
    filename = os.path.join(zip_root, zip_to_download),
    repo_type = "dataset",
    local_dir = HF_REPO_ID,
)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    extract_dir = os.path.join(HF_REPO_ID, zip_root)
    zip_ref.extractall(extract_dir)