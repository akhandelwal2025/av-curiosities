import zipfile
import os
import shutil
import polars as pl 
import subprocess
import tempfile
import Dracopy
from huggingface_hub import hf_hub_download

# ========= DATASET DIRECTORIES =========
HF_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
LOCAL_DIR = "data/"

# ========= DATASET CONFIGS =========
CLIP_LENGTH_SEC = 20
CAMERA_NAMES = [
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]

LIDAR_NAMES = [
    "lidar_top_360fov"
]

RADAR_NAMES = [
    "radar_corner_front_left_srr_0",
    "radar_corner_front_left_srr_3",
    "radar_corner_front_right_srr_0",
    "radar_corner_front_right_srr_3",
    "radar_corner_rear_left_srr_0",
    "radar_corner_rear_left_srr_3",
    "radar_corner_rear_right_srr_0",
    "radar_corner_rear_right_srr_3",
    "radar_front_center_imaging_lrr_1",
    "radar_front_center_mrr_2",
    "radar_front_center_srr_0",
    "radar_rear_left_mrr_2",
    "radar_rear_left_srr_0",
    "radar_rear_right_mrr_2",
    "radar_rear_right_srr_0",
    "radar_side_left_srr_0",
    "radar_side_left_srr_3",
    "radar_side_right_srr_0",
    "radar_side_right_srr_3",
]

class NvidiaPhysicalAIUnifiedFrame:
    # general wrapper class to store data from all sensors
    def __init__(self):
        # key = <sensor_name>, value = base64 snippet
        self.cameras = {}
        self.lidars = {}
        self.radars = {}

    def __getitem__(self, keys):
        # expected format of keys: (sensor_type, sensor_name)
        sensor_type, sensor_name = keys
        if sensor_type == "camera":
            return self.cameras.get(sensor_name, None)
        elif sensor_type == "lidar":
            return self.lidars.get(sensor_name, None)
        elif sensor_type == "radar":
            return self.radars.get(sensor_name, None)
        else:
            raise RuntimeError(f"{sensor_type} not in ['camera', 'lidar', 'radar']")
    
    def __setitem__(self, keys, value):
        # expected format of keys: (sensor_type, sensor_name)
        sensor_type, sensor_name = keys
        if sensor_type == "camera":
            self.cameras[sensor_name] = value
        elif sensor_type == "lidar":
            self.lidars[sensor_name] = value
        elif sensor_type == "radar":
            self.radars[sensor_name] = value
        else:
            raise RuntimeError(f"{sensor_type} not in ['camera', 'lidar', 'radar']")

class NvidiaPhysicalAIDataloader:
    def __init__(self,
                 dataloader_cfg):
        self.dataloader_cfg = dataloader_cfg
        clip_idx_parquet_file = os.path.join(HF_REPO_ID, "clip_index.parquet")
        self.clips_df = pl.read_parquet(clip_idx_parquet_file) 

        """
        Keeping track of the terminology here:
            - snippet: X sec portion of a clip
            - clip: 20 sec piece of data for a sensor
            - chunk: group of 100 clips
        """
        self.chunk_counter = 0
        self.clip_counter = 0
        self.snippet_counter = 0
        assert(CLIP_LENGTH_SEC % dataloader_cfg["snippet_length_sec"] == 0,
               "snippet length must be a divisor of clip length")
        self.snippet_length_sec = dataloader_cfg["snippet_length_sec"]
        self.snippet_length_msec = self.snippet_length_sec * 1000
        self.snippets_per_clip = CLIP_LENGTH_SEC / self.snippet_length_sec

        self.str_to_sensor_map = {
            "camera": CAMERA_NAMES if self.dataloader_cfg["use_cameras"] else [],
            "lidar": LIDAR_NAMES if self.dataloader_cfg["use_lidars"] else [],
            "radar": RADAR_NAMES if self.dataloader_cfg["use_radars"] else []
        }
    
    def delete_chunk(self):
        for sensor_name in ["camera", "lidar", "radar"]:
            shutil.rmtree(os.path.join(HF_REPO_ID, sensor_name))

    def download_chunk(self):
        for sensor_type, sensor_list in self.str_to_sensor_map.items():
            for sensor_name in sensor_list:
                zip_root = f"{sensor_type}/{sensor_name}"
                zip_to_download = f"{sensor_name}.chunk_{self.chunk_counter:04d}.zip"
                zip_path = hf_hub_download(
                    repo_id = HF_REPO_ID,
                    filename = os.path.join(zip_root, zip_to_download),
                    repo_type = "dataset",
                    local_dir = HF_REPO_ID,
                )
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    extract_dir = os.path.join(HF_REPO_ID, zip_root)
                    zip_ref.extractall(extract_dir)
    
    def load_clips(self):
        def load_video_clip(clip_id
                            sensor_type,
                            sensor_name):
            video_root = f"{sensor_type}/{sensor_name}"
            video_to_load = f"{clip_id}.{sensor_name}.mp4"
            video_path = os.path.join(video_root, video_to_load)
            
            # use ffmpeg to split into snippets and load all of them
            temp_dir = tempfile.mkdtemp()
            output_pattern = os.path.join(temp_dir, "clip_%03d.mp4")
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-c', 'copy',  # Copy without re-encoding
                '-f', 'segment',
                '-segment_time', str(clip_duration_sec),
                '-reset_timestamps', '1',
                output_pattern,
                '-y'  # Overwrite without asking
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            clips_b64 = []
            clip_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('clip_')])
            for clip_file in clip_files:
                clip_path = os.path.join(temp_dir, clip_file)
                with open(clip_path, 'rb') as f:
                    video_bytes = f.read()
                    clip_b64 = base64.b64encode(video_bytes).decode('utf-8')
                    clips_b64.append(clip_b64)
            return clips_b64
            
        def load_pc_parquet_clip(clip_id,
                                 sensor_type,
                                 sensor_name):
            pc_root = f"{sensor_type}/{sensor_name}"
            pc_to_load = f"{clip_id}.{sensor_name}.parquet"
            pc_path = os.path.join(pc_root, pc_to_load)
            pc_df = pl.read_parquet(pc_path)
            return pc_df
        
        unified_clips = NvidiaPhysicalAIUnifiedFrame()
        clip_id = self.clips_df[self.clip_counter]["clip_id"].item()
        for sensor_type, sensor_list in self.str_to_sensor_map.items():
            for sensor_name in sensor_list:
                if sensor_type == "camera":
                    unified_clips[sensor_type, sensor_name] = load_video_clip(clip_id, sensor_type, sensor_name)
                else:
                    unified_clips[sensor_type, sensor_name] = load_pc_parquet_clip(clip_id, sensor_type, sensor_name)
        return unified_clips

    def load_snippets(self):
        # assumes that self.unified_clips has been updated to contain the 
        # most up-to-date frames
        def load_video_snippet(self,
                               sensor_type,
                               sensor_name):
            return self.unified_clips[sensor_type, sensor_name][self.snippet_counter]
        
        def load_pc_snippet(self,
                            sensor_type,
                            sensor_name):
            #TODO (Ankit): Technically doesn't make any sense to return a pc snippet
            # this will never be used. Instead this function should return pc frames
            # which is probably what the whole dataloader should become
            return None
        
        unified_snippets = NvidiaPhysicalAIUnifiedFrame()
        for sensor_type, sensor_list in self.str_to_sensor_map.items():
            for sensor_name in sensor_list:
                if sensor_type == "camera":
                    unified_snippets[sensor_type, sensor_name] = load_video_snippet(sensor_type, sensor_name)
                else:
                    unified_snippets[sensor_type, sensor_name] = load_pc_snippet(sensor_type, sensor_name)
        return unified_snippets
    
    # each clip is 20 sec long. for good performance of the embedding model,
    # need to return snippets that are X sec long. 
    def __iter__(self):
        if(self.snippet_counter == 0):
            chunk = self.clips_df[self.clip_counter]["chunk"].item()
            if(chunk > self.chunk_counter):
                self.delete_chunk()
                self.chunk_counter = chunk
                self.download_chunk()
            self.unified_clips = self.load_clips()

        unified_snippets = self.load_snippets()

        # increment clip + snippet counters
        self.snippet_counter += 1
        if(self.snippet_counter == self.snippets_per_clip):
            self.clip_counter += 1
            self.snippet_counter = 0
        
        yield unified_snippets
    