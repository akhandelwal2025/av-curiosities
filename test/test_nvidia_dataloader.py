from dataloaders.frame_loader import NvidiaPhysicalAIDataloader
import yaml

with open('configs/nvidia_physical_ai.yaml', 'r') as file:
    dataloader_cfg = yaml.safe_load(file)

dataloader = NvidiaPhysicalAIDataloader(dataloader_cfg)
for unified_feature in dataloader.generate_snippets():
    breakpoint()
