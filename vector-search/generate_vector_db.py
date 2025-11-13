from dataloaders.frame_loader import NvidiaPhysicalAIDataloader
import yaml
import requests

def generate_payload(snippet_b64):
    return {
        "input": [
            f"data:video/webm;base64,{snippet_b64}"
        ],
        "request_type": "query",
        "encoding_format": "float",
        "model": "nvidia/cosmos-embed1"
    }

def main():
    with open('configs/nvidia_physical_ai.yaml', 'r') as file:
        dataloader_cfg = yaml.safe_load(file)

    dataloader = NvidiaPhysicalAIDataloader(dataloader_cfg)
    snippet_counter = 0
    for unified_snippet in dataloader.generate_snippets():
        print(f"========== SNIPPET {snippet_counter} ==========")
        for camera_name, snippet_b64 in unified_snippet.cameras.items():
            payload = generate_payload(snippet_b64)
            r = requests.post("http://localhost:8000/v1/embeddings", json=payload)
            print(f"{camera_name}: {r.json}")

if __name__ == "__main__":
    main()