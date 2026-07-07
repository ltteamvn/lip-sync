import os
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Create directories
models_dir = "./models"
os.makedirs(f"{models_dir}/musetalk", exist_ok=True)
os.makedirs(f"{models_dir}/musetalkV15", exist_ok=True)
os.makedirs(f"{models_dir}/sd-vae", exist_ok=True)
os.makedirs(f"{models_dir}/whisper", exist_ok=True)
os.makedirs(f"{models_dir}/dwpose", exist_ok=True)
os.makedirs(f"{models_dir}/face-parse-bisent", exist_ok=True)

def download_file_from_url(url, dest_path):
    if os.path.exists(dest_path):
        print(f"Already exists: {dest_path}")
        return
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

# 1. Download MuseTalk weights from HF
print("Downloading MuseTalk V1.0 weights...")
hf_hub_download(repo_id="TMElyralab/MuseTalk", filename="musetalk/musetalk.json", local_dir=models_dir)
hf_hub_download(repo_id="TMElyralab/MuseTalk", filename="musetalk/pytorch_model.bin", local_dir=models_dir)

print("Downloading MuseTalk V1.5 weights...")
hf_hub_download(repo_id="TMElyralab/MuseTalk", filename="musetalkV15/musetalk.json", local_dir=models_dir)
hf_hub_download(repo_id="TMElyralab/MuseTalk", filename="musetalkV15/unet.pth", local_dir=models_dir)

# 2. Download SD VAE weights
print("Downloading Stable Diffusion VAE weights...")
hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse", filename="config.json", local_dir=f"{models_dir}/sd-vae")
hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse", filename="diffusion_pytorch_model.bin", local_dir=f"{models_dir}/sd-vae")

# 3. Download Whisper tiny weight from HuggingFace (For transformers WhisperModel compatibility)
print("Downloading Whisper tiny model...")
hf_hub_download(repo_id="openai/whisper-tiny", filename="config.json", local_dir=f"{models_dir}/whisper")
hf_hub_download(repo_id="openai/whisper-tiny", filename="preprocessor_config.json", local_dir=f"{models_dir}/whisper")
hf_hub_download(repo_id="openai/whisper-tiny", filename="pytorch_model.bin", local_dir=f"{models_dir}/whisper")

# 4. Download DWPose weights
print("Downloading DWPose weights...")
hf_hub_download(repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.pth", local_dir=f"{models_dir}/dwpose")

# 5. Download face-parse-bisent weights
print("Downloading face-parse-bisent weights...")
hf_hub_download(repo_id="ManyOtherFunctions/face-parse-bisent", filename="79999_iter.pth", local_dir=f"{models_dir}/face-parse-bisent")
hf_hub_download(repo_id="ManyOtherFunctions/face-parse-bisent", filename="resnet18-5c106cde.pth", local_dir=f"{models_dir}/face-parse-bisent")

print("All MuseTalk weights have been downloaded successfully!")
