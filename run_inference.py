# -*- coding: utf-8 -*-
import os
import sys
import torch

# Disable torch.compile to avoid MSVC compiler warm-up time and Triton errors
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Monkeypatch basicsr / torchvision functional_tensor
try:
    import torchvision
    import torchvision.transforms.functional as tv_F
    from types import ModuleType
    m = ModuleType("torchvision.transforms.functional_tensor")
    m.rgb_to_grayscale = tv_F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = m
except Exception as e:
    print(f"Basicsr monkeypatch failed: {e}")

from gui_app import config as cfg
from gui_app.worker import VideoWorker

params = {
    'sadtalker_root'   : cfg.sadtalker_root(),
    'checkpoint_path'  : cfg.checkpoint_path(),
    'ffmpeg_path'      : cfg.ffmpeg_path(),
    'output_dir'       : 'd:/nhep/results',
    'source_image'     : 'C:/Users/This PC/Downloads/z8015781174079_14ca09a9fd72e75585c37b2072568d4c.jpg',
    'mode'             : 'audio',
    'script'           : '',
    'voice'            : 'vi-VN-HoaiMyNeural',
    'rate'             : '+0%',
    'audio_file'       : 'C:/Users/This PC/Downloads/n.mp3',
    'bbox_shift'       : 0,
    'extra_margin'     : 10,
    'parsing_mode'     : 'raw',
    'left_cheek_width' : 90,
    'right_cheek_width': 90,
    'use_enhancer'     : True,
    'batch_size'       : 4,
    'device'           : 'cuda'
}

print("Starting MuseTalk generation process...")
print(f"Source Image: {params['source_image']}")
print(f"Audio File: {params['audio_file']}")
print(f"Output Directory: {params['output_dir']}")
print(f"Enhancer (GFPGAN): {params['use_enhancer']}")
print("----------------------------------------")

worker = VideoWorker(params)

# Connect signals to print logs in terminal
worker.log.connect(lambda msg: print(msg))
worker.progress.connect(lambda val: print(f"Progress: {val}%"))
worker.finished.connect(lambda path: print(f"\nSUCCESS! Video generated at: {path}"))
worker.error.connect(lambda err: print(f"\nERROR: {err}"))

# Run synchronously
worker.run()
