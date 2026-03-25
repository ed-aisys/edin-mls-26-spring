#!/usr/bin/env python3
"""
Nsight Compute profiling — single inference pass for per-kernel metrics.
Run with: ncu --metrics ... python3 ncu_profile.py
"""

import os
import sys
import torch

os.environ["HF_HOME"] = "/home/s2884198/.cache/huggingface"

print("Loading model...")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "glm_asr_triton_template"))
from glm_asr_triton_template.weight_loader import load_model_from_hf

model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

import soundfile as sf
audio, sr = sf.read("test_audio.wav")
audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
if torch.cuda.is_available():
    audio_tensor = audio_tensor.cuda()

print("Warming up (2 runs)...")
for _ in range(2):
    with torch.no_grad():
        if hasattr(model, 'generate_v8b'):
            model.generate_v8b(audio_tensor, max_new_tokens=13)
        else:
            model.generate(audio_tensor, max_new_tokens=13)

torch.cuda.synchronize()
print("Running profiled inference...")

with torch.no_grad():
    if hasattr(model, 'generate_v8b'):
        output = model.generate_v8b(audio_tensor, max_new_tokens=13)
    else:
        output = model.generate(audio_tensor, max_new_tokens=13)

torch.cuda.synchronize()
print("Done.")
