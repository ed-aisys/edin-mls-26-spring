#!/usr/bin/env python3
"""
Nsight Compute profiling — single inference pass for per-kernel metrics.
"""

import os
import sys
import torch

os.environ["HF_HOME"] = "/home/s2884198/.cache/huggingface"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Loading model...")
from glm_asr_triton_template.weight_loader import load_model_from_hf

model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

# Load audio the same way benchmark_student.py does
import soundfile as sf
audio, sr = sf.read("test_audio.wav")

inputs = processor.apply_transcription_request(audio)
device = "cuda" if torch.cuda.is_available() else "cpu"
input_features = inputs['input_features'].to(device, dtype=torch.float32)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs.get('attention_mask')
if attention_mask is not None:
    attention_mask = attention_mask.to(device)

print(f"input_features shape: {input_features.shape}")
print(f"input_ids shape: {input_ids.shape}")

print("Warming up (2 runs)...")
for _ in range(2):
    with torch.no_grad():
        if hasattr(model, 'generate_v8b'):
            model.generate_v8b(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=13,
                do_sample=False
            )
        else:
            model.generate(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=13,
                do_sample=False
            )

torch.cuda.synchronize()
print("Running profiled inference...")

with torch.no_grad():
    if hasattr(model, 'generate_v8b'):
        output = model.generate_v8b(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=13,
            do_sample=False
        )
    else:
        output = model.generate(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=13,
            do_sample=False
        )

torch.cuda.synchronize()
print("Done.")
