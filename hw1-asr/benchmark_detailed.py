#!/usr/bin/env python3
"""
Detailed Benchmark Script with Operator-level Profiling
Measures execution time for each operator/layer in the model.

Usage:
    python benchmark_detailed.py <folder_name>
    python benchmark_detailed.py glm_asr_cutile_template --nsys
    python benchmark_detailed.py glm_asr_triton_example
"""

import argparse
import time
import sys
import os
import numpy as np


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing."""

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.start_event = cp.cuda.Event()
        self.end_event = cp.cuda.Event()

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        self.end_event.synchronize()
        # CuPy uses get_elapsed_time instead of elapsed_time
        return self.cp.cuda.get_elapsed_time(self.start_event, self.end_event)


class TorchTimer:
    """Torch event-based timer for accurate GPU timing."""

    def __init__(self):
        import torch
        self.torch = torch
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None
            self._start_time = None

    def start(self):
        if self.start_event is not None:
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self):
        if self.start_event is not None:
            self.end_event.record()
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        elapsed = (time.perf_counter() - self._start_time) * 1000
        return elapsed



def detailed_profile(model, input_features, input_ids, input_features_mask, num_runs=3):
    """Detailed profiling of model components."""
    import cupy as cp

    results = {}
    timer = CUDATimer()

    print("\n" + "="*70)
    print("DETAILED OPERATOR PROFILING")
    print("="*70)

    # 1. Profile Audio Encoder
    print("\n[1/4] Profiling Audio Encoder...")
    encoder_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        audio_features = model.audio_encoder(input_features)
        elapsed = timer.stop()
        encoder_times.append(elapsed)
    results['audio_encoder'] = {
        'mean': np.mean(encoder_times),
        'std': np.std(encoder_times),
        'min': np.min(encoder_times),
        'max': np.max(encoder_times)
    }
    print(f"  Audio Encoder: {results['audio_encoder']['mean']:.2f}ms (+/- {results['audio_encoder']['std']:.2f}ms)")

    # 2. Profile Multi-modal Projector
    print("\n[2/4] Profiling Multi-modal Projector...")
    projector_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        projected = model.multi_modal_projector(audio_features)
        elapsed = timer.stop()
        projector_times.append(elapsed)
    results['projector'] = {
        'mean': np.mean(projector_times),
        'std': np.std(projector_times),
        'min': np.min(projector_times),
        'max': np.max(projector_times)
    }
    print(f"  Projector: {results['projector']['mean']:.2f}ms (+/- {results['projector']['std']:.2f}ms)")

    # 3. Profile Text Decoder (prefill phase)
    print("\n[3/4] Profiling Text Decoder (Prefill)...")

    # Build input embeddings
    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)

    # Find audio token positions
    audio_token_id = 59260
    audio_mask = (input_ids == audio_token_id)

    # Create combined embeddings
    combined_embeds = text_embeds.copy()
    if cp.any(audio_mask):
        audio_positions = cp.where(audio_mask[0])[0]
        num_audio_tokens = len(audio_positions)
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    prefill_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        # Call with inputs_embeds argument
        hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
        elapsed = timer.stop()
        prefill_times.append(elapsed)
    results['decoder_prefill'] = {
        'mean': np.mean(prefill_times),
        'std': np.std(prefill_times),
        'min': np.min(prefill_times),
        'max': np.max(prefill_times)
    }
    print(f"  Decoder Prefill: {results['decoder_prefill']['mean']:.2f}ms (+/- {results['decoder_prefill']['std']:.2f}ms)")

    # 4. Profile Decode Steps (autoregressive)
    print("\n[4/4] Profiling Decode Steps...")
    decode_times = []
    num_decode_steps = 10

    # Get logits and sample first token
    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = cp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    for step in range(num_decode_steps):
        cp.cuda.Device().synchronize()
        timer.start()

        # Single decode step
        next_embed = embed_tokens(next_token)
        step_hidden = model.text_decoder(inputs_embeds=next_embed)
        step_logits = model.lm_head(step_hidden)
        next_token = cp.argmax(step_logits[:, -1, :], axis=-1, keepdims=True)

        elapsed = timer.stop()
        decode_times.append(elapsed)

    results['decode_step'] = {
        'mean': np.mean(decode_times),
        'std': np.std(decode_times),
        'min': np.min(decode_times),
        'max': np.max(decode_times)
    }
    print(f"  Decode Step (avg): {results['decode_step']['mean']:.2f}ms (+/- {results['decode_step']['std']:.2f}ms)")

    # 5. Profile individual layers in decoder
    print("\n[5] Profiling Individual Decoder Layers...")
    layer_times = []

    try:
        test_input = combined_embeds
        seq_len = test_input.shape[1]

        # Try to get layers - different model versions have different structures
        if hasattr(model.text_decoder, 'layers'):
            layers = model.text_decoder.layers[:5]
        else:
            layers = []

        for i, layer in enumerate(layers):
            times = []
            for _ in range(num_runs):
                cp.cuda.Device().synchronize()
                timer.start()
                # Try calling with position_ids if needed
                try:
                    test_output = layer(test_input)
                except TypeError:
                    position_ids = cp.arange(seq_len, dtype=cp.int64).reshape(1, -1)
                    test_output = layer(test_input, position_ids=position_ids)
                elapsed = timer.stop()
                times.append(elapsed)

            layer_times.append({
                'name': f'layer_{i}',
                'mean': np.mean(times),
                'std': np.std(times)
            })
            print(f"  Layer {i}: {np.mean(times):.2f}ms (+/- {np.std(times):.2f}ms)")
            test_input = test_output
    except Exception as e:
        print(f"  Layer profiling skipped: {e}")

    results['layers'] = layer_times

    return results


def detailed_profile_torch(model, input_features, input_ids, input_features_mask, num_runs=3):
    """Detailed profiling of model components (Torch)."""
    import torch

    results = {}
    timer = TorchTimer()

    print("\n" + "="*70)
    print("DETAILED OPERATOR PROFILING (TORCH)")
    print("="*70)

    print("\n[1/4] Profiling Audio Encoder...")
    encoder_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        audio_features = model.audio_encoder(input_features)
        elapsed = timer.stop()
        encoder_times.append(elapsed)
    results['audio_encoder'] = {
        'mean': np.mean(encoder_times),
        'std': np.std(encoder_times),
        'min': np.min(encoder_times),
        'max': np.max(encoder_times)
    }
    print(f"  Audio Encoder: {results['audio_encoder']['mean']:.2f}ms (+/- {results['audio_encoder']['std']:.2f}ms)")

    print("\n[2/4] Profiling Multi-modal Projector...")
    projector_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        projected = model.multi_modal_projector(audio_features)
        elapsed = timer.stop()
        projector_times.append(elapsed)
    results['projector'] = {
        'mean': np.mean(projector_times),
        'std': np.std(projector_times),
        'min': np.min(projector_times),
        'max': np.max(projector_times)
    }
    print(f"  Projector: {results['projector']['mean']:.2f}ms (+/- {results['projector']['std']:.2f}ms)")

    print("\n[3/4] Profiling Text Decoder (Prefill)...")
    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)

    audio_token_id = 59260
    audio_mask = (input_ids == audio_token_id)

    combined_embeds = text_embeds.clone()
    if torch.any(audio_mask):
        audio_positions = torch.where(audio_mask[0])[0]
        num_audio_tokens = int(audio_positions.numel())
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    prefill_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
        elapsed = timer.stop()
        prefill_times.append(elapsed)
    results['decoder_prefill'] = {
        'mean': np.mean(prefill_times),
        'std': np.std(prefill_times),
        'min': np.min(prefill_times),
        'max': np.max(prefill_times)
    }
    print(f"  Decoder Prefill: {results['decoder_prefill']['mean']:.2f}ms (+/- {results['decoder_prefill']['std']:.2f}ms)")

    print("\n[4/4] Profiling Decode Steps...")
    decode_times = []
    num_decode_steps = 10

    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    for _ in range(num_decode_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        next_embed = embed_tokens(next_token)
        step_hidden = model.text_decoder(inputs_embeds=next_embed)
        step_logits = model.lm_head(step_hidden)
        next_token = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
        elapsed = timer.stop()
        decode_times.append(elapsed)

    results['decode_step'] = {
        'mean': np.mean(decode_times),
        'std': np.std(decode_times),
        'min': np.min(decode_times),
        'max': np.max(decode_times)
    }
    print(f"  Decode Step (avg): {results['decode_step']['mean']:.2f}ms (+/- {results['decode_step']['std']:.2f}ms)")

    print("\n[5] Profiling Individual Decoder Layers...")
    layer_times = []

    try:
        test_input = combined_embeds
        seq_len = test_input.shape[1]

        if hasattr(model.text_decoder, 'layers'):
            layers = model.text_decoder.layers[:5]
        else:
            layers = []

        for i, layer in enumerate(layers):
            times = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.start()
                try:
                    test_output = layer(test_input)
                except TypeError:
                    position_ids = torch.arange(seq_len, dtype=torch.int64, device=test_input.device).reshape(1, -1)
                    test_output = layer(test_input, position_ids=position_ids)
                elapsed = timer.stop()
                times.append(elapsed)

            layer_times.append({
                'name': f'layer_{i}',
                'mean': np.mean(times),
                'std': np.std(times)
            })
            print(f"  Layer {i}: {np.mean(times):.2f}ms (+/- {np.std(times):.2f}ms)")
            test_input = test_output
    except Exception as e:
        print(f"  Layer profiling skipped: {e}")

    results['layers'] = layer_times

    return results



def print_summary(component_results):
    """Print a summary table of all profiling results."""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    print("\n{:<35} {:>12} {:>12}".format("Component", "Time (ms)", "% of Total"))
    print("-"*60)

    # Calculate total time
    total = 0
    if component_results:
        for key in ['audio_encoder', 'projector', 'decoder_prefill']:
            if key in component_results:
                total += component_results[key]['mean']
        # Add estimated decode time (50 steps)
        if 'decode_step' in component_results:
            total += component_results['decode_step']['mean'] * 50

    if component_results:
        for key, label in [
            ('audio_encoder', 'Audio Encoder'),
            ('projector', 'Multi-modal Projector'),
            ('decoder_prefill', 'Decoder (Prefill)'),
        ]:
            if key in component_results:
                t = component_results[key]['mean']
                pct = (t / total * 100) if total > 0 else 0
                print(f"{label:<35} {t:>10.2f}ms {pct:>10.1f}%")

        if 'decode_step' in component_results:
            t = component_results['decode_step']['mean'] * 50
            pct = (t / total * 100) if total > 0 else 0
            print(f"{'Decoder (50 decode steps)':<35} {t:>10.2f}ms {pct:>10.1f}%")

    print("-"*60)
    print(f"{'TOTAL (estimated for 50 tokens)':<35} {total:>10.2f}ms")


def run_nsys_profile(folder, audio_path=None, runs=1):
    """Run Nsight Systems profiling."""
    import subprocess

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_name = f"profile_{folder}"

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--output", output_name,
        "--force-overwrite", "true",
        "python", os.path.join(script_dir, "benchmark_student.py"),
        folder, "--warmup", "1", "--runs", str(runs)
    ]

    if audio_path:
        cmd.extend(["--audio", audio_path])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=script_dir)
    print(f"\nProfile saved to: {output_name}.nsys-rep")
    print("Open with: nsys-ui " + output_name + ".nsys-rep")


def main():
    parser = argparse.ArgumentParser(description='Detailed operator profiling')
    parser.add_argument('folder', type=str, nargs='?', default='glm_asr_cutile_example',
                       help='Folder name to benchmark')
    parser.add_argument('--audio', type=str, help='Path to test audio file')
    parser.add_argument('--runs', type=int, default=3, help='Number of profiling runs')
    parser.add_argument('--nsys', action='store_true', help='Run Nsight Systems profiling')
    args = parser.parse_args()

    print("="*70)
    print("GLM-ASR Detailed Operator Profiling")
    print("="*70)

    # Run nsys if requested
    if args.nsys:
        run_nsys_profile(args.folder, args.audio, args.runs)
        return 0

    use_torch_backend = 'triton' in args.folder.lower()

    # Full profiling

    # Load test audio
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("\nLoading test audio...")
    audio_path = args.audio or os.path.join(script_dir, "test_audio.wav")

    import wave
    import struct

    with wave.open(audio_path, 'rb') as wav:
        sr = wav.getframerate()
        n_frames = wav.getnframes()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw_data = wav.readframes(n_frames)

        if sample_width == 2:
            audio_array = np.array(struct.unpack(f'<{n_frames * n_channels}h', raw_data), dtype=np.float32)
            audio_array = audio_array / 32768.0
        else:
            audio_array = np.zeros(n_frames, dtype=np.float32)

        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

    print(f"Audio: {len(audio_array)/sr:.2f}s @ {sr}Hz")

    # Load model
    folder_path = os.path.join(script_dir, args.folder)
    sys.path.insert(0, folder_path)

    # Clear cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv']:
            del sys.modules[mod_name]

    print(f"\nLoading model from {args.folder}...")
    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    if use_torch_backend:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(processor, 'apply_transcription_request'):
            inputs = processor.apply_transcription_request(audio_array)
            input_features = inputs.input_features.to(device=device, dtype=torch.float32)
            input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)
            input_features_mask = None
            if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
                input_features_mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)
        else:
            features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
            input_features = features['input_features'].to(device=device, dtype=torch.float32)
            input_ids = torch.tensor([[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]],
                                     dtype=torch.int64, device=device)
            input_features_mask = None

        print(f"Input features shape: {input_features.shape}")
        print(f"Input IDs shape: {input_ids.shape}")

        component_results = detailed_profile_torch(model, input_features, input_ids, input_features_mask, num_runs=args.runs)
    else:
        import cupy as cp
        if hasattr(processor, 'apply_transcription_request'):
            inputs = processor.apply_transcription_request(audio_array)
            input_features = cp.asarray(inputs.input_features.numpy(), dtype=cp.float32)
            input_ids = cp.asarray(inputs.input_ids.numpy(), dtype=cp.int64)
            input_features_mask = None
            if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
                input_features_mask = cp.asarray(inputs.input_features_mask.numpy(), dtype=cp.float32)
        else:
            features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
            input_features = cp.asarray(features['input_features'].numpy(), dtype=cp.float32)
            input_ids = cp.array([[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]], dtype=cp.int64)
            input_features_mask = None

        print(f"Input features shape: {input_features.shape}")
        print(f"Input IDs shape: {input_ids.shape}")

        component_results = detailed_profile(model, input_features, input_ids, input_features_mask, num_runs=args.runs)

    # Print summary
    print_summary(component_results)

    sys.path.remove(folder_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
