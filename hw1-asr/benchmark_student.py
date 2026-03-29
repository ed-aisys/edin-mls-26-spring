#!/usr/bin/env python3
"""
Student Version Benchmark Script
Tests student implementations against expected output.

Usage:
    python benchmark_student.py <folder_name>
    python benchmark_student.py glm_asr_cutile_template
    python benchmark_student.py glm_asr_triton_example
    python benchmark_student.py glm_asr_scratch
    python benchmark_student.py <folder_name> --warmup-benchmarks 1 --benchmark-repeats 3
"""

import argparse
import time
import sys
import os
import numpy as np
import importlib


def get_attention_mode_label():
    """Return the template attention backend requested for this process."""
    return os.environ.get("GLM_ASR_ATTENTION_MODE", "auto")


def load_expected_text(audio_path):
    """
    Load expected transcription from a corresponding .txt file.
    Looks for a text file with the same name as the audio file (without extension).
    Extracts text after 'Expected transcription:' line.

    Args:
        audio_path: Path to the audio file (e.g., 'test_audio.wav')

    Returns:
        Expected transcription text, or None if file not found
    """
    if not audio_path:
        return None

    # Remove .wav, .flac, etc. extension and add .txt
    base_path = os.path.splitext(audio_path)[0]
    txt_path = base_path + '.txt'

    if not os.path.exists(txt_path):
        return None

    try:
        with open(txt_path, 'r') as f:
            content = f.read()

        # Find the "Expected transcription:" line
        if 'Expected transcription:' in content:
            # Get everything after "Expected transcription:"
            parts = content.split('Expected transcription:', 1)
            if len(parts) > 1:
                expected_text = parts[1].strip().split('\n')[0].strip()
                return expected_text
    except Exception as e:
        print(f"Warning: Could not read expected text from {txt_path}: {e}")

    return None


def download_librispeech_sample():
    """Download a LibriSpeech sample audio file."""
    import urllib.request
    import tarfile
    import io

    cache_dir = os.path.expanduser("~/.cache/glm_asr")
    os.makedirs(cache_dir, exist_ok=True)
    audio_path = os.path.join(cache_dir, "test_audio.flac")

    if os.path.exists(audio_path):
        return audio_path

    print("Downloading LibriSpeech sample...")
    # Use a direct FLAC file from OpenSLR
    url = "https://www.openslr.org/resources/12/test-clean/61/70968/61-70968-0000.flac"

    try:
        # Try downloading from a simpler source
        urllib.request.urlretrieve(url, audio_path)
        return audio_path
    except:
        return None


def load_test_audio(audio_path=None):
    """Load LibriSpeech test audio."""
    import wave
    import struct

    def read_wav(filepath):
        """Read wav file using standard library."""
        with wave.open(filepath, 'rb') as wav:
            sr = wav.getframerate()
            n_channels = wav.getnchannels()
            n_frames = wav.getnframes()
            sample_width = wav.getsampwidth()

            raw_data = wav.readframes(n_frames)

            if sample_width == 2:
                fmt = f'<{n_frames * n_channels}h'
                audio = np.array(struct.unpack(fmt, raw_data), dtype=np.float32)
                audio = audio / 32768.0
            elif sample_width == 4:
                fmt = f'<{n_frames * n_channels}i'
                audio = np.array(struct.unpack(fmt, raw_data), dtype=np.float32)
                audio = audio / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

            return audio, sr

    # If audio_path specified, use it directly
    if audio_path and os.path.exists(audio_path):
        audio_paths = [audio_path]
    else:
        # Try to load from common locations - prioritize local test_audio.wav
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_paths = [
            os.path.join(script_dir, "test_audio.wav"),  # Local test audio first
            "/tmp/test_audio.wav",
            os.path.expanduser("~/.cache/glm_asr/test_audio.wav"),
            os.path.expanduser("~/.cache/glm_asr/test_audio.flac"),
            "../test_audio.wav",
        ]

    audio_array = None
    sr = 16000
    used_audio_path = None

    for path in audio_paths:
        if os.path.exists(path):
            try:
                audio_array, sr = read_wav(path)
                used_audio_path = path
                print(f"Loaded audio from {path}")
                break
            except Exception as e:
                continue

    if audio_array is None:
        # Use synthetic test audio
        print("Using synthetic test audio (for structure validation only)")
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio_array = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio_array.astype(np.float32), "[synthetic]", duration

    # Try to load expected text from corresponding .txt file
    expected_text = None
    if used_audio_path:
        expected_text = load_expected_text(used_audio_path)

    # Fallback to default if not found
    if expected_text is None:
        expected_text = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"

    # Resample to 16kHz if needed
    target_sr = 16000
    if sr != target_sr:
        try:
            from scipy import signal
            num_samples = int(len(audio_array) * target_sr / sr)
            audio_array = signal.resample(audio_array, num_samples)
        except ImportError:
            # Simple linear interpolation
            old_indices = np.arange(len(audio_array))
            new_length = int(len(audio_array) * target_sr / sr)
            new_indices = np.linspace(0, len(audio_array) - 1, new_length)
            audio_array = np.interp(new_indices, old_indices, audio_array)

    duration = len(audio_array) / target_sr
    return audio_array.astype(np.float32), expected_text, duration


def benchmark_cutile_folder(folder_name, audio_array, num_warmup=1, num_runs=3):
    """Benchmark a CuTile implementation folder."""
    import cupy as cp

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Check required files
    required_files = ['__init__.py', 'model.py', 'layers.py', 'weight_loader.py']
    missing = [f for f in required_files if not os.path.exists(os.path.join(folder_path, f))]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    # Add folder to path
    sys.path.insert(0, folder_path)

    # Clear cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv', 'decode_attention']:
            del sys.modules[mod_name]

    # Apply version-specific configurations
    if 'example' in folder_name.lower():
        print("Applying baseline configuration (example)...")
        layers = importlib.import_module("layers")
        layers.Linear.BACKEND = 'cublas'
        layers.MLP.FUSED = False
        if hasattr(layers, 'AudioMLP'):
            layers.AudioMLP.FUSED = False
    print(f"Loading model from {folder_name}...")
    print(f"Attention mode: {get_attention_mode_label()}")
    from weight_loader import load_model_from_hf

    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    # Prepare inputs
    input_features, input_ids, input_features_mask = prepare_inputs(audio_array, processor)

    # Determine generate function
    generate_fn = model.generate
    if hasattr(model, 'generate_v8b'):
        generate_fn = model.generate_v8b
    elif hasattr(model, 'generate_v8'):
        generate_fn = model.generate_v8
    elif hasattr(model, 'generate_v6'):
        generate_fn = model.generate_v6

    print(f"Using generate function: {generate_fn.__name__}")

    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        try:
            _ = generate_fn(
                input_features, input_ids=input_ids, input_features_mask=input_features_mask,
                max_new_tokens=100, temperature=1.0, top_k=1
            )
        except TypeError:
            # Try without some arguments
            _ = generate_fn(
                input_features, input_ids=input_ids,
                max_new_tokens=100, temperature=1.0, top_k=1
            )
        cp.cuda.Device().synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        try:
            output = generate_fn(
                input_features, input_ids=input_ids, input_features_mask=input_features_mask,
                max_new_tokens=100, temperature=1.0, top_k=1
            )
        except TypeError:
            output = generate_fn(
                input_features, input_ids=input_ids,
                max_new_tokens=100, temperature=1.0, top_k=1
            )
        cp.cuda.Device().synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        tokens = output.shape[1] - input_ids.shape[1]
        print(f"  Run {i+1}: {elapsed:.1f}ms ({tokens} tokens)")

    # Decode output
    generated_np = cp.asnumpy(output)
    transcription = decode_output(generated_np, processor)

    # Clean up
    sys.path.remove(folder_path)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'transcription': transcription,
        'tokens': tokens
    }


def benchmark_triton_folder(folder_name, audio_array, num_warmup=1, num_runs=3):
    """Benchmark a Triton implementation folder (Torch tensors)."""
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    required_files = ['__init__.py', 'model.py', 'layers.py', 'weight_loader.py']
    missing = [f for f in required_files if not os.path.exists(os.path.join(folder_path, f))]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    sys.path.insert(0, folder_path)

    for mod_name in list(sys.modules.keys()):
        if mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv', 'decode_attention']:
            del sys.modules[mod_name]

    if 'example' in folder_name.lower():
        print("Applying baseline configuration (example)...")
        layers = importlib.import_module("layers")
        layers.Linear.BACKEND = 'cublas'
        layers.MLP.FUSED = False
        if hasattr(layers, 'EncoderMLP'):
            layers.EncoderMLP.FUSED = False

    print(f"Loading model from {folder_name}...")
    print(f"Attention mode: {get_attention_mode_label()}")
    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_features, input_ids, input_features_mask = prepare_inputs_torch(
        audio_array, processor, device
    )

    generate_fn = model.generate
    if hasattr(model, 'generate_v8b'):
        generate_fn = model.generate_v8b
    elif hasattr(model, 'generate_v8'):
        generate_fn = model.generate_v8
    elif hasattr(model, 'generate_v6'):
        generate_fn = model.generate_v6

    print(f"Using generate function: {generate_fn.__name__}")

    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            try:
                _ = generate_fn(
                    input_features, input_ids=input_ids, input_features_mask=input_features_mask,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )
            except TypeError:
                _ = generate_fn(
                    input_features, input_ids=input_ids,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            try:
                output = generate_fn(
                    input_features, input_ids=input_ids, input_features_mask=input_features_mask,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )
            except TypeError:
                output = generate_fn(
                    input_features, input_ids=input_ids,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        tokens = output.shape[1] - input_ids.shape[1]
        print(f"  Run {i+1}: {elapsed:.1f}ms ({tokens} tokens)")

    generated_np = output.detach().cpu().numpy()
    transcription = decode_output(generated_np, processor)

    sys.path.remove(folder_path)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'transcription': transcription,
        'tokens': tokens
    }


def benchmark_scratch_folder(folder_name, audio_array, num_warmup=1, num_runs=3):
    """Benchmark the scratch (PyTorch) implementation folder."""
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    sys.path.insert(0, folder_path)

    print(f"Loading PyTorch model from {folder_name}...")
    from torch_glm import load_model_and_processor

    model, processor = load_model_and_processor(dtype='float32')

    # Prepare inputs
    inputs = processor.apply_transcription_request(audio_array)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    input_features = inputs['input_features'].to(device, dtype=dtype)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    input_len = input_ids.shape[1]

    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False
            )
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        tokens = output.shape[1] - input_len
        print(f"  Run {i+1}: {elapsed:.1f}ms ({tokens} tokens)")

    # Decode output
    generated_ids = output[:, input_len:]
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    if isinstance(transcription, list):
        transcription = transcription[0]
    if "Please transcribe this audio into text" in transcription:
        transcription = transcription.split("Please transcribe this audio into text")[-1].strip()

    sys.path.remove(folder_path)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'transcription': transcription,
        'tokens': tokens
    }


def prepare_inputs(audio_array, processor):
    """Prepare inputs for CuTile model."""
    import cupy as cp
    if hasattr(processor, 'apply_transcription_request'):
        inputs = processor.apply_transcription_request(audio_array)
        input_features = cp.asarray(inputs.input_features.numpy(), dtype=cp.float32)
        input_ids = cp.asarray(inputs.input_ids.numpy(), dtype=cp.int64)
        input_features_mask = None
        if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
            input_features_mask = cp.asarray(inputs.input_features_mask.numpy(), dtype=cp.float32)
    else:
        # Manual processing
        features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
        input_features = cp.asarray(features['input_features'].numpy(), dtype=cp.float32)

        mel_frames = input_features.shape[-1]
        num_audio_tokens = max(1, mel_frames // 2 // 4)

        # Build input_ids
        user_token_id = 59253
        assistant_token_id = 59254
        begin_audio_token_id = 59261
        end_audio_token_id = 59262
        audio_token_id = 59260
        newline_token_id = 10
        prompt_token_ids = [9249, 70891, 419, 7122, 1119, 1467]

        input_ids_list = [user_token_id, newline_token_id, begin_audio_token_id]
        input_ids_list.extend([audio_token_id] * num_audio_tokens)
        input_ids_list.extend([end_audio_token_id, user_token_id, newline_token_id])
        input_ids_list.extend(prompt_token_ids)
        input_ids_list.extend([assistant_token_id, newline_token_id])

        input_ids = cp.array([input_ids_list], dtype=cp.int64)
        input_features_mask = None

    return input_features, input_ids, input_features_mask


def prepare_inputs_torch(audio_array, processor, device):
    """Prepare inputs for Triton model (Torch tensors)."""
    import torch
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

        mel_frames = input_features.shape[-1]
        num_audio_tokens = max(1, mel_frames // 2 // 4)

        user_token_id = 59253
        assistant_token_id = 59254
        begin_audio_token_id = 59261
        end_audio_token_id = 59262
        audio_token_id = 59260
        newline_token_id = 10
        prompt_token_ids = [9249, 70891, 419, 7122, 1119, 1467]

        input_ids_list = [user_token_id, newline_token_id, begin_audio_token_id]
        input_ids_list.extend([audio_token_id] * num_audio_tokens)
        input_ids_list.extend([end_audio_token_id, user_token_id, newline_token_id])
        input_ids_list.extend(prompt_token_ids)
        input_ids_list.extend([assistant_token_id, newline_token_id])

        input_ids = torch.tensor([input_ids_list], dtype=torch.int64, device=device)
        input_features_mask = None

    return input_features, input_ids, input_features_mask


def decode_output(generated_np, processor):
    """Decode output tokens to text."""
    try:
        if hasattr(processor, 'tokenizer'):
            transcription = processor.tokenizer.decode(generated_np[0], skip_special_tokens=True)
        elif hasattr(processor, 'decode'):
            transcription = processor.decode(generated_np[0], skip_special_tokens=True)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-ASR-Nano-2512", trust_remote_code=True)
            transcription = tokenizer.decode(generated_np[0], skip_special_tokens=True)

        if "Please transcribe this audio into text" in transcription:
            transcription = transcription.split("Please transcribe this audio into text")[-1].strip()
        return transcription
    except Exception as e:
        return f"[decode error: {e}]"


def check_transcription(transcription, expected):
    """Check if transcription matches expected text."""
    # Normalize both strings
    trans_norm = transcription.upper().strip()
    exp_norm = expected.upper().strip()

    # Remove punctuation
    import re
    trans_norm = re.sub(r'[^\w\s]', '', trans_norm)
    exp_norm = re.sub(r'[^\w\s]', '', exp_norm)

    # Check similarity
    trans_words = set(trans_norm.split())
    exp_words = set(exp_norm.split())

    if not exp_words:
        return True, 1.0

    overlap = len(trans_words & exp_words)
    accuracy = overlap / len(exp_words)

    return accuracy > 0.8, accuracy


def aggregate_benchmark_runs(results_list):
    """Aggregate multiple full benchmark passes after warmup."""
    means = np.array([result['mean'] for result in results_list], dtype=np.float64)
    stds = np.array([result['std'] for result in results_list], dtype=np.float64)
    tokens = np.array([result['tokens'] for result in results_list], dtype=np.float64)

    transcriptions = [result['transcription'] for result in results_list]
    transcription = transcriptions[-1]
    if transcriptions:
        counts = {}
        for item in transcriptions:
            counts[item] = counts.get(item, 0) + 1
        transcription = max(counts.items(), key=lambda item: (item[1], item[0] == transcriptions[-1]))[0]

    return {
        'mean': float(np.mean(means)),
        'std': float(np.std(means)),
        'inner_std_mean': float(np.mean(stds)),
        'min': float(np.min(means)),
        'max': float(np.max(means)),
        'transcription': transcription,
        'tokens': int(round(float(np.mean(tokens)))) if len(tokens) else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Student version benchmark')
    parser.add_argument('folder', type=str, help='Folder name to benchmark (e.g., glm_asr_cutile_template)')
    parser.add_argument('--audio', type=str, help='Path to test audio file (wav/flac)')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument(
        '--benchmark-repeats',
        type=int,
        default=1,
        help='Number of full benchmark passes to keep and aggregate',
    )
    parser.add_argument(
        '--warmup-benchmarks',
        type=int,
        default=0,
        help='Number of full benchmark passes to discard before aggregation',
    )
    args = parser.parse_args()

    if args.benchmark_repeats < 1:
        parser.error('--benchmark-repeats must be at least 1')
    if args.warmup_benchmarks < 0:
        parser.error('--warmup-benchmarks cannot be negative')

    print("=" * 70)
    print("GLM-ASR Student Version Benchmark")
    print("=" * 70)

    # Load test audio
    print("\nLoading test audio...")
    audio_array, expected, duration = load_test_audio(args.audio)
    print(f"Audio duration: {duration:.2f}s")
    print(f"Expected: {expected}")

    # Determine implementation type
    folder = args.folder
    is_scratch = 'scratch' in folder.lower()
    is_triton = 'triton' in folder.lower()

    print("\n" + "=" * 70)
    print(f"Testing: {folder}")
    print("=" * 70)

    try:
        if is_scratch:
            benchmark_fn = benchmark_scratch_folder
        elif is_triton:
            benchmark_fn = benchmark_triton_folder
        else:
            benchmark_fn = benchmark_cutile_folder

        total_passes = args.warmup_benchmarks + args.benchmark_repeats
        measured_results = []

        for pass_idx in range(total_passes):
            is_warmup = pass_idx < args.warmup_benchmarks
            label = (
                "warmup, discarded"
                if is_warmup
                else f"measured {len(measured_results) + 1}/{args.benchmark_repeats}"
            )
            print("\n" + "=" * 70)
            print(f"BENCHMARK PASS {pass_idx + 1}/{total_passes} ({label})")
            print("=" * 70)

            pass_results = benchmark_fn(folder, audio_array, args.warmup, args.runs)

            if not is_warmup:
                measured_results.append(pass_results)

        if not measured_results:
            raise ValueError("At least one measured benchmark pass is required.")

        if len(measured_results) == 1:
            results = measured_results[0]
        else:
            results = aggregate_benchmark_runs(measured_results)

        # Print results
        print("\n" + "=" * 70)
        if args.benchmark_repeats > 1 or args.warmup_benchmarks > 0:
            print(
                "AGGREGATED RESULTS "
                f"({len(measured_results)} measured benchmark passes, "
                f"{args.warmup_benchmarks} warmup discarded)"
            )
        else:
            print("RESULTS")
        print("=" * 70)
        print(f"Time: {results['mean']:.1f}ms (+/- {results['std']:.1f}ms)")
        print(f"Tokens: {results['tokens']}")
        print(f"Speed: {results['mean']/results['tokens']:.2f}ms/token")
        print(f"\nTranscription: {results['transcription']}")

        # Check correctness
        if expected != "[synthetic]":
            passed, accuracy = check_transcription(results['transcription'], expected)
            print(f"\nAccuracy: {accuracy*100:.1f}%")
            if passed:
                print("Status: PASS")
            else:
                print("Status: FAIL - Transcription does not match expected")
                print(f"Expected: {expected}")
        else:
            print("\nStatus: STRUCTURE OK (using synthetic audio, transcription not validated)")

        return 0 if (expected == "[synthetic]" or passed) else 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
