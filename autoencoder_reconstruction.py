import argparse
import sys
import os
from pathlib import Path
import torch
import json
import pandas as pd
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import torchaudio

from utils.basic_utils import auto_download_checkpoints, rand_fix, count_parameters, remove_weight_norms
from utils.audio_utils import load_audio

# --- Model Imports ---
# sr_noncausal_vae
from modules_noncausal.stable_audio_tools.models import create_model_from_config as create_model_sr_noncausal_vae
from modules_noncausal.stable_audio_tools.models.utils import load_ckpt_state_dict as load_ckpt_sr_noncausal_vae
from modules_noncausal.stable_audio_tools.utils.torch_common import copy_state_dict as copy_state_sr_noncausal_vae

# sr_causal_vae 
from modules_causal.stable_audio_tools.models import create_model_from_config as create_model_sr_causal_vae
from modules_causal.stable_audio_tools.models.utils import load_ckpt_state_dict as load_ckpt_sr_causal_vae
from modules_causal.stable_audio_tools.utils.torch_common import copy_state_dict as copy_state_sr_causal_vae

# soundctm_vae
from soundctm_vae.dac.model.dac import GaussianDAC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aac", ".m4a"}

def build_soundctm_vae(ckpt_path):
    stage1_path = auto_download_checkpoints(ckpt_path)
    kwargs = {
        "folder": f"{stage1_path}",
        "map_location": "cpu",
        "package": False,
    }
    vae, _ = GaussianDAC.load_from_folder(**kwargs)
    vae.__class__.remove_weight_norm = remove_weight_norms
    vae.eval()
    vae.remove_weight_norm()
    return vae

# --- Wrapper across soundctm_vae and sr_vae_variants---
class VAEWrapper(torch.nn.Module):
    def __init__(self, model, variant, chunk_duration=30.0):
        super().__init__()
        self.model = model
        self.variant = variant
        self.chunk_duration = chunk_duration
        
        if variant == 'soundctm_vae':
            self.target_sr = 44100
            self.target_channels = 1
        else:
            self.target_sr = 48000
            self.target_channels = 2
            
        self.samples_per_chunk = int(self.target_sr * self.chunk_duration)
    
    # --- Forward ---
    def _forward_one_chunk(self, wav_chunk: torch.Tensor):
        if self.variant == 'soundctm_vae':
            z = self.model.encode_to_latent(wav_chunk, sample_rate=self.target_sr)
            recon = self.model.decode_to_waveform(z)
            return recon
        else:
            z = self.model.encode(wav_chunk)
            return self.model.decode(z)

    def forward(self, wav: torch.Tensor):
        B, C, T = wav.shape
        
        if T <= self.samples_per_chunk:
            return self._forward_one_chunk(wav)
        
        chunks = []
        for t in range(0, T, self.samples_per_chunk):
            end = min(t + self.samples_per_chunk, T)
            wav_chunk = wav[:, :, t:end]

            try:
                recon_chunk = self._forward_one_chunk(wav_chunk)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"Chunk size {self.chunk_duration}s is still too large for VRAM.") from e
                raise e
            
            chunks.append(recon_chunk.cpu())
        
        return torch.cat(chunks, dim=-1).to(wav.device)


@torch.no_grad()
def process_batch(
    tasks: List[Tuple[Path, Path]],
    wrapper: VAEWrapper,
    device: torch.device,
    dtype: torch.dtype,
    target_duration: Optional[float] = None
):
    batch_wavs = []
    valid_tasks = [] 

    # Load Data
    for in_path, out_path in tasks:
        wav, _ = load_audio(
            in_path, 
            target_sr=wrapper.target_sr, 
            target_channels=wrapper.target_channels,
            target_duration=target_duration,
        )
        if wav is not None:
            batch_wavs.append(wav)
            valid_tasks.append((in_path, out_path, wav))

    if not batch_wavs:
        return

    max_len = max([w.size(1) for w in batch_wavs])
    padded_wavs = []
    for w in batch_wavs:
        if w.size(1) < max_len:
            pad = max_len - w.size(1)
            w = torch.nn.functional.pad(w, (0, pad))
        padded_wavs.append(w)
            
    batch_wavs_device = torch.stack(padded_wavs).to(device, dtype=dtype) # (B, C, T)

    # Model Inference
    try:
        recon_batch = wrapper(batch_wavs_device).to(dtype=torch.float32).cpu() # (B, C, T)
    except Exception as e:
        log.error(f"Inference failed for batch: {e}")
        return

    # Save Outputs
    for i, (in_path, out_path, original_wav) in enumerate(valid_tasks):
        reconstructed_wav = recon_batch[i]
        original_length = original_wav.size(1)
        
        if reconstructed_wav.size(1) > original_length:
            reconstructed_wav = reconstructed_wav[:, :original_length]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Format Handling
        suffix = out_path.suffix.lower()
        if suffix not in AUDIO_EXTS: 
            suffix = ".flac"
            out_path = out_path.with_suffix(".flac")
        save_fmt = suffix.replace(".", "")
        
        try:
            torchaudio.save(str(out_path), reconstructed_wav, wrapper.target_sr, format=save_fmt)
        except Exception:
            out_path_safe = out_path.with_suffix(".flac")
            torchaudio.save(str(out_path_safe), reconstructed_wav, wrapper.target_sr, format="flac")

def main(args):
    modes = sum([bool(args.input_file), bool(args.csv_file), (bool(args.input_dir) and bool(args.output_dir))])
    if modes != 1:
        log.error("Specify only one of input type: --input_file, --csv_file, or --input_dir.")
        sys.exit(1)
    if args.csv_file and not args.output_dir:
        log.error("CSV mode requires --output_dir.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    rand_fix(5031)
    
    # Load Model
    log.info(f"Loading Model Variant: {args.model_variant}")
    audio_vae = None
    
    if args.model_variant in ['sr_noncausal', 'sr_causal']:
        if not args.config: parser.error(f"--config required for {args.model_variant}")
        with open(args.config) as f: config = json.load(f)
        
        if args.model_variant == 'sr_noncausal':
            audio_vae = create_model_sr_noncausal_vae(config)
            copy_state_sr_noncausal_vae(audio_vae, load_ckpt_sr_noncausal_vae(args.ckpt))
        else:
            audio_vae = create_model_sr_causal_vae(config)
            copy_state_sr_causal_vae(audio_vae, load_ckpt_sr_causal_vae(args.ckpt))
    
    elif args.model_variant == 'soundctm_vae':
        audio_vae = build_soundctm_vae(args.ckpt)
        dtype = torch.float32  # soundctm_vae only works in float32
        log.info(f"fallback: {dtype}. soundctm_vae only works in float32.")

    audio_vae.to(device, dtype=dtype).eval().requires_grad_(False)
    count_parameters(audio_vae, args.model_variant)

    # Wrap VAE
    wrapped_vae = VAEWrapper(audio_vae, args.model_variant)
    
    # Prepare audio tracks to process
    all_samples: List[Tuple[Path, Path]] = []
    
    if args.input_file:
        in_path = Path(args.input_file)
        out_path = Path(args.output_file) if args.output_file else in_path.parent / f"{in_path.stem}_recon.flac"
        all_samples.append((in_path, out_path))
        args.batch_size = 1

    elif args.csv_file:
        df = pd.read_csv(args.csv_file)
        col = next((c for c in df.columns if c.lower() in ["filepath"]), None)
        if not col: raise ValueError("CSV needs a path column.")
        out_root = Path(args.output_dir)
        for p in df[col].astype(str):
            if Path(p).exists():
                all_samples.append((Path(p), out_root / f"{Path(p).stem}_recon.flac"))
        log.info(f"CSV: Found {len(all_samples)} files.")

    elif args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        all_exts = AUDIO_EXTS.union(VIDEO_EXTS)
        files = [p for p in in_dir.rglob("*") if p.suffix.lower() in all_exts]
        for f in files:
            out_path = (out_dir / f.relative_to(in_dir)).with_suffix(".flac")
            all_samples.append((f, out_path))
        log.info(f"Directory: Found {len(all_samples)} files.")

    log.info("Start inference...")
    
    target_duration = args.duration if args.duration > 0 else None
    
    total_batches = (len(all_samples) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(all_samples), args.batch_size), desc="Processing", total=total_batches):
        current_batch = all_samples[i:i + args.batch_size]
        process_batch(
            current_batch, 
            wrapped_vae, 
            device, 
            dtype=dtype,
            target_duration=target_duration
        )
    log.info("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE inference")
    
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--model_variant", type=str, required=True, choices=['sr_noncausal', 'sr_causal', 'soundctm_vae'])
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="ckpt")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--duration", type=float, default=-1.0)

    main(parser.parse_args())