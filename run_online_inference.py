import argparse
import torch
import torchaudio
import os
import json
import sys
from tqdm import tqdm
from pathlib import Path

sys.path.append(".") 

from modules_causal.stable_audio_tools.models import create_model_from_config
from modules_causal.stable_audio_tools.models.utils import load_ckpt_state_dict
from modules_causal.stable_audio_tools.utils.torch_common import copy_state_dict

from utils.basic_utils import remove_weight_norms, count_parameters
from utils.audio_utils import load_audio
from utils.online_utils import OnlineVAEProcessor, convert_streaming_inplace

torch.backends.cuda.enable_flash_sdp = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

logging = sys.modules.get("logging")
if logging:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger()

def load_model(config_path, ckpt_path, device, dtype):
    with open(config_path) as f:
        config = json.load(f)
    model = create_model_from_config(config)
    if ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(ckpt_path))
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model


def main(args):
    torch.manual_seed(5031)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = getattr(torch, args.dtype)
    latents_preloaded = None
    if args.mode == 'decode':
        print(f"Inspecting input file: {args.input_file}")
        try:
            latents_preloaded = torch.load(args.input_file, map_location='cpu')
            
            if isinstance(latents_preloaded, torch.Tensor):
                input_dtype = latents_preloaded.dtype
                if input_dtype != target_dtype:
                    print(f"INFO: Input latents are {input_dtype}. Overwriting target_dtype from {target_dtype} to {input_dtype}.")
                    target_dtype = input_dtype
            else:
                print("Warning: Loaded file is not a Tensor. Proceeding with default settings.")
                
        except Exception as e:
            print(f"Error: Failed to load input file for inspection. {e}")
            sys.exit(1)

    print(f"Loading model... Mode: {args.mode}, dtype: {target_dtype}")

    model = load_model(args.config, args.ckpt, device, target_dtype)
    ratio = model.downsampling_ratio
    convert_streaming_inplace(model, device=device, dtype=target_dtype)
    count_parameters(model, 'sr_causal')

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    processor = OnlineVAEProcessor(
        model, args.mode, 
        batch_size=1, 
        device=device, 
        dtype=target_dtype, 
        use_cuda_graph=args.use_cuda_graph,
    )

    if args.mode == 'encode':
        print(f"Encoding {args.input_file}...")
        wav, sr = load_audio(
            Path(args.input_file), 
            target_sr=model.sample_rate, 
            target_channels=2, 
            target_duration=args.target_duration
        )
        wav = wav.to(device)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)  # [1, C, T]
        # Padding to multiple of ratio
        if wav.shape[-1] % ratio != 0:
            pad = ratio - (wav.shape[-1] % ratio)
            wav = torch.nn.functional.pad(wav, (0, pad))

        latents_list = []
        
        with processor:
            for i in tqdm(range(0, wav.shape[-1], ratio)):
                chunk = wav[:, :, i : i + ratio]
                z = processor.encode_chunk(chunk)
                latents_list.append(z.detach().cpu())

        full_latents = torch.cat(latents_list, dim=-1)
        full_latents_to_save = full_latents.to(target_dtype).clone().contiguous()
        save_path = os.path.join(args.output_dir, f"{base_name}_latents.pt")
        torch.save(full_latents_to_save, save_path)
        print(f"saved latents: {save_path}")
        
        original_file_size = os.path.getsize(Path(args.input_file)) / 1024 / 1024
        file_size = os.path.getsize(save_path) / 1024 / 1024
        print(f"  - shape: {full_latents_to_save.shape}")
        print(f"  - dtype: {full_latents_to_save.dtype}")
        print(f"  - original file Size: {original_file_size:.2f} MB")
        print(f"  - saved file Size: {file_size:.2f} MB")

    elif args.mode == 'decode':
        print(f"Decoding {args.input_file}...")
        latents = latents_preloaded.to(device)
        if latents.dim() == 2: latents = latents.unsqueeze(0)
        
        
        T_latent = latents.shape[-1]
        output_wavs = []

        with processor:
            for t in tqdm(range(T_latent)):
                z_chunk = latents[:, :, t : t + 1]
                wav_chunk = processor.decode_chunk(z_chunk) # -> [1, C, ratio]
                output_wavs.append(wav_chunk.detach().cpu().float())

        full_recon = torch.cat(output_wavs, dim=-1)
        save_path = os.path.join(args.output_dir, f"{base_name}_decoded.flac")
        torchaudio.save(save_path, full_recon.squeeze(0), model.sample_rate)
        print(f"Saved audio: {save_path}")

    else:
        print(f"Full reconstruction {args.input_file}...")
        wav, sr = load_audio(
            Path(args.input_file), 
            target_sr=model.sample_rate, 
            target_channels=2, 
            target_duration=args.target_duration
        )
        wav = wav.to(device)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)  # [1, C, T]
        original_length = wav.shape[-1]
        if wav.shape[-1] % ratio != 0:
            pad = ratio - (wav.shape[-1] % ratio)
            wav = torch.nn.functional.pad(wav, (0, pad))

        output_wavs = []
        with processor:
            for i in tqdm(range(0, wav.shape[-1], ratio)):
                chunk = wav[:, :, i : i + ratio]
                rec_chunk = processor.process_full_chunk(chunk)
                output_wavs.append(rec_chunk.detach().cpu().float())

        full_recon = torch.cat(output_wavs, dim=-1)
        
        if full_recon.shape[-1] > original_length:
            full_recon = full_recon[..., :original_length]
        save_path = os.path.join(args.output_dir, f"{base_name}_full_recon.flac")
        torchaudio.save(save_path, full_recon.squeeze(0), model.sample_rate)
        print(f"Saved audio: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online VAE Inference (Single File)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file (.wav/.flac for encode/reconstruction, .pt for decode)")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--target_duration", type=float, default=5, help="audio duration")
    parser.add_argument("--config", type=str, default="modules_causal/stable_audio_tools/configs/model_configs/autoencoders/sr_ds_1600_dim_64_causal.json")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--mode", type=str, default="reconstruction", choices=["reconstruction", "encode", "decode"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use_cuda_graph", action="store_true")
    
    args = parser.parse_args()

    main(args)