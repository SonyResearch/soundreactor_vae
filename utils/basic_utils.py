import os
import random
import logging
import numpy as np
import torch
import requests
from urllib.parse import urljoin
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.weight_norm import WeightNorm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()



# --- Helper Functions & Classes ---

def rand_fix(seed):
    """Fixes random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def count_parameters(model, model_name="Model"):
    """Counts and logs the number of parameters in the encoder and decoder."""
    total_params = sum(p.numel() for p in model.parameters())
    
    enc_params = 0
    dec_params = 0
    other_params = 0

    # for soundctm_vae
    if hasattr(model, 'pre_conv') and hasattr(model, 'post_conv'):
        if hasattr(model, 'encoder'): enc_params += sum(p.numel() for p in model.encoder.parameters())
        if hasattr(model, 'pre_conv'): enc_params += sum(p.numel() for p in model.pre_conv.parameters())
        if hasattr(model, 'decoder'): dec_params += sum(p.numel() for p in model.decoder.parameters())
        if hasattr(model, 'post_conv'): dec_params += sum(p.numel() for p in model.post_conv.parameters())
    
    # for sr_noncausal_vae and sr_causal_vae
    elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        if model.encoder: enc_params += sum(p.numel() for p in model.encoder.parameters())
        if hasattr(model, 'bottleneck') and model.bottleneck: enc_params += sum(p.numel() for p in model.bottleneck.parameters())
        if model.decoder: dec_params += sum(p.numel() for p in model.decoder.parameters())
        if hasattr(model, 'pretransform') and model.pretransform: other_params += sum(p.numel() for p in model.pretransform.parameters())

    # Fallback
    else:
        if hasattr(model, 'encoder'): enc_params += sum(p.numel() for p in model.encoder.parameters())
        elif hasattr(model, 'enc'): enc_params += sum(p.numel() for p in model.enc.parameters())
        if hasattr(model, 'decoder'): dec_params += sum(p.numel() for p in model.decoder.parameters())
        elif hasattr(model, 'dec'): dec_params += sum(p.numel() for p in model.dec.parameters())

    log.info(f"--- {model_name} Parameter Count ---")
    log.info(f"Total:   {total_params / 1e6:.2f} M")
    log.info(f"Encoder: {enc_params / 1e6:.2f} M")
    log.info(f"Decoder: {dec_params / 1e6:.2f} M")
    if other_params > 0: log.info(f"Other:   {other_params / 1e6:.2f} M")
    log.info("------------------------------------")

# --- GaussianDAC Utils ---
def remove_weight_norms(self):
    for module in self.modules():
        if isinstance(module, WeightNorm):
            remove_weight_norm(module)

def download_checkpoint(url, local_path):
    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return
    print(f"Downloading {url} to {local_path} ...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")

def auto_download_checkpoints(ckpt_path):
    base_url = "https://huggingface.co/koichisaito/soundctm_dit/resolve/main/"
    print(f"Using base URL: {base_url}")
    vae_file = "gaussiandac/weights.pth"

    vae_local_path = os.path.join(ckpt_path, vae_file)
    download_checkpoint(urljoin(base_url, "utils_checkpoints/vae/" + vae_file), vae_local_path)
    stage1_path = ckpt_path
    
    return stage1_path