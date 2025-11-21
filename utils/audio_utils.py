import typing as tp
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torchaudio
import logging
from torio.io import StreamingMediaDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()


def save_audios(audios: torch.Tensor, filenames: tp.List[str], output_root: str, sr: int):

    bs, ch, L = audios.shape
    assert len(filenames) == bs

    for idx in range(bs):
        audio = audios[idx].to(torch.float).cpu()
        filename = filenames[idx]
        save_path = f"{output_root}/{filename}.wav"

        # mkdir
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(save_path, audio, sr, encoding='PCM_F')
        
        
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

def load_audio(
    path: Path, 
    target_sr: int, 
    target_channels: int,
    target_duration: Optional[float] = None,
) -> Tuple[Optional[torch.Tensor], int]:
    
    ext = path.suffix.lower()
    waveform = None
    
    # 1. Load Waveform
    try:
        waveform, sr_orig = torchaudio.load(str(path))
        if sr_orig != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr_orig, new_freq=target_sr,
                lowpass_filter_width=64, rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser", beta=14.769656459379492
            )
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        return None, -1

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        
    current_channels = waveform.size(0)
    if target_channels == 2:
        if current_channels == 1: waveform = waveform.repeat(2, 1)
        elif current_channels > 2: waveform = waveform[:2, :]
    elif target_channels == 1:
        if current_channels > 1: waveform = waveform.mean(dim=0, keepdim=True)

    length = waveform.size(1)
    
    if target_duration is not None and target_duration > 0:
        max_samples = int(target_sr * target_duration)
        if length < max_samples:
            pad = max_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad), mode="constant", value=0)
        else:
            waveform = waveform[:, :max_samples]
            
    return waveform, target_sr
