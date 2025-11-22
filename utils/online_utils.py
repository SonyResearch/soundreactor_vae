import time
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from modules_causal.stable_audio_tools.models.autoencoders import (
    AudioAutoencoder, 
    StreamingOobleckDecoder, OobleckDecoder,
    StreamingOobleckEncoder, OobleckEncoder
)
from modules_causal.stable_audio_tools.models.streaming_utils import CUDAGraphed

def _unwrap_compiled(mod: torch.nn.Module) -> torch.nn.Module:
    return getattr(mod, "_orig_mod", mod)

def _has_streaming_iface(mod: torch.nn.Module) -> bool:
    base = _unwrap_compiled(mod)
    return hasattr(mod, "streaming") or hasattr(base, "streaming")

def strip_weight_norm_streaming_(module: torch.nn.Module):
    for name, m in module.named_modules():
        if hasattr(m, "conv") and hasattr(m.conv, "weight"):
            try:
                nn_utils.remove_weight_norm(m.conv)
            except Exception:
                pass
        if hasattr(m, "convtr") and hasattr(m.convtr, "weight"):
            try:
                nn_utils.remove_weight_norm(m.convtr)
            except Exception:
                pass
    return module

def convert_streaming_inplace(
    ae: "AudioAutoencoder",
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    requires_grad: bool = False
):
    
    enc = ae.encoder
    assert isinstance(enc, OobleckEncoder), "encoder must be OobleckEncoder"
    assert enc.causal, "StreamingOobleckEncoder supports only causal encoders"
    
    dec = ae.decoder
    assert isinstance(dec, OobleckDecoder), "decoder must be OobleckDecoder"
    assert dec.causal, "StreamingOobleckDecoder supports only causal decoders"
    
    senc = StreamingOobleckEncoder(
        in_channels=enc.in_channels,
        channels=enc.channels,
        latent_dim=enc.latent_dim,
        c_mults=enc.input_c_mults,
        strides=enc.strides,
        use_snake=enc.use_snake,
        antialias_activation=enc.antialias_activation,
        causal=True
    )
    if hasattr(senc, "initialize_from_conversion"):
        senc.initialize_from_conversion(enc)
    else:
        senc.load_state_dict(enc.state_dict())

    strip_weight_norm_streaming_(senc) 
    
    sdec = StreamingOobleckDecoder(
        out_channels=dec.out_channels,
        channels=dec.channels,
        latent_dim=dec.latent_dim,
        c_mults=dec.input_c_mults,
        strides=dec.strides,
        use_snake=dec.use_snake,
        antialias_activation=dec.antialias_activation,
        use_nearest_upsample=dec.use_nearest_upsample,
        final_tanh=dec.final_tanh,
        causal=True,
    )
    if hasattr(sdec, "initialize_from_conversion"):
        sdec.initialize_from_conversion(dec)
    else:
        sdec.load_state_dict(dec.state_dict())

    strip_weight_norm_streaming_(sdec) 
    
    if dtype is None:
        try:
            dtype = next(enc.parameters()).dtype
        except StopIteration:
            dtype = torch.float32
            
    ae.encoder = senc.to(device=device, dtype=dtype).eval().requires_grad_(requires_grad)
    ae.decoder = sdec.to(device=device, dtype=dtype).eval().requires_grad_(requires_grad)

class OnlineVAEProcessor:
    def __init__(
        self,
        model,
        mode: str,  # 'reconstruction', 'encode', 'decode'
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        use_cuda_graph: bool = True,
        compile_warmup_steps: int = 3,
    ):
        self.model = model
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.use_cuda_graph = use_cuda_graph
        self.compile_warmup_steps = compile_warmup_steps

        self.ratio = int(model.downsampling_ratio)
        self.io_channels = model.io_channels

        self.latent_dim = getattr(model, "latent_dim", None) 
        if self.latent_dim is None and hasattr(model, "bottleneck"):
             self.latent_dim = model.bottleneck.latent_dim

        self.enc_stream_owner = None
        self.dec_stream_owner = None
        self._stream_cms = [] 
        
        if self.mode in ['reconstruction', 'encode']:
            enc = self.model.encoder
            assert _has_streaming_iface(enc), "Encoder must expose .streaming()"
            self.enc_stream_owner = enc if hasattr(enc, "streaming") else _unwrap_compiled(enc)

        if self.mode in ['reconstruction', 'decode']:
            dec = self.model.decoder
            assert _has_streaming_iface(dec), "Decoder must expose .streaming()"
            self.dec_stream_owner = dec if hasattr(dec, "streaming") else _unwrap_compiled(dec)

        # CUDA Graph State
        self._step_graphed: CUDAGraphed | None = None

        if self.mode == 'encode':
            self._step_func_base = self._step_encode_only
        elif self.mode == 'decode':
            self._step_func_base = self._step_decode_only
        else:
            self._step_func_base = self._step_full
            
        self._step = self._step_func_base

    def _enter_streaming(self):
        assert len(self._stream_cms) == 0, "Streaming already active"
        
        if self.enc_stream_owner:
            cm = self.enc_stream_owner.streaming(self.batch_size)
            cm.__enter__()
            self._stream_cms.append(cm)
            
        if self.dec_stream_owner:
            cm = self.dec_stream_owner.streaming(self.batch_size)
            cm.__enter__()
            self._stream_cms.append(cm)

    def _exit_streaming(self):
        while self._stream_cms:
            cm = self._stream_cms.pop()
            cm.__exit__(None, None, None)


    def _step_encode_only(self, x):
        return self.model.encode(x)

    def _step_decode_only(self, z):
        return self.model.decode(z)

    def _step_full(self, x):
        z = self.model.encode(x)
        rec = self.model.decode(z)
        return rec

    # --- CUDA Graph Setup ---

    def _capture_graph(self):
        use_cuda = (isinstance(self.device, torch.device) and self.device.type == "cuda")
        if not (self.use_cuda_graph and use_cuda and torch.cuda.is_available()):
            self._step = self._step_func_base
            return

        print(f"Capturing CUDA Graph for mode: {self.mode}...")
        
        if self.mode in ['reconstruction', 'encode']:
            dummy_in = torch.zeros(
                self.batch_size, self.io_channels, self.ratio,
                device=self.device, dtype=self.dtype
            )
        else: # decode
            dummy_in = torch.zeros(
                self.batch_size, self.latent_dim, 1,
                device=self.device, dtype=self.dtype
            )

        if self._step_graphed is None:
            self._step_graphed = CUDAGraphed(self._step_func_base, warmup_steps=self.compile_warmup_steps)
        else:
            self._step_graphed.reset(warmup_steps=self.compile_warmup_steps)

        # Warmup & Capture
        with torch.inference_mode():
            dummy_out = self._step_graphed(dummy_in)
            if use_cuda: torch.cuda.synchronize()
        
        self._step = self._step_graphed
        print("CUDA Graph captured.")
        
        print("Resetting streaming states after warmup...")
        self._reset_streaming_states()

    # --- Context Manager Interface ---

    def __enter__(self):
        self._enter_streaming()
        if self.use_cuda_graph:
            self._capture_graph()
        else:
            self._reset_streaming_states()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._step = self._step_func_base
        if self._step_graphed:
            self._step_graphed.reset(warmup_steps=0)
        self._exit_streaming()
        
    def _reset_streaming_states(self):
        reset_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        for mod in self.model.modules():
            if hasattr(mod, '_streaming_state') and mod._streaming_state is not None:
                if hasattr(mod._streaming_state, 'reset'):
                    mod._streaming_state.reset(reset_mask)

    # --- Public Methods ---

    @torch.no_grad()
    def encode_chunk(self, wav_chunk: torch.Tensor):
        if self.mode == 'decode': raise RuntimeError("Mode is decode")
        if wav_chunk.shape[-1] != self.ratio:
            raise ValueError(f"Input length must be {self.ratio}")
        
        if wav_chunk.dtype != self.dtype: wav_chunk = wav_chunk.to(self.dtype)
        return self._step(wav_chunk)

    @torch.no_grad()
    def decode_chunk(self, latent_chunk: torch.Tensor):
        if self.mode == 'encode': raise RuntimeError("Mode is encode")
        if latent_chunk.shape[-1] != 1:
            raise ValueError("Latent time-step must be 1")

        if latent_chunk.dtype != self.dtype: latent_chunk = latent_chunk.to(self.dtype)
        return self._step(latent_chunk)

    @torch.no_grad()
    def process_full_chunk(self, wav_chunk: torch.Tensor):
        if self.mode != 'reconstruction': raise RuntimeError("Mode is not reconstruction")
        if wav_chunk.shape[-1] != self.ratio:
            raise ValueError(f"Input length must be {self.ratio}")

        if wav_chunk.dtype != self.dtype: wav_chunk = wav_chunk.to(self.dtype)
        return self._step(wav_chunk)