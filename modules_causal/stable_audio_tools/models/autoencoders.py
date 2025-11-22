from contextlib import nullcontext
import math
import typing as tp

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torchaudio import transforms as T
from alias_free_torch import Activation1d
# from dac.nn.layers import WNConv1d, WNConvTranspose1d
from einops import rearrange


# from ..inference.sampling import sample
from ..inference.utils import prepare_audio
from .blocks import SnakeBeta
from .bottleneck import Bottleneck, DiscreteBottleneck
# from .diffusion import ConditionedDiffusionModel, DAU1DCondWrapper, UNet1DCondWrapper, DiTWrapper
from .factory import create_pretransform_from_config, create_bottleneck_from_config
from .pretransforms import Pretransform
from .adp import Conv1d, ConvTranspose1d
from .streaming import StreamingContainer
from .streaming_utils import convert_wnconv1d_to_streamingconv1d, convert_wnconvtranspose1d_to_streamingconvtranspose1d
from .streaming_conv import StreamingConv1d, StreamingConvTranspose1d, FastStreamingConvTranspose1d

def WNConv1d(*args, **kwargs):
    if "causal" in kwargs and kwargs["causal"]:
        return weight_norm(Conv1d(*args, **kwargs))
    else:
        if "causal" in kwargs: del kwargs["causal"]  # Remove causal argument if it exists, as Conv1d does not support it
        return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    if "causal" in kwargs and kwargs["causal"]:
        return weight_norm(ConvTranspose1d(*args, **kwargs))
    else:
        if "causal" in kwargs: del kwargs["causal"]
        return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def get_activation(activation: tp.Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

class ResidualUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, use_snake: bool = False, antialias_activation: bool = False, causal: bool = False,
        ):
        super().__init__()
        self.dilation = dilation
        self.causal = causal
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation

        if self.causal:
            padding = 0
        else:
            padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation, padding=padding, causal=causal),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, causal=causal)
        )

    def forward(self, x):
        res = x

        # Disable checkpoint until tensor mismatch is fixed
        # x = checkpoint(self.layers, x)
        x = self.layers(x)

        return x + res

class StreamingResidualUnit(StreamingContainer):

    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False, causal=False):
        super().__init__()
        self.dilation = dilation
        self.causal = causal
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation

        if not self.causal:
            raise ValueError("StreamingResidualUnit does not support non-causal convolutions yet.")
        
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            StreamingConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, causal=causal, norm="weight_norm"),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            StreamingConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, causal=causal, norm="weight_norm")
        )

    def initialize_from_conversion(self, residual_unit: ResidualUnit):
        """Initialize the streaming residual unit from a regular residual unit."""
        self.in_channels = residual_unit.in_channels
        self.out_channels = residual_unit.out_channels
        self.dilation = residual_unit.dilation
        self.use_snake = residual_unit.use_snake
        self.antialias_activation = residual_unit.antialias_activation

        self.layers[0] = get_activation("snake" if self.use_snake else "elu", antialias=self.antialias_activation, channels=self.out_channels)
        # copy snake parameters, as these are learned
        if self.use_snake:
            # just copy the alpha and beta parameters
            with torch.no_grad():
                self.layers[0].alpha.copy_(residual_unit.layers[0].alpha)
                self.layers[0].beta.copy_(residual_unit.layers[0].beta)
        self.layers[1] = convert_wnconv1d_to_streamingconv1d(residual_unit.layers[1], StreamingConv1d)
        self.layers[2] = get_activation("snake" if self.use_snake else "elu", antialias=self.antialias_activation, channels=self.out_channels)
        if self.use_snake:
            # just copy the alpha and beta parameters
            with torch.no_grad():
                self.layers[2].alpha.copy_(residual_unit.layers[2].alpha)
                self.layers[2].beta.copy_(residual_unit.layers[2].beta)
        self.layers[3] = convert_wnconv1d_to_streamingconv1d(residual_unit.layers[3], StreamingConv1d)

    def forward(self, x):
        res = x
        
        if self.training:
            # TODO: check if checkpointing can work here, would be nice to have
            x = self.layers(x)
        else:
            x = self.layers(x)

        return x + res


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        return rearrange(x, '... a b -> ... b a')

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, use_snake: bool = False, antialias_activation: bool = False, causal: bool = False):
        super().__init__()
        
        from .dac import SConv1d
        self.causal = causal
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation

        # if self.causal:
        #     padding = 0
        # else:
        #     padding = math.ceil(stride/2)

        layers: list[nn.Module] = [
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake, causal=causal),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake, causal=causal),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake, causal=causal),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            # SConv1d(in_channels=in_channels, out_channels=out_channels,
            #          kernel_size=2 * stride, stride=stride),
            # WNConv1d(in_channels=in_channels, out_channels=out_channels,
            #          kernel_size=2 * stride, stride=stride, padding=padding, causal=causal),
        ]
        
        if causal:
            conv_down = WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=0,
                causal=True,
            )
        else:
            conv_down = SConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
            )
        layers.append(conv_down)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class StreamingEncoderBlock(StreamingContainer):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, causal=False):
        super().__init__()
        self.causal = causal
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation

        if not self.causal:
            raise ValueError("StreamingEncoderBlock does not support non-causal convolutions yet.")
        
        self.layers = nn.Sequential(
            StreamingResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake, causal=causal),
            StreamingResidualUnit(in_channels=in_channels,
                            out_channels=in_channels, dilation=3, use_snake=use_snake, causal=causal),
            StreamingResidualUnit(in_channels=in_channels,
                            out_channels=in_channels, dilation=9, use_snake=use_snake, causal=causal),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            StreamingConv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=2*stride, stride=stride, causal=causal, norm="weight_norm"),
        )

    def forward(self, x):
        return self.layers(x)

    def initialize_from_conversion(self, encoder_block: EncoderBlock):
        """Initialize the streaming encoder block from a regular encoder block."""
        self.in_channels = encoder_block.in_channels
        self.out_channels = encoder_block.out_channels
        self.stride = encoder_block.stride
        self.use_snake = encoder_block.use_snake
        self.antialias_activation = encoder_block.antialias_activation

        self.layers[0].initialize_from_conversion(encoder_block.layers[0])
        self.layers[1].initialize_from_conversion(encoder_block.layers[1])
        self.layers[2].initialize_from_conversion(encoder_block.layers[2])
        self.layers[3] = get_activation("snake" if self.use_snake else "elu", antialias=self.antialias_activation, channels=self.in_channels)
        if self.use_snake:
            # just copy the alpha and beta parameters
            with torch.no_grad():
                self.layers[3].alpha.copy_(encoder_block.layers[3].alpha)
                self.layers[3].beta.copy_(encoder_block.layers[3].beta)
        self.layers[4] = convert_wnconv1d_to_streamingconv1d(encoder_block.layers[4], StreamingConv1d)


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int,
        use_snake: bool = False, antialias_activation: bool = False, 
        use_nearest_upsample: bool = False, causal: bool = False
    ):
        super().__init__()
        self.causal = causal
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation
        self.use_nearest_upsample = use_nearest_upsample
        # if self.causal:
        #     padding = 0
        # else:
        #     padding = math.ceil(stride/2)

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                        out_channels=out_channels, 
                        kernel_size=2*stride,
                        stride=1,
                        bias=False, causal=causal)
            )
        else:
            if causal:
                upsample_layer = WNConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2*stride, stride=stride, padding=0, causal=causal
                )
            else:
                from .dac import SConvTranspose1d
                upsample_layer = SConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride, stride=stride,
                    )
        layers: list[nn.Module] = [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake, causal=causal),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake, causal=causal),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake, causal=causal),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class StreamingDecoderBlock(StreamingContainer):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False, causal=False):
        super().__init__()
        self.causal = causal
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation
        self.use_nearest_upsample = use_nearest_upsample

        if not self.causal:
            raise ValueError("StreamingDecoderBlock does not support non-causal convolutions yet.")

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                StreamingConv1d(in_channels=in_channels,
                        out_channels=out_channels, 
                        kernel_size=2*stride,
                        stride=1,
                        bias=False, causal=causal, norm="weight_norm")
            )
        else:
            upsample_layer = StreamingConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2*stride, stride=stride, causal=causal, norm="weight_norm"
            )
            
            # upsample_layer = FastStreamingConvTranspose1d(
            #     in_channels=in_channels,
            #     out_channels=out_channels,
            #     kernel_size=2*stride, stride=stride, causal=causal
            # )
            
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            StreamingResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake, causal=causal),
            StreamingResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake, causal=causal),
            StreamingResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake, causal=causal),
        )

    def forward(self, x):
        return self.layers(x)

    def initialize_from_conversion(self, decoder_block: DecoderBlock):
        """Initialize the streaming decoder block from a regular decoder block."""
        self.in_channels = decoder_block.in_channels
        self.out_channels = decoder_block.out_channels
        self.stride = decoder_block.stride
        self.use_snake = decoder_block.use_snake
        self.antialias_activation = decoder_block.antialias_activation
        self.use_nearest_upsample = decoder_block.use_nearest_upsample

        layers = []
        layers.append(get_activation("snake" if self.use_snake else "elu", antialias=self.antialias_activation, channels=self.in_channels))
        if self.use_snake:
            # just copy the alpha and beta parameters
            with torch.no_grad():
                layers[0].alpha.copy_(decoder_block.layers[0].alpha)
                layers[0].beta.copy_(decoder_block.layers[0].beta)
        layers.append(convert_wnconvtranspose1d_to_streamingconvtranspose1d(decoder_block.layers[1], StreamingConvTranspose1d))
        sru1 = StreamingResidualUnit(
            in_channels=decoder_block.layers[2].in_channels,
            out_channels=decoder_block.layers[2].out_channels,
            dilation=decoder_block.layers[2].dilation,
            use_snake=decoder_block.layers[2].use_snake,
            antialias_activation=decoder_block.layers[2].antialias_activation,
            causal=decoder_block.layers[2].causal
        )
        sru1.initialize_from_conversion(decoder_block.layers[2])
        layers.append(sru1)
        sru2 = StreamingResidualUnit(
            in_channels=decoder_block.layers[3].in_channels,
            out_channels=decoder_block.layers[3].out_channels,
            dilation=decoder_block.layers[3].dilation,
            use_snake=decoder_block.layers[3].use_snake,
            antialias_activation=decoder_block.layers[3].antialias_activation,
            causal=decoder_block.layers[3].causal
        )
        sru2.initialize_from_conversion(decoder_block.layers[3])
        layers.append(sru2)
        sru3 = StreamingResidualUnit(
            in_channels=decoder_block.layers[4].in_channels,
            out_channels=decoder_block.layers[4].out_channels,
            dilation=decoder_block.layers[4].dilation,
            use_snake=decoder_block.layers[4].use_snake,
            antialias_activation=decoder_block.layers[4].antialias_activation,
            causal=decoder_block.layers[4].causal
        )
        sru3.initialize_from_conversion(decoder_block.layers[4])
        layers.append(sru3)

        self.layers = nn.Sequential(*layers)

class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: tp.List[int] = [1, 2, 4, 8],
        strides: tp.List[int] = [2, 4, 8, 8],
        use_snake: bool = False,
        antialias_activation: bool = False,
        causal: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.causal = causal
        self.latent_dim = latent_dim
        self.input_c_mults = c_mults
        self.strides = strides
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation

        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        if self.causal:
            # Causal padding: pad only on the left
            padding = 0
        else:
            # Non-causal padding: pad on both sides
            padding = 3

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=padding, causal=causal)
        ]

        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake, causal=causal)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1 if not causal else 0, causal=causal)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class StreamingOobleckEncoder(StreamingContainer):

    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 causal=True
        ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.causal = causal
        self.latent_dim = latent_dim
        self.input_c_mults = c_mults
        self.strides = strides
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation


          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)
        if not self.causal:
            raise ValueError("StreamingOobleckEncoder does not support non-causal convolutions yet.")
        
        layers = [
            StreamingConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, causal=causal, norm="weight_norm")
        ]

        for i in range(self.depth-1):
            layers += [StreamingEncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake, causal=causal)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            StreamingConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, causal=causal, norm="weight_norm")
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def initialize_from_conversion(self, oobleck_encoder: OobleckEncoder):
        """Initialize the streaming oobleck encoder from a regular oobleck encoder."""
        self.in_channels = oobleck_encoder.in_channels
        self.channels = oobleck_encoder.channels
        self.causal = oobleck_encoder.causal
        self.latent_dim = oobleck_encoder.latent_dim
        self.input_c_mults = oobleck_encoder.input_c_mults
        self.strides = oobleck_encoder.strides
        self.use_snake = oobleck_encoder.use_snake
        self.antialias_activation = oobleck_encoder.antialias_activation
        self.depth = oobleck_encoder.depth
        
        print(f"Initializing StreamingOobleckEncoder with in_channels={self.in_channels}, channels={self.channels}, latent_dim={self.latent_dim}, input_c_mults={self.input_c_mults}, strides={self.strides}, use_snake={self.use_snake}, antialias_activation={self.antialias_activation}, causal={self.causal}")
        c_mults = [1] + oobleck_encoder.input_c_mults

        layers = []
        layers.append(convert_wnconv1d_to_streamingconv1d(oobleck_encoder.layers[0], StreamingConv1d))
        for i in range(self.depth-1):
            seb = StreamingEncoderBlock(
                in_channels=c_mults[i] * oobleck_encoder.channels,
                out_channels=c_mults[i+1] * oobleck_encoder.channels,
                stride=oobleck_encoder.strides[i],
                use_snake=oobleck_encoder.use_snake,
                antialias_activation=oobleck_encoder.antialias_activation,
                causal=oobleck_encoder.causal
            )
            seb.initialize_from_conversion(oobleck_encoder.layers[i + 1])
            layers.append(seb)
        
        act = get_activation("snake" if oobleck_encoder.use_snake else "elu", antialias=oobleck_encoder.antialias_activation, channels=c_mults[-1] * oobleck_encoder.channels)
        if oobleck_encoder.use_snake:
            # just copy the alpha and beta parameters
            with torch.no_grad():
                act.alpha.copy_(oobleck_encoder.layers[-2].alpha)
                act.beta.copy_(oobleck_encoder.layers[-2].beta)
        layers.append(act)
        layers.append(convert_wnconv1d_to_streamingconv1d(oobleck_encoder.layers[-1], StreamingConv1d))
        self.layers = nn.Sequential(*layers)

class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: tp.List[int] = [1, 2, 4, 8],
        strides: tp.List[int] = [2, 4, 8, 8],
        use_snake: bool = False,
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
        final_tanh: bool = True,
        causal: bool = False
    ):
        super().__init__()
        self.out_channels = out_channels
        self.causal = causal
        self.channels = channels
        self.latent_dim = latent_dim
        self.input_c_mults = c_mults
        self.strides = strides
        self.use_nearest_upsample = use_nearest_upsample
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation
        self.final_tanh = final_tanh

        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        if self.causal:
            # Causal padding: pad only on the left
            padding = 0
        else:
            # Non-causal padding: pad on both sides
            padding = 3
        # print(f"Initializing OobleckDecoder with out_channels={out_channels}, channels={channels}, latent_dim={latent_dim}, c_mults={c_mults}, strides={strides}, use_snake={use_snake}, antialias_activation={antialias_activation}, use_nearest_upsample={use_nearest_upsample}, final_tanh={final_tanh}, causal={causal}, padding={padding}")
        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=padding, causal=causal),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i] * channels, 
                out_channels=c_mults[i - 1] * channels,
                stride=strides[i - 1],
                use_snake=use_snake, 
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample,
                causal=causal
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=padding, bias=False, causal=causal),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class StreamingOobleckDecoder(StreamingContainer):

    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True,
                 causal=True):
        super().__init__()
        self.out_channels = out_channels
        self.causal = causal
        self.channels = channels
        self.latent_dim = latent_dim
        self.input_c_mults = c_mults
        self.strides = strides
        self.use_nearest_upsample = use_nearest_upsample
        self.use_snake = use_snake
        self.antialias_activation = antialias_activation
        self.final_tanh = final_tanh

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        if not self.causal:
            raise ValueError("StreamingOobleckDecoder does not support non-causal convolutions yet.")

        layers = [
            StreamingConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, causal=causal, norm="weight_norm"),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [StreamingDecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                use_snake=use_snake, 
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample,
                causal=causal
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            StreamingConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, causal=causal, norm="weight_norm"),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def initialize_from_conversion(self, oobleck_decoder: OobleckDecoder):
        """Initialize the streaming oobleck decoder from a regular oobleck decoder."""
        self.out_channels = oobleck_decoder.out_channels
        self.causal = oobleck_decoder.causal
        assert self.causal, "StreamingOobleckDecoder must be causal"
        self.channels = oobleck_decoder.channels
        self.latent_dim = oobleck_decoder.latent_dim
        self.input_c_mults = oobleck_decoder.input_c_mults
        self.strides = oobleck_decoder.strides
        self.use_nearest_upsample = oobleck_decoder.use_nearest_upsample
        self.use_snake = oobleck_decoder.use_snake
        self.antialias_activation = oobleck_decoder.antialias_activation
        self.final_tanh = oobleck_decoder.final_tanh
        self.depth = oobleck_decoder.depth
        # print all these
        print(f"Initializing StreamingOobleckDecoder with out_channels={self.out_channels}, channels={self.channels}, latent_dim={self.latent_dim}, input_c_mults={self.input_c_mults}, strides={self.strides}, use_snake={self.use_snake}, antialias_activation={self.antialias_activation}, use_nearest_upsample={self.use_nearest_upsample}, final_tanh={self.final_tanh}, causal={self.causal}")
        c_mults = [1] + self.input_c_mults
        layers = []
        layers.append(convert_wnconv1d_to_streamingconv1d(oobleck_decoder.layers[0], StreamingConv1d))

        layer_ctr = 1
        for i in range(self.depth-1, 0, -1):
            sdb = StreamingDecoderBlock(
                in_channels=c_mults[i] * oobleck_decoder.channels, 
                out_channels=c_mults[i-1] * oobleck_decoder.channels,
                stride=oobleck_decoder.strides[i-1], 
                use_snake=oobleck_decoder.use_snake, 
                antialias_activation=oobleck_decoder.antialias_activation,
                use_nearest_upsample=oobleck_decoder.use_nearest_upsample,
                causal=oobleck_decoder.causal
            )
            sdb.initialize_from_conversion(oobleck_decoder.layers[layer_ctr])
            layers.append(sdb)
            layer_ctr += 1
        act = get_activation("snake" if oobleck_decoder.use_snake else "elu", antialias=oobleck_decoder.antialias_activation, channels=oobleck_decoder.input_c_mults[0] * oobleck_decoder.channels)
        if oobleck_decoder.use_snake:
            # just copy the alpha and beta parameters
            with torch.no_grad():
                act.alpha.copy_(oobleck_decoder.layers[-3].alpha)
                act.beta.copy_(oobleck_decoder.layers[-3].beta)
        layers.append(act)
        layers.append(convert_wnconv1d_to_streamingconv1d(oobleck_decoder.layers[-2], StreamingConv1d))
        layers.append(nn.Tanh() if oobleck_decoder.final_tanh else nn.Identity())

        self.layers = nn.Sequential(*layers)

class DACEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        # from dac.model.dac import Encoder as DACEncoder
        from .dac import Encoder as DACEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACEncoder(d_latent=encoder_out_dim, **kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x


class DACDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        # from dac.model.dac import Decoder as DACDecoder
        from .dac import Decoder as DACDecoder

        self.decoder = DACDecoder(**kwargs, input_channel=latent_dim, d_out=out_channels)
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int,
        io_channels: int = 2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels: tp.Optional[int] = None,
        out_channels: tp.Optional[int] = None,
        soft_clip: bool = False
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.min_length = self.downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels if (in_channels is None) else in_channels
        self.out_channels = io_channels if (out_channels is None) else out_channels

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.pretransform = pretransform

        self.soft_clip = soft_clip
        self.is_discrete = self.bottleneck and self.bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, **kwargs):
        """
        iterate_batch (int) can be used as max batch size of processing
        """
        if self.pretransform and not skip_pretransform:
            with nullcontext() if self.pretransform.enable_grad else torch.no_grad():
                if iterate_batch:
                    max_bs = int(iterate_batch)
                    n_iter = int(math.ceil(audio.shape[0] / max_bs))
                    audios = []
                    for i in range(n_iter):
                        audios.append(self.pretransform.encode(audio[i * max_bs:(i + 1) * max_bs]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)

        if self.encoder:
            if iterate_batch:
                max_bs = int(iterate_batch)
                n_iter = int(math.ceil(audio.shape[0] / max_bs))
                latents = []
                for i in range(n_iter):
                    latents.append(self.encoder(audio[i * max_bs:(i + 1) * max_bs]))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio)
        else:
            latents = audio

        info = {}
        if self.bottleneck:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)

        return (latents, info) if return_info else latents

    def decode(self, latents, iterate_batch=False, **kwargs):
        if self.bottleneck:
            if iterate_batch:
                max_bs = int(iterate_batch)
                n_iter = int(math.ceil(latents.shape[0] / max_bs))
                decoded = []
                for i in range(n_iter):
                    decoded.append(self.bottleneck.decode(latents[i * max_bs:(i + 1) * max_bs]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            max_bs = int(iterate_batch)
            n_iter = int(math.ceil(latents.shape[0] / max_bs))
            decoded = []
            for i in range(n_iter):
                decoded.append(self.decoder(latents[i * max_bs:(i + 1) * max_bs]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)

        if self.pretransform:
            with torch.no_grad() if not self.pretransform.enable_grad else nullcontext():
                if iterate_batch:
                    max_bs = int(iterate_batch)
                    n_iter = int(math.ceil(decoded.shape[0] / max_bs))
                    decodeds = []
                    for i in range(n_iter):
                        decodeds.append(self.pretransform.decode(decoded[i * max_bs:(i + 1) * max_bs]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)

        return decoded

    def decode_tokens(self, tokens, **kwargs):
        '''
        Decode discrete tokens to audio
        Only works with discrete autoencoders
        '''
        assert isinstance(self.bottleneck, DiscreteBottleneck), "decode_tokens only works with discrete autoencoders"

        latents = self.bottleneck.decode_tokens(tokens, **kwargs)

        return self.decode(latents, **kwargs)

    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list] * batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"

        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape) == 2, "Audio should be shape (Channels x Length) with no batch dimension"
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length,
                                         target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor
        return torch.stack(new_audio)

    def encode_audio(
        self,
        audio,
        chunked: bool = False,
        chunk_size: int = 128,
        overlap: int = 4,
        max_batch_size: int = 1,
        **kwargs
    ):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        bs, n_ch, sample_length = audio.shape
        compress_ratio = self.downsampling_ratio
        assert n_ch == self.in_channels
        assert sample_length % compress_ratio == 0, 'The audio length must be a multiple of compression ratio.'

        latent_length = sample_length // compress_ratio
        chunk_size_l = chunk_size
        overlap_l = overlap
        hopsize_l = chunk_size - overlap

        # window for cross-fade of latent vectors
        win = torch.bartlett_window(overlap * 2, device=audio.device)

        if not chunked:
            # encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            # chunked encoding for lower memory consumption

            # converting a unit from latents to samples
            chunk_size *= compress_ratio
            overlap *= compress_ratio
            hopsize = chunk_size - overlap

            # zero padding
            n_chunk = int(math.ceil((sample_length - chunk_size) / hopsize)) + 1
            pad_len = chunk_size + hopsize * (n_chunk - 1) - sample_length
            audio = F.pad(audio, (0, pad_len))

            chunks = []
            for i in range(n_chunk):
                head = i * hopsize
                chunk = audio[..., head:head + chunk_size]
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)  # (bs, n_chunk, n_ch, chunk_size)
            chunks = rearrange(chunks, "b n c l -> (b n) c l")

            # batched encoding
            n_iter = int(math.ceil(chunks.shape[0] / max_batch_size))
            zs = []
            for i in range(n_iter):
                head = i * max_batch_size
                chunks_ = chunks[head: head + max_batch_size]
                z_ = self.encode(chunks_)
                zs.append(z_)

            zs = torch.cat(zs, dim=0)
            zs = rearrange(zs, "(b n) c l -> b n c l", b=bs)  # (bs, n_chunk, latent_dim, chank_size_l)

            # cross-fade of latent vectors
            latents = torch.zeros((bs, self.latent_dim, audio.shape[-1] // compress_ratio), device=audio.device)
            for i in range(n_chunk):
                z_ = zs[:, i]
                if i != 0:
                    z_[:, :, :overlap_l] *= win[None, None, :overlap_l]
                if i != n_chunk - 1:
                    z_[:, :, -overlap_l:] *= win[None, None, -overlap_l:]

                head = i * hopsize_l
                latents[..., head: head + chunk_size_l] += z_

            # fix size
            latents = latents[..., :latent_length]  # (bs, latent_dim, latent_length)

            return latents

    def decode_audio(
        self,
        latents,
        chunked=False,
        chunk_size=128,
        overlap=4,
        max_batch_size: int = 1,
        **kwargs
    ):
        '''
        Decode latents to audio.
        '''
        bs, latent_dim, latent_length = latents.shape
        compress_ratio = self.downsampling_ratio
        assert latent_dim == self.latent_dim

        hopsize = chunk_size - overlap
        chunk_size_s = chunk_size * compress_ratio
        overlap_s = overlap * compress_ratio
        hopsize_s = hopsize * compress_ratio
        sample_length = latent_length * compress_ratio

        # window for cross-fade of audio samples
        win = torch.bartlett_window(overlap_s * 2, device=latents.device)

        if not chunked:
            # decode the entire latent in parallel
            return self.decode(latents, **kwargs)
        else:
            # chunked decoding

            # reflect padding
            n_chunk = int(math.ceil((latent_length - chunk_size) / hopsize)) + 1
            pad_len = chunk_size + hopsize * (n_chunk - 1) - latent_length
            latents = F.pad(latents, (0, pad_len), mode='reflect')

            chunks = []
            for i in range(n_chunk):
                head = i * hopsize
                chunk = latents[..., head: head + chunk_size]
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)
            chunks = rearrange(chunks, "b n c l -> (b n) c l")

            # batched decoding
            n_iter = int(math.ceil(chunks.shape[0] / max_batch_size))
            xs = []
            for i in range(n_iter):
                head = i * max_batch_size
                chunks_ = chunks[head: head + max_batch_size]
                x_ = self.decode(chunks_)
                xs.append(x_)

            xs = torch.cat(xs, dim=0)
            xs = rearrange(xs, "(b n) c l -> b n c l", b=bs)  # (bs, n_chunk, n_ch, chank_size_sample)

            # cross-fade of audio samples
            audios = torch.zeros((bs, xs.shape[2], latents.shape[-1] * compress_ratio), device=latents.device)
            for i in range(n_chunk):
                x_ = xs[:, i]
                if i != 0:
                    x_[:, :, :overlap_s] *= win[None, None, :overlap_s]
                if i != n_chunk - 1:
                    x_[:, :, -overlap_s:] *= win[None, None, -overlap_s:]

                head = i * hopsize_s
                audios[..., head: head + chunk_size_s] += x_

            # fix size
            audios = audios[..., :sample_length]  # (bs, n_ch, sample_length)

            return audios

    @torch.no_grad()
    def reconstruct_audio(
        self,
        audio,
        chunked: bool = True,
        chunk_size: int = 128,
        overlap: int = 4,
        max_batch_size: int = 1,
        **kwargs
    ):
        '''
        Encode and decode audios at once.
        '''
        bs, n_ch, sample_length = audio.shape
        compress_ratio = self.downsampling_ratio
        assert n_ch == self.in_channels

        # window for cross-fade of audio samples
        overlap_s = overlap * compress_ratio
        win = torch.bartlett_window(overlap_s * 2, device=audio.device)

        if not chunked:
            return self.decode(self.encode(audio, **kwargs), **kwargs)
        else:
            # chunked encoding for lower memory consumption

            # converting a unit from latents to samples
            chunk_size *= compress_ratio
            overlap *= compress_ratio
            hopsize = chunk_size - overlap

            # zero padding
            n_chunk = int(math.ceil((sample_length - chunk_size) / hopsize)) + 1
            pad_len = chunk_size + hopsize * n_chunk - sample_length
            audio = F.pad(audio, (0, pad_len))

            chunks = []
            for i in range(n_chunk):
                head = i * hopsize
                chunk = audio[..., head:head + chunk_size]
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)  # (bs, n_chunk, n_ch, chunk_size)
            chunks = rearrange(chunks, "b n c l -> (b n) c l")

            # batched reconstruction
            n_iter = int(math.ceil(chunks.shape[0] / max_batch_size))
            xs = []
            for i in range(n_iter):
                head = i * max_batch_size
                chunks_ = chunks[head: head + max_batch_size]
                x_ = self.decode(self.encode(chunks_))
                xs.append(x_)

            xs = torch.cat(xs, dim=0)
            xs = rearrange(xs, "(b n) c l -> b n c l", b=bs)  # (bs, n_chunk, n_ch, chank_size_sample)

            # cross-fade of audio samples
            audio_rec = torch.zeros((bs, xs.shape[2], audio.shape[-1]), device=audio.device)
            for i in range(n_chunk):
                x_ = xs[:, i]
                if i != 0:
                    x_[:, :, :overlap_s] *= win[None, None, :overlap_s]
                if i != n_chunk - 1:
                    x_[:, :, -overlap_s:] *= win[None, None, -overlap_s:]

                head = i * hopsize
                audio_rec[:, :, head: head + chunk_size] += x_

            # fix size
            audio_rec = audio_rec[..., :sample_length]  # (bs, n_ch, sample_length)

            return audio_rec


# class DiffusionAutoencoder(AudioAutoencoder):
#     def __init__(
#         self,
#         diffusion: ConditionedDiffusionModel,
#         diffusion_downsampling_ratio,
#         *args,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)

#         self.diffusion = diffusion
#         self.min_length = self.downsampling_ratio * diffusion_downsampling_ratio

#         if self.encoder:
#             # Shrink the initial encoder parameters to avoid saturated latents
#             with torch.no_grad():
#                 for param in self.encoder.parameters():
#                     param *= 0.5

#     def decode(self, latents, steps=100):
#         upsampled_length = latents.shape[2] * self.downsampling_ratio

#         if self.bottleneck:
#             latents = self.bottleneck.decode(latents)

#         if self.decoder:
#             latents = self.decode(latents)

#         # Upsample latents to match diffusion length
#         if latents.shape[2] != upsampled_length:
#             latents = F.interpolate(latents, size=upsampled_length, mode='nearest')

#         noise = torch.randn(latents.shape[0], self.io_channels, upsampled_length, device=latents.device)
#         decoded = sample(self.diffusion, noise, steps, 0, input_concat_cond=latents)

#         if self.pretransform:
#             if self.pretransform.enable_grad:
#                 decoded = self.pretransform.decode(decoded)
#             else:
#                 with torch.no_grad():
#                     decoded = self.pretransform.decode(decoded)

#         return decoded

# AE factories


def create_encoder_from_config(encoder_config: tp.Dict[str, tp.Any]):
    encoder_type = encoder_config["type"]

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(**encoder_config["config"])
    elif encoder_type == "seanet":
        from encodec.modules import SEANetEncoder
        seanet_encoder_config = encoder_config["config"]
        # SEANet encoder expects strides in reverse order
        seanet_encoder_config["ratios"] = list(reversed(seanet_encoder_config.get("ratios", [2, 2, 2, 2, 2])))
        encoder = SEANetEncoder(**seanet_encoder_config)
    elif encoder_type == "dac":
        dac_config = encoder_config["config"]
        encoder = DACEncoderWrapper(**dac_config)
    elif encoder_type == "local_attn":
        from .local_attention import TransformerEncoder1D
        local_attn_config = encoder_config["config"]
        encoder = TransformerEncoder1D(**local_attn_config)
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")

    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder


def create_decoder_from_config(decoder_config: tp.Dict[str, tp.Any]):
    decoder_type = decoder_config["type"]

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(**decoder_config["config"])
    elif decoder_type == "seanet":
        from encodec.modules import SEANetDecoder
        decoder = SEANetDecoder(**decoder_config["config"])
    elif decoder_type == "dac":
        dac_config = decoder_config["config"]
        decoder = DACDecoderWrapper(**dac_config)
    elif decoder_type == "local_attn":
        from .local_attention import TransformerDecoder1D
        local_attn_config = decoder_config["config"]
        decoder = TransformerDecoder1D(**local_attn_config)
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")

    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder


def create_autoencoder_from_config(config: tp.Dict[str, tp.Any]):

    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])
    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config["latent_dim"]
    downsampling_ratio = ae_config["downsampling_ratio"]
    io_channels = ae_config["io_channels"]
    sample_rate = config["sample_rate"]

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)
    pretransform = ae_config.get("pretransform", None)

    if pretransform:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )


# def create_diffAE_from_config(config: tp.Dict[str, tp.Any]):

#     diffae_config = config["model"]

#     if "encoder" in diffae_config:
#         encoder = create_encoder_from_config(diffae_config["encoder"])
#     else:
#         encoder = None

#     if "decoder" in diffae_config:
#         decoder = create_decoder_from_config(diffae_config["decoder"])
#     else:
#         decoder = None

#     diffusion_model_type = diffae_config["diffusion"]["type"]

#     if diffusion_model_type == "DAU1d":
#         diffusion = DAU1DCondWrapper(**diffae_config["diffusion"]["config"])
#     elif diffusion_model_type == "adp_1d":
#         diffusion = UNet1DCondWrapper(**diffae_config["diffusion"]["config"])
#     elif diffusion_model_type == "dit":
#         diffusion = DiTWrapper(**diffae_config["diffusion"]["config"])

#     latent_dim = diffae_config["latent_dim"]
#     downsampling_ratio = diffae_config["downsampling_ratio"]
#     io_channels = diffae_config["io_channels"]
#     sample_rate = config["sample_rate"]

#     bottleneck = diffae_config.get("bottleneck", None)
#     pretransform = diffae_config.get("pretransform", None)

#     if pretransform:
#         pretransform = create_pretransform_from_config(pretransform, sample_rate)

#     if bottleneck:
#         bottleneck = create_bottleneck_from_config(bottleneck)

#     if diffusion_model_type == "DAU1d":
#         diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["strides"])
#     elif diffusion_model_type == "adp_1d":
#         diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["factors"])
#     elif diffusion_model_type == "dit":
#         diffusion_downsampling_ratio = 1
#     else:
#         raise NotImplementedError(f"No such model type: '{diffusion_model_type}'")

#     return DiffusionAutoencoder(
#         encoder=encoder,
#         decoder=decoder,
#         diffusion=diffusion,
#         io_channels=io_channels,
#         sample_rate=sample_rate,
#         latent_dim=latent_dim,
#         downsampling_ratio=downsampling_ratio,
#         diffusion_downsampling_ratio=diffusion_downsampling_ratio,
#         bottleneck=bottleneck,
#         pretransform=pretransform
#     )

def convert_encoder2streaming(encoder: nn.Module, device='cpu', requires_grad=False):
    """
    Convert an encoder to a streaming encoder.
    This is useful for converting OobleckEncoder to StreamingOobleckEncoder.
    """
    if isinstance(encoder, OobleckEncoder):
        streaming_enc = StreamingOobleckEncoder(
            in_channels=encoder.in_channels,
            channels=encoder.channels,
            latent_dim=encoder.latent_dim,
            c_mults=encoder.input_c_mults,
            strides=encoder.strides,
            use_snake=encoder.use_snake,
            antialias_activation=encoder.antialias_activation,
            causal=encoder.causal
        )
        streaming_enc.initialize_from_conversion(encoder)
        return streaming_enc.to(device).requires_grad_(requires_grad)

def convert_decoder2streaming(decoder: nn.Module, device='cpu', requires_grad=False):
    """
    Convert a decoder to a streaming decoder.
    This is useful for converting OobleckDecoder to StreamingOobleckDecoder.
    """
    if isinstance(decoder, OobleckDecoder):
        streaming_dec = StreamingOobleckDecoder(
            out_channels=decoder.out_channels,
            channels=decoder.channels,
            latent_dim=decoder.latent_dim,
            c_mults=decoder.input_c_mults,
            strides=decoder.strides,
            use_snake=decoder.use_snake,
            antialias_activation=decoder.antialias_activation,
            use_nearest_upsample=decoder.use_nearest_upsample,
            final_tanh=decoder.final_tanh,
            causal=True
        )
        streaming_dec.initialize_from_conversion(decoder)
        return streaming_dec.to(device).requires_grad_(requires_grad)