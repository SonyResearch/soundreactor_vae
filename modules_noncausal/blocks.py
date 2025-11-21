
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d


def concat_video_with_delta(video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: Tensor of shape [batch, time, height, width, dimension].
        Returns:
            video_concat: 
            Tensor of shape [batch, time, height, width, dimension * 2],
            where the last dimension is concatenation of the original video feature 
            and the temporal difference features.
            For t=0, delta is video[0] - 0; for t>=1, delta is video[t] - video[t-1].
        """
        B, T, H, W, D = video.shape

        # zero_frame = torch.zeros(B, 1, H, W, D, device=video.device, dtype=video.dtype)
        # video_shift = torch.cat([zero_frame, video[:, :-1]], dim=1)
        # video_delta = video - video_shift
        # video_concat = torch.cat([video, video_delta], dim=-1)
        video_shift = torch.roll(video, shifts=1, dims=1)
        video_shift[:, 0] = 0
        video_concat  = torch.cat([video, video - video_shift], dim=-1)  # [B, T, H, W, 2D]
        
        return video_concat

class Downsample(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

class ConvBlock(nn.Module):
    """
    Conv block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(
        self,
        dim,
        drop_path=0.,
        #  layer_scale=None,
        kernel_size=3
        ):
        super().__init__()
        """
        Args:
            drop_path: drop path.
            layer_scale: layer scale coefficient.
            kernel_size: kernel size.
        """
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        # self.layer_scale = layer_scale
        # if layer_scale is not None and type(layer_scale) in [int, float]:
        #     self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        #     self.layer_scale = True
        # else:
        #     self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # if self.layer_scale:
        #     x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x