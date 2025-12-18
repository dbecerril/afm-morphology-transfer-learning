# models/ae_unet_gn.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """
    Feature-wise linear modulation:
      h' = (1 + gamma(aux)) * h + beta(aux)

    aux is pooled to (B, aux_ch, 1, 1) so it can't inject spatial texture.
    """
    def __init__(self, aux_ch: int, feat_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(aux_ch, 2 * feat_ch, kernel_size=1),
        )
        # start near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        gb = self.net(aux)
        gamma, beta = gb.chunk(2, dim=1)
        return (1.0 + gamma) * h + beta

def gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    # Ensure groups divides channels
    g = min(num_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            gn(out_ch, groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            gn(out_ch, groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)  # strided conv
        self.block = ConvBlock(in_ch, out_ch, groups)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = ConvBlock(in_ch + skip_ch, out_ch, groups)

    def forward(self, x, skip):
        x = self.up(x)
        # handle any off-by-one due to odd sizes (shouldn't happen with 256, but safe)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class AFMUNetAutoencoder(nn.Module):
    """
    U-Net-ish autoencoder.
    - Good recon/denoising
    - Stable with small batch sizes via GroupNorm
    - Provides an embedding for clustering via global pooling on bottleneck
    """
    def __init__(self,in_channels: int = 1,base: int = 32,groups: int = 8,out_channels: int = 1,aux_channels: int = 0,aux_dropout: float = 0.0,):
        super().__init__()

        self.enc0 = ConvBlock(in_channels, base, groups)        # 256x256
        self.enc1 = Down(base, base * 2, groups)                # 128x128
        self.enc2 = Down(base * 2, base * 4, groups)            # 64x64
        self.enc3 = Down(base * 4, base * 8, groups)            # 32x32

        # bottleneck at 16x16
        self.bottleneck_down = nn.Conv2d(base * 8, base * 8, 3, stride=2, padding=1)
        self.bottleneck = ConvBlock(base * 8, base * 8, groups)

        # decoder
        self.up3 = Up(base * 8, base * 8, base * 4, groups)     # 32x32
        self.up2 = Up(base * 4, base * 4, base * 2, groups)     # 64x64
        self.up1 = Up(base * 2, base * 2, base, groups)         # 128x128

        self.up0 = nn.Upsample(scale_factor=2, mode="nearest")  # 256x256
        self.dec0 = ConvBlock(base + base, base, groups)        # concat with enc0 skip

        self.out = nn.Conv2d(base, out_channels, 1)
        #FiLM conditioning
        self.aux_channels = aux_channels
        self.aux_dropout = float(aux_dropout)

        if aux_channels > 0:
            # late conditioning: start conservative (decoder only)
            self.film_up3 = FiLM(aux_channels, base * 4)  # output of up3
            self.film_up2 = FiLM(aux_channels, base * 2)  # output of up2
            self.film_up1 = FiLM(aux_channels, base)      # output of up1
            self.film_dec0 = FiLM(aux_channels, base)     # output of dec0

    def forward(self, topo: torch.Tensor, aux: torch.Tensor | None = None):
        # Encode using topo only
        s0 = self.enc0(topo)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b = self.bottleneck_down(s3)
        b = self.bottleneck(b)

        # Optional aux conditioning (late)
        if self.aux_channels > 0:
            if aux is None:
                raise ValueError("Model was created with aux_channels>0 but aux=None was passed to forward().")

            # Aux dropout to prevent over-reliance
            if self.training and self.aux_dropout > 0:
                if torch.rand((), device=aux.device) < self.aux_dropout:
                    aux = torch.zeros_like(aux)

        x = self.up3(b, s3)
        if self.aux_channels > 0:
            x = self.film_up3(x, aux)

        x = self.up2(x, s2)
        if self.aux_channels > 0:
            x = self.film_up2(x, aux)

        x = self.up1(x, s1)
        if self.aux_channels > 0:
            x = self.film_up1(x, aux)

        x = self.up0(x)
        x = torch.cat([x, s0], dim=1)
        x = self.dec0(x)
        if self.aux_channels > 0:
            x = self.film_dec0(x, aux)

        return self.out(x)


    @torch.no_grad()
    def embed(self, x):
        """
        Return a vector embedding per patch for clustering (B, D).
        We use global average pooling of the bottleneck feature map.
        """
        xtopo = x[:, 0:1]  # topo channel only
        s0 = self.enc0(xtopo)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck_down(s3)
        b = self.bottleneck(b)
        # global average pool: (B, C, H, W) -> (B, C)
        return b.mean(dim=(-2, -1))
