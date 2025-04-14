"""
References: https://github.com/FoundationVision/VAR/blob/main/models/basic_vae.py#L163
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# __all__ = ['Encoder', 'Decoder', ]

# swish non-linearity
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalise(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# I believe the original implementation of the VAR model is only using the upsample and downsample of ratio 2.
# Therefore, I will also do this. However, this is not the common upsample where you just apply a transpose convolution
# The author has however chosen to first interpolate and apply the convolution. Using this method avoids checkerboard
# artifacts that come from transpose convolution, and provides smoother and more stable upscaling.
class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2,
                                    padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


# Then, we need to define the ResnetBlock and the Attention used in both the encoder and decoder.
# The resnet is used in downsampling and the attention is used at the end of the downsampling to model the global
# correlation between the spatial embeddings
class ResnetBlock(nn.Module):
    """
    A common convolutional residual block.
    """

    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()

        # Define the model dimension
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # Define the model architecture
        self.norm1 = Normalise(self.in_channels)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.norm2 = Normalise(self.out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                          stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))  # Using inplace for better memory efficiency
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels  # This is d_k in the attention model

        self.norm = Normalise(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels=in_channels, out_channels=3 * in_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                        padding=0)

    def forward(self, x):
        qkv: torch.Tensor = self.qkv(x)
        B, _, H, W = qkv.shape  # Should be [B, 3C, H, W]
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(dim=1)

        # QK Multiplications & Softmax
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # [B, HW, C]
        k = k.view(B, C, H * W).contiguous()  # [B, C, HW]
        w = torch.bmm(q, k).mul_(self.w_ratio)  # [B, HW, HW]    w[B, i, j] = sum_c q[B,i,C]k[B,C,j]
        w = F.softmax(w, dim=2)

        # Attend to V Note that this is in a slightly different way of implementation than what it was described in
        # the original Attention is All You Need paper simply because we want the final outcome be of shape [B, C, H,
        # W]. Now, in order to achieve this, normally W has shape [B, HW(q), HW(k)] and V has shape [B, HW(v),
        # C] as implies in the original paper. However, here V has shape [B, C, HW(v)] and would normally need to
        # transpose all the V- matrices to change V to be of shape [B, HW(v), C] matching with the normal paper. In
        # other words, we need to do W * V.t() per batch. However, then, we need to transpose again to get the final
        # Head matrix (H) to be of shape [B, C, H*W] and eventually turning it into [B, C, H, W]. Therefore,
        # in order to safe one transpose operation, We can simply do V * W.t() per batch. Hence, we have the
        # following implementation.
        v = v.view(B, C, H * W).contiguous()  # [B, C, HW]
        w = w.permute(0, 2, 1).contiguous()  # [B, HW(q), HW(v)]
        h = torch.bmm(v, w)  # [B, C, HW(q)]
        h = h.view(B, C, H, W).contiguous()  # [B, C, H, W]

        return x + self.proj_out(h)


def make_attn(in_channels, using_sa=True):
    """
    To build in the attention block if we decide to, otherwise fill the gap with identity mapping.
    :param in_channels: The number of in_channels
    :param using_sa: Using self-attention or not.
    :return: AttnBlock() | Identity()
    """
    return AttnBlock(in_channels=in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):
    def __init__(
            self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
            dropout=0.0, in_channels=3,
            z_channels, double_z=False, using_sa=True, using_mid_sa=True
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling First, we have to downsample the image to allow for compressing the latent vectors First,
        # We have to convert the image into a latent representation with a convolution (conv1).
        #
        # Then we have to repeatedly downsample and augment the data with extra channels as defined in ch_mult
        # channel multipliers. Note that the images first go through the resnet to enrich the number channels like
        # how normal resnet do it. Note that for each resnet block, an optional attention block can be appended to
        # the end of each resnet block in the final resolution layer. If it is not in the final layer, a downsample
        # by ratio of two is applied to the image.
        #
        # After that it is the middle part of the encoder with ResNet -> Attention -> ResNet
        #
        # Then it is the end with norm -> conv

        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalise(block_in)
        self.conv_out = nn.Conv2d(in_channels=block_in, out_channels=(2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:        # This is sufficient to check whether we are using
                                                            # self-attention
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


class Decoder(nn.Module):
    def __init__(
            self, *, ch=128, ch_mult=(1,2,4,8), num_res_blocks=2,
            dropout=0.0, in_channels=3,         # in_channels: raw image channels
            z_channels, using_sa=True, using_mid_sa=True
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1, ) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = nn.Conv2d(in_channels=z_channels, out_channels=block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(in_channels=block_in)
            self.up.insert(0, up)                           # prepend this upsampling layer

        # end
        self.norm_out = Normalise(block_in)
        self.conv_out = nn.Conv2d(in_channels=block_in, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        # z to block_in then to middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h