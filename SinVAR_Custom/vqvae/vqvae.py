"""
References: https://github.com/FoundationVision/VAR/blob/main/models/vqvae.py
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .encoder_decoder import Encoder, Decoder
from .quatizer import VectorQuantizer2


class VQVAE(nn.Module):
    def __init__(
            self, vocab_size=4096, z_channels=32, ch=128, dropout=0.0,
            beta=0.25,
            using_znorm=False,
            quant_conv_ks=3,
            quant_resi=0.5,
            share_quant_resi=4,
            default_qresi_counts=0,
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            test_mode=True,
            ddconfig=None
    ):
        super().__init__()
        self.test_mode = test_mode

        # Encoder & Decoder Setup
        self.V, self.Cvae = vocab_size, z_channels
        # Add in Default if it is not needed
        if ddconfig is None:
            ddconfig = dict(
                dropout=dropout, ch=ch, z_channels=z_channels,
                in_channels=3, ch_mult=(1, 2, 4), num_res_blocks=2,
                using_sa=True, using_mid_sa=True
            )
        ddconfig.pop('double_z', None)
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Quantiser setup
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)
        self.quantize = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi,
            share_quant_resi=share_quant_resi
        )
        self.quant_conv = nn.Conv2d(self.Cvae, self.Cvae, kernel_size=quant_conv_ks, stride=1,
                                    padding=quant_conv_ks // 2)
        self.post_quant_conv = nn.Conv2d(self.Cvae, self.Cvae, kernel_size=quant_conv_ks, stride=1,
                                         padding=quant_conv_ks // 2)

        if self.test_mode:
            self.eval()
            [p.requires_grad for p in self.parameters()]

    def forward(self, inp: torch.Tensor, ret_usages=False):
        f_hat, usages, vq_loss, perplexity = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss, perplexity

    def fhat_to_image(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor,
                     v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[
        List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)

    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, last_one=True))).clamp_(-1,
                                                                                                                    1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in
                    self.quantize.embed_to_fhat(ms_h_BChw, last_one=False)]

    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
                                 last_one=False) -> List[torch.Tensor]:
        """
        This is basically the forward function with the option to return the accumulated list of representation of the reconstructed images or
        :param x:
        :param v_patch_nums:
        :param last_one:
        :return:
        """
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != \
                self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
