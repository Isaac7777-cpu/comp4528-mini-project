from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VectorQuantizer2']


class VectorQuantizer2(nn.Module):
    """
    Note that this is the novel multi-scale vector quantizer that is proposed in the paper. I will inherit all the
    configurations they have done.
    """

    def __init__(
            self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
            default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # Non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity) for _ in
                                            range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1:
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(share_quant_resi)]))

        # Use EMA to update the codebook to avoid the need to use any auxiliary loss.
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.Cvae)

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta} | S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    # ============ 'forward' function only for training ============
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor, torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        # print(B, C, H, W)
        f_no_grad = f_BChw.detach()  # As mentioned in the version 2 of the VQ-VAE implementation, the gradient
        # before the codebook is copied and does not flow through the codebook.

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        mean_vq_loss: torch.Tensor = 0.0
        vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
        SN = len(self.v_patch_nums)
        # This is where the magic happens --- the use of multi-scale quantised representation vector in the codebook.
        for si, pn in enumerate(self.v_patch_nums):
            # First, the input vector detached ones (f_rest), we need to downsample from [B, C, h, w] to [B, C, pn,
            # pn]. Then, depending on whether we use znorm mode. For znorm mode, the finding of the representation
            # quantised vector is find by using the one that matches the direction the best in the code book. If not
            # using znorm mode, then we are using the codebook vector that has the lowest l2 distance.
            #
            # Note that the embedding dimension is always the same, but the grid-size for containing this embedding
            # is not the same and hence have multi-scale.
            if self.using_znorm:
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (
                        si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                rest_NC = F.normalize(rest_NC, dim=-1)
                idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                # This part calculate the difference/loss by using l2 norm and separate it into three parts --- the
                # square of each and the dot product between.
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (
                        si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)

            # Count the number of codebook vector is used
            hit_V = idx_N.bincount(minlength=self.vocab_size).float()

            # calc loss
            # print(f"{rest_NC.shape=}")
            # print(f"{d_no_grad.shape=}")
            # print(f"{idx_N.shape=}")
            idx_Bhw = idx_N.view(B, pn, pn)  # [B, hw, hw]
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),  # self.embedding(idx_Bhw) has shape [B, h, w, C]
                    size=(H, W), mode='bicubic'
                ).contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)     # Apply the residual mapping. Note that this hopes to
                                                                # help with capturing a bit more information for a given
                                                                # reconstructed vector from quantised vectors
            f_hat.add(h_BChw)                                   # Add the so-far decoded information to the result first
            f_rest.sub_(h_BChw)                                 # Note that this is essential as we only want to
                                                                # represent the residual vector by the quantised vector
                                                                # from finer and finer scale

            # ema update
            if self.training:
                if self.record_hit == 0:
                    self.ema_vocab_hit_SV[si].copy_(hit_V)
                elif self.record_hit < 100:
                    self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul_(0.1))
                else:
                    self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul_(0.01))
                self.record_hit += 1
            vocab_hit_V.add_(hit_V)  # Update the statistic for the codebook
            mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

        total_codes = torch.sum(vocab_hit_V)
        avg_probs = vocab_hit_V / total_codes.clamp(min=1.0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        margin = (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in
                      enumerate(self.v_patch_nums)]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss, perplexity

    # ============ `forward` is only used in VAE training ============

    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        This function is essentially doing the step in the forward function from after obtaining the embedding vector
        until obtaining the reconstructed vector from using the quantised vector.
        :param ms_h_BChw: This is the list of multi-scale quantised vector
        :param last_one: Whether we want to only keep the final accumulated results; otherwise, will return the steps of accumulated steps
        :return: As demonstrated in the :attr:`last_one`
        """
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
        for si, pn in enumerate(self.v_patch_nums):  # This is basically the same as the forward function above
            h_BChw = ms_h_BChw[si]
            if si < len(self.v_patch_nums) - 1:
                h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            if last_one:
                ls_f_hat_BChw = f_hat
            else:
                ls_f_hat_BChw.append(f_hat.clone())

        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool,
                           v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) \
            -> List[Union[torch.Tensor, torch.LongTensor]]:
        """
        This function maps from the normal input of the quantizer (i.e. the encoded version of the input image) to the
        reconstructured version, which is essentially the same as in :meth:`forward` but just not calculating the loss
        and performing the ema update.
        :param f_BChw: The encoded image feature map
        :param to_fhat: Whether to output the final output as the reconstructed version by the quantised representation or just a list of codebook index
        :param v_patch_nums: This is the patches sizes
        :return: As described in `v_patch_nums`
        """
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_bl: List[torch.Tensor] = []

        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            # As usual, convert from [B, C, h, w] to [B, h, w, C]
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )

            # Find the useful codebook vector either using cosine similarity or L2 norm
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = (torch.sum(z_NC.square(), dim=1, keepdim=True)
                             + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False))
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # Euclidean norm
                idx_N = torch.argmin(d_no_grad, dim=1)

            # Upsample to the original size and
            # 1. Add to the resultant vector f_hat
            # 2. Subtract from the residual of the vector waiting for explain
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = (
                F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si/(SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))

        return f_hat_or_idx_bl

    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        f"""
        This is to generate a "sequence" that is required for VAR training.
        :param gt_ms_idx_Bl: The list of [B, h, w] tensors containing the codebook indexes / logits
        :return: The tensor of shape [B, sum(pn_next * pn_next), C] 
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN - 1):
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))       # The final shape is [B, h * w, C]
        return torch.cat(next_scales, dim=1) if len(next_scales) > 1 else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]), mode='area')

class Phi(nn.Conv2d):
    """
    This function is to use the current amount to estimate the residual after using the quantised representation of the vector.
    There are also different forms as defined below.

    The forward h_BChw is simply the unquantized encoded vector.
    """

    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw: torch.Tensor) -> torch.Tensor:
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(self.qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self):
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self):
        return f'ticks={self.ticks}'
