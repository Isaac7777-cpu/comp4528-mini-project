import math
from functools import partial
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from SinVAR.var.var_components import AdaLNBeforeHead, AdaLNSelfAttn, SelfAttnBlock, NormBeforeHead
from SinVAR.vqvae.quatizer import VectorQuantizer2
from SinVAR.vqvae.vqvae import VQVAE


class SharedAdaLin(nn.Linear):
    def forward(self, input):
        C = self.weight.shape[0] // 6
        return super().forward(input).view(-1, 1, 6, C)


class VARClassLabel(nn.Module):
    """
    deprecated.
    """

    def __init__(
            self, vae_local: VQVAE,
            num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            device="mps"
    ):
        super().__init__()

        # Hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.prog_si = -1

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=device)

        # 1. Input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. Class Embedding
        init_std = math.sqrt((1 / self.C) / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32,
                                       device=device)
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. Absolute Position Embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment emedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. Backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False),
                                            SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln, block_idx=block_idx, embed_dim=self.C,
                norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
            )
            for block_idx in range(depth)
        ])

        # 5. Attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        # This is the special square attention mask mentioned in the paper as well.
        d: torch.Tensor = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L,
                                                                                                              1)
        dT = d.transpose(1, 2)  # 1, 1, L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. Classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
            self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
            more_smooth=False
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gambel softmax; only used in visualisation, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        raise NotImplementedError("Have not been implemented...")

    def forward(self, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        """
        This function is mainly done in two steps:

        1. Generate the unconditional class label
        2. Run through the AdaLN


        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L - self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        sos = cond_BD = self.class_emb(torch.tensor(self.num_classes, device=x_BLCv_wo_first_l.device)).expand(B, -1)
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

        if self.prog_si == 0:
            x_BLC = sos  # Not progressive
        else:
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC; pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # Hack: get the dtype if mixed prevision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m,
                            (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                             nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VAR(nn.Module):
    def __init__(
            self, vae_local: VQVAE,
            depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            device="mps"
    ):
        super().__init__()
        self.patch_nums = patch_nums

        # Hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        # Progressive Training Setup
        self.prog_si = -1
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=device)

        # 1. Input (word) Embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. SOS Token Embedding
        init_std = math.sqrt((1 / self.C) / 3)
        self.sos_token = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.sos_token.data, mean=0, std=init_std)

        # 3. Absolute Position Embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC,
                            dim=1)  # Positional Embedding, defined for all L positions in one batch. Shape: [1, L, C]
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. Backbone Blocks
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SelfAttnBlock(
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm
            )
            for block_idx in range(depth)
        ])

        print(
            f'\n[constructor]  ==== UNCONDITIONAL VAR: using SelfAttnBlock ({depth} blocks, no class label) ====\n'
            f'    [VAR config ] embed_dim={self.C}, num_heads={num_heads}, depth={depth}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask sued in training (for masking out the future)
        #    It won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([
            torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L,
                                                                             self.L)  # This is the attention mask
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. Classifier Head. This is the last layer to map from the model latent dimension to the vocab space
        self.head_nm = NormBeforeHead(self.C, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual) -> torch.Tensor:
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float()))

    @torch.no_grad()
    def autoregressive_infer(self, B: int, g_seed: Optional[int] = None) -> torch.Tensor:
        """
        Only used for inference, on autoregressive way
        :param B: The batch size. One should ensure that the input is indeed of size B in the first dimension
        :param g_seed: The random generator seed
        :return: The embedding. Need to go through vae to decode
        """
        training = self.training
        self.eval()
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng;

        sos = self.sos_token.expand(B, self.first_l, -1)

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos + lvl_pos[:, :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks: b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            cur_L += pn * pn
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, attn_bias=None)
            logits_BlV = self.get_logits(x)

            # idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            idx_Bl = logits_BlV.argmax(dim=-1)
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)

            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L: cur_L + self.patch_nums[si + 1] ** 2]

        for b in self.blocks: b.attn.kv_caching(False)

        self.train(training)    # Re-establish training state.
        return self.vae_proxy[0].fhat_to_image(f_hat).add_(1).mul_(0.5)     # Denormalise, from [-1, 1] to [0, 1]

    @torch.no_grad()
    def autoregressive_infer_with_context(
            self,
            context: torch.Tensor,    # [B, C, H, W]
            context_start_idx: int,
            single_injection: bool = False
    ) -> torch.Tensor:
        """
        This auto-regress with some context image.
        :param context: The input image that needs to be used for context.
                        Should have shape [B, 3, H, W] (H = W = 128 for this project)
        :param context_start_idx: The patch to inject the correct map.
        :return: The conditionally generated image.
        """
        assert context_start_idx < len(self.patch_nums) - 1, "[VAR] Context amount should be leave at least one place for the model to predict."

        training = self.training
        self.eval()

        B = context.shape[0]

        ############### Calculate the Tensor to be Injected #########

        context_embedings: List[torch.Tensor] = []

        for target_idx in range(context_start_idx + 1):
            # Now, we need to actually use the correct index for all pi less than context start index
            end_idx = sum(pn**2 for pn in self.patch_nums[1:target_idx+1])
            start_idx = end_idx - self.patch_nums[target_idx] ** 2
            # print(f"{start_idx=}, {end_idx=}")

            # Then, we need to determine the actual correct index with the encoder
            gt_idx_Bl: List[torch.LongTensor] = self.vae_proxy[0].img_to_idxBl(context)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)[:, self.first_l + start_idx: self.first_l + end_idx]
            # print(f"{gt_BL.shape=}, {self.L=}")

            # Pre-compute all the required embeddings
            target_pn = self.patch_nums[target_idx]
            context_embedding_BChw = self.vae_quant_proxy[0].embedding(gt_BL)
            context_embedding_BChw = context_embedding_BChw.transpose(1, 2).reshape(B, self.Cvae, target_pn, target_pn)
            context_embedings.append(context_embedding_BChw)

        ############### Basic Autoregressive #######################
        sos = self.sos_token.expand(B, self.first_l, -1)

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos + lvl_pos[:, :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks: b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            cur_L += pn * pn
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, attn_bias=None)
            logits_BlV = self.get_logits(x)         # This is a logit probability tensor (close to one-hot)
            # print(logits_BlV, logits_BlV.shape)

            idx_Bl = logits_BlV.argmax(dim=-1)      # Pick the most likely label determinately
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)

            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.Cvae, pn, pn) \
                     if (not single_injection and si > context_start_idx) or (single_injection and si != context_start_idx) \
                     else context_embedings[si]
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L: cur_L + self.patch_nums[si + 1] ** 2]

        for b in self.blocks: b.attn.kv_caching(False)

        self.train(training)    # Re-establish training state.
        return self.vae_proxy[0].fhat_to_image(f_hat).add_(1).mul_(0.5)     # Denormalise, from [-1, 1] to [0, 1]

    def forward(self, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B: int = x_BLCv_wo_first_l.shape[0]  # Batch Size

        sos = self.sos_token.expand(B, self.first_l, -1)
        if self.prog_si == 0:                   # Progressive training and currently at the starting location.
            x_BLC = sos
        else:
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]

        x_BLC = x_BLC.to(dtype=torch.float32)
        attn_bias = attn_bias.to(dtype=torch.float32)

        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float())

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC  # logits BLV, V is vocab_size

    def init_weights(self, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None

            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
                    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
            )):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        depth = len(self.blocks)
        scale = 1 / math.sqrt(2 * depth)
        for block_idx, sab in enumerate(self.blocks):
            sab.attn.proj.weight.data.mul_(scale)
            sab.ffn.fc2.weight.data.mul_(scale)
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
