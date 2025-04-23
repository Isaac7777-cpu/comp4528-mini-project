from SinVAR.vqvae import VQVAE
from SinVAR.var import VAR
from typing import Tuple
from torch import nn


def build_vae_var(
        # Shared args
        device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        # VQVAE args
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        # VAR args
        depth=16, attn_l2_norm=True,
        init_head=0.02, init_std=-1,  # init_std < 0: automated
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
    ).to(device)
    var_wo_ddp.init_weights(init_head=init_head, init_std=init_std)

    return vae_local, var_wo_ddp
