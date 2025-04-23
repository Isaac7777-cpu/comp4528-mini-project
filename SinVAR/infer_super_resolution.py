import os.path as osp
import torch
from SinVAR.utils.model_builder import build_vae_var
from SinVAR.var import VAR
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms import v2
import os
from skimage.exposure import match_histograms
import time

time_str: str = time.strftime("%Y%m%d-%H:%M:%S")

vae_var_config = {
    # Shared
    "device": torch.device("mps" if torch.mps.is_available() else "cpu"),
    "patch_nums": (1, 2, 3, 4, 5, 6, 8),

    # VAR config (customizable for your setup)
    "depth": 8,  # VAR transformer depth
    "attn_l2_norm": True,

    # Initialisation options (irrelevant for non-adaptive setup)
    "init_head": 0.02,
    "init_std": -1,  # Use default init scheme
}

vae_local, var_wo_ddp = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,  # hard-coded VQVAE hyperparameters
    device=vae_var_config['device'],
    depth=vae_var_config['depth'], attn_l2_norm=vae_var_config['attn_l2_norm'],
    init_head=vae_var_config['init_head'], init_std=vae_var_config['init_std'],
    patch_nums=vae_var_config['patch_nums']
)

# Download from Hugging Face
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt = 'vae_ch160v4096z32.pth'
local_path = './pretrained-models'
if not osp.exists(os.path.join(local_path, vae_ckpt)): os.system(f'wget {hf_home}/{vae_ckpt}')

# load checkpoints
vae_local.load_state_dict(torch.load(os.path.join(local_path, vae_ckpt), map_location=vae_var_config['device']),
                          strict=True)

print('')

# load var
model_path = "./runs/models/var_training_v0.5.1_depth=10_train_size=1024_epoc=100_best_val.pth"
var_wo_ddp : VAR = torch.load(model_path, weights_only=False)

print("[INFER_LOG] Successfully loaded Visual AutoRegressive Model\n")

# Load previously generated images
inp_img_path = './generated_images/unconditional/image_0.png'
train_img_path = './data/transformed_image.jpg'
input_img = read_image(inp_img_path).float() / 255.0                 # Convert to float tensor [0,1]
train_img = read_image(train_img_path).float() / 255.0                 # Convert to float tensor [0,1]
train_img = v2.Resize(size=input_img.shape[2])(train_img.cpu())

# Matching color distribution does not work very well.
# However, since I have used some color jittering in training for generalisability in training. However, it is better to
# revert some of it here given that we are doing autoregressive predictions based on autoregressive predictions.
# # Match with the base image first
# input_img_np = input_img.numpy()
# train_img_np = train_img.numpy()
# print(f"{input_img_np.shape=}, {train_img_np.shape=}")
# matched_np = match_histograms(input_img_np, train_img_np, channel_axis=-1)
# matched_tensor = torch.from_numpy(matched_np)
# matched_tensor = v2.functional.adjust_brightness(input_img, 1.6)
# matched_tensor = v2.functional.adjust_contrast(matched_tensor, 1.005)
# matched_tensor = v2.functional.adjust_saturation_image(matched_tensor, 1.005)
# matched_tensor = v2.functional.adjust_hue(matched_tensor, 0.001)

# Create directory to save images
save_dir = f"./generated_images/conditional/{time_str}"
os.makedirs(save_dir, exist_ok=True)

matched_tensor = v2.Resize(size=128)(train_img)

# Save the Reconstructed Image before attempting super-resolution
save_image(matched_tensor, f"./generated_images/conditional/{time_str}/adjusted.png")

# Scaled up the image
enlarged_img = (v2.Resize(size=256)(matched_tensor)).to(torch.device(vae_var_config['device']))
print(enlarged_img.shape)
patches = []
for i in range(2):          # vertical: 0 and 1
    for j in range(2):      # horizontal: 0 and 1
        top = i * 128
        left = j * 128
        patch = enlarged_img[:, top:top + 128, left:left + 128]
        patches.append(patch)
context_tensor = torch.stack(patches, dim=0).to(vae_var_config['device'])   # [B=2, C=3, H=128, W=128]

# Obtaining the image
output: torch.Tensor = var_wo_ddp.autoregressive_infer_with_context(context=context_tensor, context_start_idx=4)

print("[INFER_LOG] Outputs generated...\n")

# Make sure output is on CPU
output = output.cpu()

print(f"[INFER_LOG] Saving generated images to {os.path.join(os.getcwd(), save_dir)}\n")

# Save each image
for i in range(output.size(0)):
    img = output[i]  # shape: [3, 128, 128]
    save_image(img, os.path.join(save_dir, f"generated_image_{i}.png"))

for i in range(context_tensor.shape[0]):
    img = context_tensor[i]
    save_image(img, os.path.join(save_dir, f"original_image_{i}.png"))

top_stitched = torch.cat([output[0], output[1]], dim=2)
bottom_stitched = torch.cat([output[2], output[3]], dim=2)
full = torch.cat([top_stitched, bottom_stitched], dim=1)

save_image(full, os.path.join(save_dir, f"stitched_image.png"))

print("[INFER_LOG] Done")
