# File: ESRGAN/post_train_eval_teacher.py

import os, sys, time, torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from tqdm import tqdm

# Add utils path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from nafnet_teacher import load_nafnet_teacher
from dataset import ImagePairDataset

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
val_lr = 'data/Dataset/Split/val/blurry'
val_hr = 'data/Dataset/Split/val/sharp'
ckpt_teacher = 'checkpoints/motion_deblurring_teacher_best.pth'

# Dataset
val_set = ImagePairDataset(val_lr, val_hr)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Load teacher model
model = load_nafnet_teacher(device=DEVICE)
model.load_state_dict(torch.load(ckpt_teacher, map_location=DEVICE))
model.eval()

# PSNR
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else 100

# Evaluation
print("\n🔍 Evaluating Teacher Model...")
total_ssim, total_psnr, total_fps = 0, 0, 0

with torch.no_grad():
    for lr, hr in tqdm(val_loader):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        # Timing for FPS
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        pred = model(lr)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # convert ms → seconds
        fps = 1.0 / elapsed_time
        total_fps += fps

        # Clamp predictions
        pred = torch.clamp(pred.float(), 0.0, 1.0)
        hr = torch.clamp(hr.float(), 0.0, 1.0)

        # Metrics
        total_ssim += ssim(pred, hr, data_range=1.0).item()
        total_psnr += psnr(pred, hr).item()

# Final averages
avg_ssim = total_ssim / len(val_loader)
avg_psnr = total_psnr / len(val_loader)
avg_fps = total_fps / len(val_loader)

# Results
print(f"\n✅ Teacher Evaluation Results")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average FPS : {avg_fps:.2f}")
