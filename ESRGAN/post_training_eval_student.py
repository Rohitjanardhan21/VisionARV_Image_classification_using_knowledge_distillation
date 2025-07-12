# File: ESRGAN/post_training_eval_student.py

import os, sys, time, math
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from dataset import ImagePairDataset

from student_model_enhanced import load_student_model, StudentNetEnhanced
from torch.cuda.amp import autocast

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
val_lr = 'data/Dataset/Split/val/blurry'
val_hr = 'data/Dataset/Split/val/sharp'
ckpt_path = 'checkpoints/best_student_model.pth'
save_output_dir = 'eval_outputs_student'
os.makedirs(save_output_dir, exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset and Loader
val_set = ImagePairDataset(val_lr, val_hr, transform=transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# PSNR
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse.item() > 0 else 100

# Load model
def load_student_model_device(path=None, device='cpu'):
    model = StudentNetEnhanced(channels=32, num_blocks=6).to(device)  # Match checkpoint parameters
    if path and os.path.exists(path):
        print(f"🔁 Resuming from checkpoint: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()

model = load_student_model_device(ckpt_path, device=DEVICE)

# Evaluation
total_ssim, total_psnr, total_fps = 0, 0, 0
num_samples = len(val_loader)

if num_samples == 0:
    print("❌ No validation samples found. Check your dataset paths.")
    sys.exit(1)

print(f"\n🔍 Evaluating {num_samples} samples...")
plt.ion()

with torch.no_grad():
    for i, (lr, hr) in enumerate(val_loader):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        start = time.time()
        with autocast():

            pred = model(lr)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        end = time.time()

        pred = torch.clamp(pred.float(), 0.0, 1.0)
        hr = torch.clamp(hr.float(), 0.0, 1.0)

        ssim_score = ssim(pred, hr, data_range=1.0).item()
        fps_score = 1 / (end - start)

        total_ssim += ssim_score
        total_psnr += psnr(pred, hr)
        total_fps += fps_score

        # Save images
        save_image(pred, os.path.join(save_output_dir, f"pred_{i}.png"))
        save_image(lr, os.path.join(save_output_dir, f"input_{i}.png"))
        save_image(hr, os.path.join(save_output_dir, f"target_{i}.png"))

        # Visualize first 5
        if i < 5:
            grid = make_grid(torch.cat([lr, pred, hr], dim=0), nrow=3).cpu()
            plt.figure(figsize=(12, 5))
            plt.imshow(grid.permute(1, 2, 0).numpy())
            plt.title(f"Sample {i} | SSIM: {ssim_score:.4f} | FPS: {fps_score:.2f}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_output_dir, f"compare_{i}.png"))
            print(f"✅ Saved & displayed sample {i}")
            plt.show(block=True)
            plt.close()

# Final stats
print("\n✅ Student Model Evaluation Complete")
print(f"Average SSIM: {total_ssim / num_samples:.4f}")
print(f"Average PSNR: {total_psnr / num_samples:.2f} dB")
print(f"Average FPS : {total_fps / num_samples:.2f}")
