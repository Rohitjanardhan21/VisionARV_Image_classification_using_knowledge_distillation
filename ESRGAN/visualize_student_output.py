# File: ESRGAN/visualize_student_output.py

import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import time
import torch.nn.functional as F
import random

from student_model_enhanced import load_student_model

# Ensure utils is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from dataset import ImagePairDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
val_lr = 'data/Dataset/Split/val/blurry'
val_hr = 'data/Dataset/Split/val/sharp'
ckpt_student = 'checkpoints/best_student_model.pth'
output_dir = 'results/student_output'
os.makedirs(output_dir, exist_ok=True)

# Transform with higher resolution for HD output
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Dataset & Dataloader
val_set = ImagePairDataset(val_lr, val_hr, transform=transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Load Student
student = load_student_model().to(DEVICE)
student.load_state_dict(torch.load(ckpt_student, map_location=DEVICE))
student.eval()

# Sharpening kernel for post-processing
def sharpen_image(tensor, strength=0.2):
    """Apply unsharp masking to enhance details"""
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=tensor.device) / 16.0
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    blurred = F.conv2d(tensor.unsqueeze(0), kernel, padding=1, groups=3)
    sharpened = tensor + strength * (tensor - blurred.squeeze(0))
    return torch.clamp(sharpened, 0.0, 1.0)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else 100

# Visualize random samples
N = 5  # Number of random samples to show
num_samples = len(val_set)
random_indices = random.sample(range(num_samples), min(N, num_samples))

print(f"\n🔍 Evaluating {len(random_indices)} random samples from validation set...")
total_ssim, total_psnr, total_fps = 0, 0, 0

for idx, i in enumerate(random_indices):
    lr, hr = val_set[i]
    lr, hr = lr.unsqueeze(0).to(DEVICE), hr.unsqueeze(0).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        pred = student(lr)
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    end = time.time()

    # Clamp to valid range and apply sharpening for HD output
    lr_disp = torch.clamp(lr.squeeze(0), 0.0, 1.0).cpu()
    pred = torch.clamp(pred.squeeze(0), 0.0, 1.0)
    pred = sharpen_image(pred, strength=0.2)
    pred = pred.cpu()
    hr_disp = torch.clamp(hr.squeeze(0), 0.0, 1.0).cpu()

    # Metrics
    ssim_score = ssim(pred.unsqueeze(0), hr_disp.unsqueeze(0), data_range=1.0).item()
    psnr_score = psnr(pred, hr_disp)
    fps = 1 / (end - start)

    total_ssim += ssim_score
    total_psnr += psnr_score
    total_fps += fps

    # Combine and plot
    combined = torch.stack([lr_disp, pred, hr_disp])  # shape: (3, C, H, W)
    grid = make_grid(combined, nrow=3)
    plt.figure(figsize=(12, 5))
    plt.imshow(to_pil_image(grid), interpolation='nearest')
    plt.title(f"Sample {i} | SSIM: {ssim_score:.4f} | PSNR: {psnr_score:.2f} | FPS: {fps:.2f}\nInput | Prediction | Ground Truth")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_random_{i}.png")
    plt.show()

print("\n✅ Student Model Evaluation Results")
print(f"Average SSIM: {total_ssim / len(random_indices):.4f}")
print(f"Average PSNR: {total_psnr / len(random_indices):.2f}")
print(f"Average FPS : {total_fps / len(random_indices):.2f}")
