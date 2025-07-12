import os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from tqdm import tqdm
import time, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from dataset import ImagePairDataset
from nafnet_teacher import load_nafnet_teacher
from vgg_perceptual_multi import VGGFeatureExtractor
from pytorch_msssim import ssim

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
train_lr = 'data/Dataset/Split/train/blurry'
train_hr = 'data/Dataset/Split/train/sharp'
val_lr = 'data/Dataset/Split/val/blurry'
val_hr = 'data/Dataset/Split/val/sharp'
save_path = 'checkpoints/motion_deblurring_teacher_best.pth'

# Dataset & Dataloader
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_set = ImagePairDataset(train_lr, train_hr, transform=transform)
val_set = ImagePairDataset(val_lr, val_hr, transform=transform)

print(f"[DEBUG] Training samples: {len(train_set)}")
print(f"[DEBUG] Validation samples: {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

# Model & Losses
model = load_nafnet_teacher(device=DEVICE)
perceptual_loss = VGGFeatureExtractor().to('cpu').eval()
l1 = nn.L1Loss()
scaler = GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# PSNR
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else 100

# Validation
@torch.no_grad()
def validate():
    model.eval()
    total_ssim, total_psnr, total_fps = 0, 0, 0
    for lr, hr in val_loader:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        start = time.time()
        with autocast():
            pred = model(lr)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        end = time.time()
        pred = torch.clamp(pred.float(), 0.0, 1.0)
        hr = torch.clamp(hr.float(), 0.0, 1.0)
        total_ssim += ssim(pred, hr, data_range=1.0).item()
        total_psnr += psnr(pred, hr).item()
        total_fps += 1 / (end - start)
    return total_ssim / len(val_loader), total_psnr / len(val_loader), total_fps / len(val_loader)

# Training Loop
best_ssim = 0
print("\n[INFO] Starting teacher training on GPU (fallback to CPU if needed)...")
for epoch in range(30):
    print(f"\n[INFO] === Epoch {epoch+1}/30 ===")
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for i, (lr, hr) in enumerate(loop):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        with autocast():
            pred = model(lr)
            pred = torch.clamp(pred.float(), 0.0, 1.0)
            hr = torch.clamp(hr.float(), 0.0, 1.0)
            loss_recon = l1(pred, hr)

        pred_rgb = pred.detach().to('cpu')[:, :3, :, :]
        hr_rgb = hr.detach().to('cpu')[:, :3, :, :]

        vgg_pred_feats = perceptual_loss(pred_rgb)
        vgg_hr_feats = perceptual_loss(hr_rgb)

        loss_perc = sum(F.l1_loss(p, h) for p, h in zip(vgg_pred_feats, vgg_hr_feats)) / len(vgg_pred_feats)

        loss = loss_recon + 0.05 * loss_perc
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    val_ssim, val_psnr, val_fps = validate()
    print(f"[INFO] Epoch {epoch+1} Validation - SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.2f}, FPS: {val_fps:.2f}")
    if val_ssim > best_ssim:
        best_ssim = val_ssim
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print("[INFO] Saved new best teacher model")
