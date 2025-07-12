# File: ESRGAN/train_kd.py

import os, torch, math, time
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from pytorch_msssim import ssim
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from dataset import ImagePairDataset
from student_model_enhanced import StudentNetEnhanced
from nafnet_teacher import load_nafnet_teacher
from vgg_perceptual_multi import VGGFeatureExtractor

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
train_lr = 'data/Dataset/Split/train/blurry'
train_hr = 'data/Dataset/Split/train/sharp'
val_lr = 'data/Dataset/Split/val/blurry'
val_hr = 'data/Dataset/Split/val/sharp'
ckpt_teacher = 'checkpoints/motion_deblurring_teacher_best.pth'
ckpt_student = 'checkpoints/best_student_model.pth'

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset & Loader
train_set = ImagePairDataset(train_lr, train_hr, transform=transform)
val_set = ImagePairDataset(val_lr, val_hr, transform=transform)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Models
student = StudentNetEnhanced(channels=32, num_blocks=6).to(DEVICE)
teacher = load_nafnet_teacher(ckpt_teacher, device='cpu').eval()
perceptual = VGGFeatureExtractor(layer='relu3_1').to(DEVICE).eval()

# Loss setup
l1 = nn.L1Loss()
scaler = GradScaler()
optimizer = torch.optim.Adam(student.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Sobel Edge Loss
def sobel_filter(img):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    img_gray = torch.mean(img, dim=1, keepdim=True)
    grad_x = F.conv2d(img_gray, sobel_x, padding=1)
    grad_y = F.conv2d(img_gray, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

def edge_loss(pred, target):
    return F.l1_loss(sobel_filter(pred), sobel_filter(target))

# PSNR
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse.item() > 0 else 100

# Validation
@torch.no_grad()
def validate():
    student.eval()
    total_ssim, total_psnr, total_fps = 0, 0, 0
    for lr, hr in val_loader:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        start = time.time()
        with autocast():
            pred = student(lr)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        end = time.time()

        # Convert to float32 to avoid Half errors
        pred = torch.clamp(pred, 0.0, 1.0).float()
        hr = torch.clamp(hr, 0.0, 1.0).float()

        total_ssim += ssim(pred, hr, data_range=1.0).item()
        total_psnr += psnr(pred, hr)
        total_fps += 1 / (end - start)

    return total_ssim / len(val_loader), total_psnr / len(val_loader), total_fps / len(val_loader)

# Resume training
if os.path.exists(ckpt_student):
    print(f"[INFO] Resuming from checkpoint: {ckpt_student}")
    student.load_state_dict(torch.load(ckpt_student))

# Training Loop
best_ssim = 0
print("\n[INFO] Starting student KD training...")
for epoch in range(40):
    print(f"\n[INFO] === Epoch {epoch+1}/40 ===")
    student.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for lr, hr in loop:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        with torch.no_grad():
            teacher_pred = teacher(lr.to('cpu')).to(DEVICE)
            teacher_feat = teacher.extract_features(lr.to('cpu')).to(DEVICE)

        with autocast():
            student_pred = student(lr)
            student_feat = student.extract_features(lr)

            # Match feature map sizes
            if student_feat.shape != teacher_feat.shape:
                teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[2:], mode='bilinear')

            student_pred = torch.clamp(student_pred, 0.0, 1.0)
            teacher_pred = torch.clamp(teacher_pred, 0.0, 1.0)

            recon = l1(student_pred, hr)
            percept = l1(perceptual(student_pred), perceptual(hr))
            edge = edge_loss(student_pred, hr)
            feat_distill = l1(student_feat, teacher_feat.detach())
            ssim_l = 1 - ssim(student_pred.float(), hr.float(), data_range=1.0)
            kd = l1(student_pred, teacher_pred)

            total_loss = (
                1.0 * recon +
                0.6 * percept +
                0.5 * edge +
                0.4 * feat_distill +
                0.3 * ssim_l +
                0.3 * kd
            )

        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        loop.set_postfix(loss=total_loss.item())

    scheduler.step()
    val_ssim, val_psnr, val_fps = validate()
    print(f"[INFO] Epoch {epoch+1} Validation - SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.2f}, FPS: {val_fps:.2f}")

    if val_ssim > best_ssim:
        best_ssim = val_ssim
        os.makedirs(os.path.dirname(ckpt_student), exist_ok=True)
        torch.save(student.state_dict(), ckpt_student)
        print("[INFO] Saved new best student model")
