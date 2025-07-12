#student_model_enhanced.py
import torch
import torch.nn as nn
import os

# --- Residual Block with InstanceNorm and LeakyReLU ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# --- Channel Attention ---
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.fc(self.avg_pool(x))
        return x * attn

# --- Spatial Attention ---
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([max_out, mean_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

# --- Student Network (Restored to match 32-channel checkpoint) ---
class StudentNetEnhanced(nn.Module):
    def __init__(self, channels=32, num_blocks=6):  # Match the checkpoint parameters
        super().__init__()
        self.entry = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.exit = nn.Sequential(
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        feat = self.entry(x)
        feat = self.blocks(feat)
        feat = self.ca(feat)
        feat = self.sa(feat)
        out = self.exit(feat)
        return self.refine(out)

    def extract_features(self, x):
        feat = self.entry(x)
        feat = self.blocks(feat)
        return feat

def load_student_model(path=None, device='cpu'):
    model = StudentNetEnhanced(channels=32, num_blocks=6).to(device)  # Match checkpoint parameters
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    return model

if __name__ == "__main__":
    model = load_student_model(device='cuda' if torch.cuda.is_available() else 'cpu')
    dummy = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
    output = model(dummy)
    print("✅ Output shape:", output.shape)
    from torchvision.utils import save_image
    save_image(output.detach().cpu(), "student_output_demo.png")
    print("💾 Output saved.")
