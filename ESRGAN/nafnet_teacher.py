import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        return identity + x * self.scale

class NAFNetTeacher(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=32, num_blocks=16):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, width, 3, padding=1)
        self.blocks = nn.Sequential(*[SimpleNAFBlock(width) for _ in range(num_blocks)])
        self.exit = nn.Conv2d(width, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)
        return x

    def extract_features(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        return x


def load_nafnet_teacher(ckpt_path=None, device='cpu'):
    model = NAFNetTeacher()
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)

if __name__ == '__main__':
    import torch
    model = load_nafnet_teacher(device='cpu')
    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape)

