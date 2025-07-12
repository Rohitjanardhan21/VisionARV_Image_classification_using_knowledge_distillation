# File: vgg_perceptual_multi.py

import torch
import torch.nn as nn
from torchvision import models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer='relu3_1', use_input_norm=True):
        super().__init__()

        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Select layer mapping
        layer_map = {
            'relu1_1': 2,
            'relu2_1': 7,
            'relu3_1': 12,
            'relu4_1': 21
        }
        assert layer in layer_map, f"Invalid layer '{layer}' for VGGFeatureExtractor."

        self.features = nn.Sequential(*[vgg[i] for i in range(layer_map[layer] + 1)])
        self.use_input_norm = use_input_norm

        if self.use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)

if __name__ == '__main__':
    model = VGGFeatureExtractor(layer='relu3_1').eval()
    dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out.shape)
