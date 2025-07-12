# File: ESRGAN/utils/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImagePairDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, limit=None):
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])[:limit]
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])[:limit]
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_paths[idx]).convert('RGB')
        hr = Image.open(self.hr_paths[idx]).convert('RGB')
        return self.transform(lr), self.transform(hr)
