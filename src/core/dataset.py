import os
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from torchvision import transforms
import torchvision.transforms as transforms
import numpy as np
import json

class InstanceRelocationDataset(Dataset):
    def __init__(self, mask_dir, offset_root, ratios, transform=None):
        
        self.mask_dir = mask_dir
        self.offset_root = offset_root
        self.ratios = ratios
        self.filenames = sorted(os.listdir(mask_dir))
        self.transform = transform

        for fname in self.filenames:
            for ratio in self.ratios:
                offset_path = os.path.join(offset_root, ratio, fname.replace('.png', '.json'))
                if not os.path.exists(offset_path):
                    raise FileNotFoundError(f"Missing offset file: {offset_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        base_name = self.filenames[idx]
        mask_path = os.path.join(self.mask_dir, base_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        if self.transform:
            mask_tensor = self.transform(mask_tensor)

        multi_offsets = []
        for ratio in self.ratios:
            offset_path = os.path.join(self.offset_root, ratio, base_name.replace('.png', '.json'))
            with open(offset_path, 'r') as f:
                offset_data = json.load(f)
            
            offsets_x = torch.tensor(offset_data["offsets_x"], dtype=torch.float32)

            multi_offsets.append(offsets_x)
        
        target = torch.cat(multi_offsets, dim=0)

        return mask_tensor, target

class Preprocess:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size  # (height, width)

    def __call__(self, image):
        _, height, width = image.shape
        target_height, target_width = self.target_size

        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left
        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top

        pad_left = max(pad_left, 0)
        pad_top = max(pad_top, 0)
        pad_right = max(pad_right, 0)
        pad_bottom = max(pad_bottom, 0)
        
        padded_image = np.pad(image.squeeze(0).numpy(), ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        return torch.tensor(padded_image).unsqueeze(0)
    
class ConvertToBF16:
    def __call__(self, image):
        
        return image.to(torch.bfloat16)

def get_transform_original(target_size):
    return transforms.Compose([
        Preprocess(target_size=target_size),  # （H，W）
        transforms.Normalize(mean=[0.5], std=[0.5])
        #ConvertToBF16()    
    ])

def build_dataloader(cfg, split='train'):
    
    assert split in ['train', 'val', 'test']
    
    data_root = cfg.dataset.root  
    mask_dir = os.path.join(data_root, f"{split}set/masks/")
    offset_root = os.path.join(data_root, f"{split}set/offsets/")
    
    target_size = tuple(cfg.dataset.img_size) # (640, 640)
    transform = get_transform_original(target_size=target_size)
    
    dataset = InstanceRelocationDataset(
        mask_dir=mask_dir,
        offset_root=offset_root,
        ratios=cfg.dataset.ratios, # ['16_9', '1_1', '9_16']
        transform=transform
    )
    
    is_shuffle = (split == 'train')
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.train.batch_size if split=='train' else cfg.test.batch_size,
        shuffle=is_shuffle,
        num_workers=cfg.system.num_workers,
        pin_memory=True
    )
    
    return loader
