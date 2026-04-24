import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np

class DRIVEDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=(512, 512)):
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        
        # Searching for images
        self.image_paths = []
        self.mask_paths = []
        
        # Determine whether it's standard DRIVE structure or flattened
        img_dir = os.path.join(root_dir, 'images')
        gt_dir = os.path.join(root_dir, '1st_manual')
        
        if os.path.exists(img_dir) and os.path.exists(gt_dir):
            self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
            # Assume corresponding masks have similar prefixes
            all_masks = sorted(glob.glob(os.path.join(gt_dir, '*.*')))
            self.mask_paths = [self.find_matching_mask(img_p, all_masks) for img_p in self.image_paths]
        else:
            # Flattened structure fallback
            self.image_paths = sorted([p for p in glob.glob(os.path.join(root_dir, '*.*')) if 'manual' not in p.lower() and 'mask' not in p.lower()])
            self.mask_paths = [self.find_matching_mask_flat(img_p, root_dir) for img_p in self.image_paths]

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def find_matching_mask(self, img_path, all_masks):
        base_name = os.path.basename(img_path).split('_')[0]
        for m in all_masks:
            if base_name in m:
                return m
        return None

    def find_matching_mask_flat(self, img_path, root_dir):
        base_name = os.path.basename(img_path).split('_')[0]
        possible_mask = os.path.join(root_dir, base_name + '_manual1.tif')
        if os.path.exists(possible_mask):
            return possible_mask
        return None

    def transform(self, image, mask):
        # Resize
        image = TF.resize(image, self.image_size)
        
        if mask is not None:
            mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flipping
        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
            
        # Random vertical flipping
        if self.is_train and random.random() > 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image) # C x H x W
        
        # Standardize image (convert RGB to gray if it's 3 channels like they did in read_DRIVE)
        if image.shape[0] == 3:
            # simple grayscale conversion: 0.2989 R + 0.5870 G + 0.1140 B
            gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
            image = gray.unsqueeze(0)

        # CLAHE can be approximated by normalizing
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        if mask is not None:
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float() # Threshold mask to binary (0/1)
        else:
            # If no mask exists, provide dummy mask (for inference without GT)
            mask = torch.zeros_like(image)
            
        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        mask_path = self.mask_paths[idx]
        mask = None
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            
        image, mask = self.transform(image, mask)
        
        return image, mask

def get_drive_dataloaders(batch_size=2):
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DRIVE/training'))
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DRIVE/test'))
    
    if not os.path.exists(train_dir):
         print(f"Warning: {train_dir} does not exist. Using dummy dataset.")
         # fallback dummy data
         train_dataset = [(torch.rand(1, 512, 512), torch.zeros(1, 512, 512)) for _ in range(4)]
         val_dataset = [(torch.rand(1, 512, 512), torch.zeros(1, 512, 512)) for _ in range(4)]
    else:
        train_dataset = DRIVEDataset(train_dir, is_train=True)
        val_dataset = DRIVEDataset(test_dir, is_train=False)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
