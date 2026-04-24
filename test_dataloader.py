import torch
from data_loader_drive import get_drive_dataloaders

def test_dataloader():
    print("Initializing dataloaders...")
    train_loader, val_loader = get_drive_dataloaders(batch_size=2)
    
    print(f"Train Loader Size: {len(train_loader)} batches")
    print(f"Val Loader Size: {len(val_loader)} batches")
    
    if len(train_loader) == 0:
        print("ERROR: Train loader is empty. Is the dataset correctly placed?")
        return

    print("Fetching one batch from train_loader...")
    images, masks = next(iter(train_loader))
    
    print(f"Image Shape: {images.shape} | Type: {images.dtype}")
    print(f"Mask Shape: {masks.shape} | Type: {masks.dtype}")
    
    print(f"Image Value Range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"Mask Unique Values: {torch.unique(masks).tolist()}")
    
    assert images.dim() == 4, "Image should be 4D tensor (B, C, H, W)"
    assert masks.dim() == 4, "Mask should be 4D tensor (B, C, H, W)"
    
    print("\nSUCCESS: Dataloader output verified.")

if __name__ == '__main__':
    test_dataloader()
