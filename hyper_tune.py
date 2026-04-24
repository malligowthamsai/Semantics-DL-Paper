import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from DSFM_Net import DSFM_Net
from antigravity_net import AntigravityNet, DSFN_Antigravity_Pipeline
from data_loader_drive import get_drive_dataloaders
from losses import DiceLoss

# Helper function for evaluation metrics
def get_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob > threshold).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5 # if only one class present
    return acc, f1, prec, rec, auc

def evaluate(model, val_loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            refined_out, dsfn_out = model(images)
            
            all_targets.append(masks.cpu().numpy().flatten())
            all_preds.append(refined_out.cpu().numpy().flatten())
            
    if len(all_targets) == 0:
        return 0, 0, 0, 0, 0
        
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    return get_metrics(all_targets, all_preds)

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    print(f"Config: {config}")
    
    # Dataloaders
    train_loader, val_loader = get_drive_dataloaders(batch_size=config['batch_size'])
    
    # Model Setup
    dsfn = DSFM_Net(channels=1, classes=1).to(device)
    anti_net = AntigravityNet(img_channels=1, mask_channels=1, out_channels=1).to(device)
    model = DSFN_Antigravity_Pipeline(dsfn, anti_net).to(device)
    
    # Loss setup
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4) # AdamW specifies weight_decay explicitly
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    best_auc = 0.0
    epochs_no_improve = 0
    early_stop_patience = 5
    
    # Evaluation on baseline initialization just to test pipeline
    if not train_loader:
        print("No training data found, aborting training...")
        return None
        
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            refined_out, dsfn_out = model(images)
            
            # Joint loss: primary BCE on dsfn coarse, and mixed on refined
            loss_dsfn = bce_loss(dsfn_out, masks)
            
            if config['fusion'] == 'add':
                # Simplified representation of how we might alter loss per strategy, but fusion mainly affects architecture
                pass
                
            if 'BCE' in config['losses'] and 'Dice' in config['losses']:
                loss_refined = bce_loss(refined_out, masks) + dice_loss(refined_out, masks)
            else:
                loss_refined = bce_loss(refined_out, masks)
                
            loss = loss_dsfn + loss_refined
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation Phase
        acc, f1, prec, rec, auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{config['epochs']} - Loss: {train_loss:.4f} | Val AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            # Could save model here
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    return best_auc

def hyperparameter_tuning():
    learning_rates = [1e-4, 1e-3]
    batch_sizes = [2, 4] # Keep small to fit memory locally
    fusion_strategies = ['concat'] # Based on prompt constraint
    losses_combos = [['BCE', 'Dice']]
    
    best_score = 0.0
    best_config = None
    
    total_configs = len(learning_rates) * len(batch_sizes) * len(losses_combos)
    idx = 1
    for lr in learning_rates:
        for bs in batch_sizes:
            for lc in losses_combos:
                config = {
                    'lr': lr, 
                    'batch_size': bs, 
                    'fusion': 'concat', 
                    'losses': lc,
                    'epochs': 5  # Keep small for demonstration/local testing
                }
                print(f"\n--- Running Config {idx}/{total_configs} ---")
                score = train_model(config)
                if score and score > best_score:
                    best_score = score
                    best_config = config
                idx += 1
                
    print("\n==============================")
    print("HYPERPARAMETER TUNING COMPLETE")
    print("==============================")
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation AUC: {best_score:.4f}")

if __name__ == '__main__':
    hyperparameter_tuning()
