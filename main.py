from brats_dataset import BraTSDataset
from segmentation_model import UNet3D
from losses import CombinedLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

def dice_score_per_class(logits, targets, eps=1e-6):
    # returns mean dice over classes (excluding background optional)
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0,4,1,2,3).float()
    dims = (0,2,3,4)
    inter = torch.sum(probs * onehot, dims)
    denom = torch.sum(probs + onehot, dims)
    dice = (2*inter + eps) / (denom + eps)
    return dice.mean().item()


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, max_norm=1.0):
    model.train()
    running_loss, running_dice = 0.0, 0.0
    for img, seg, _ in tqdm(loader):
        img, seg = img.to(device), seg.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(img)
            loss = criterion(logits, seg)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        with torch.no_grad():
            running_dice += dice_score_per_class(logits, seg)
    n = len(loader)
    return running_loss/n, running_dice/n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    for img, seg, _ in tqdm(loader):
        img, seg = img.to(device), seg.to(device)
        logits = model(img)
        loss = criterion(logits, seg)
        val_loss += loss.item()
        val_dice += dice_score_per_class(logits, seg)
    n = len(loader)
    return val_loss/n, val_dice/n

def main():
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(n_channels=4, n_classes=4).to(device)
    criterion = CombinedLoss(weight_dice=0.7, weight_ce=0.3) 
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()

    train_ds = BraTSDataset(root="data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", split='train')
    val_ds   = BraTSDataset(root="data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", split='val')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    
    best_dice, best_path = 0.0, "best_unet3d.pt"
    for epoch in range(1, epochs+1):
        print('epoch: ', epoch)
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va_loss, va_dice = validate(model, val_loader, criterion, device)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch:03d} | Train L {tr_loss:.4f} D {tr_dice:.4f} | Val L {va_loss:.4f} D {va_dice:.4f}")
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), best_path)


if __name__ == "__main__":
    main()
