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


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, max_norm=1.0):
    model.train()
    running_loss = 0.0
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
       
    n = len(loader)
    return running_loss/n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    for img, seg, _ in tqdm(loader):
        img, seg = img.to(device), seg.to(device)
        logits = model(img)
        loss = criterion(logits, seg)
        val_loss += loss.item()
    n = len(loader)
    return val_loss/n

def main():
    epochs = 10
    best_model = "best_unet3d.pt"
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(n_channels=4, n_classes=4).to(device)
    if resume:
        model.load_state_dict(torch.load(best_model))
        print(model)
        print('model loaded')
    criterion = CombinedLoss(weight_dice=0.7, weight_ce=0.3) 
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()

    train_ds = BraTSDataset(root="data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", split='train')
    val_ds   = BraTSDataset(root="data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", split='val')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    
    best_loss = 0.0 
    for epoch in range(1, epochs+1):
        print('epoch: ', epoch)
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va_loss = validate(model, val_loader, criterion, device)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch:03d} | Train L {tr_loss:.4f} | Val L {va_loss:.4f}")
        if va_loss > best_loss:
            best_loss = va_loss
            torch.save(model.state_dict(), best_model)


if __name__ == "__main__":
    main()
