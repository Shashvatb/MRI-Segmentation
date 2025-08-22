import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class UNet3D(nn.Module):
    def __init__(self, n_channels=4, n_classes=4):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))
        
        self.bottleneck = DoubleConv(256, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)         # [B, 32, D, H, W]
        x2 = self.down1(x1)      # [B, 64, D/2, H/2, W/2]
        x3 = self.down2(x2)      # [B, 128, D/4, H/4, W/4]
        x4 = self.down3(x3)      # [B, 256, D/8, H/8, W/8]
        
        x = self.bottleneck(x4) # [B, 512, D/8, H/8, W/8]

        x = self.up3(x) # [B, 128, D/4, H/4, W/4]
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='trilinear', align_corners=False)
        x = self.conv3(torch.cat([x, x3], dim=1))
        x = self.up2(x) # [B, 64, D/2, H/2, W/2]
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up1(x) # [B, 32, D, H, W]
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x = self.conv1(torch.cat([x, x1], dim=1))
        logits = self.outc(x)
        return logits