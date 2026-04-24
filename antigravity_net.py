import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AntigravityNet(nn.Module):
    """
    A lightweight Refinement Network (U-Net variant) 
    that takes the original image and the coarse prediction mask
    to output a refined probability mask.
    """
    def __init__(self, img_channels=1, mask_channels=1, out_channels=1):
        super().__init__()
        
        # Input channel: Image (1 or 3) + Coarse Mask (1)
        in_ch = img_channels + mask_channels
        
        self.encoder1 = DoubleConv(in_ch, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConv(32, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(64, 32)
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(32, 16)
        
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, image, coarse_mask):
        # image: [B, C, H, W]
        # coarse_mask: [B, 1, H, W]
        x = torch.cat([image, coarse_mask], dim=1)
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        
        # Decoder
        d1 = self.up1(e3)
        # Pad if necessary for odd dimensions
        diffY = e2.size()[2] - d1.size()[2]
        diffX = e2.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([e2, d1], dim=1)
        d1 = self.decoder1(d1)
        
        d2 = self.up2(d1)
        diffY = e1.size()[2] - d2.size()[2]
        diffX = e1.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([e1, d2], dim=1)
        d2 = self.decoder2(d2)
        
        out = self.final_conv(d2)
        return torch.sigmoid(out)

class DSFN_Antigravity_Pipeline(nn.Module):
    def __init__(self, dsfn_model, anti_model):
        super().__init__()
        self.dsfn = dsfn_model
        self.antigravity = anti_model
        
    def forward(self, x):
        dsfn_out, dsfn_v1, dsfn_v2 = self.dsfn(x)
        # dsfn_out is the probability map [B, 1, H, W]
        refined_out = self.antigravity(x, dsfn_out)
        return refined_out, dsfn_out
