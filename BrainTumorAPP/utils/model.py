import torch
import torch.nn as nn

class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(ImprovedUNet3D, self).__init__()
        features = init_features

        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = self._block(features * 4, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
