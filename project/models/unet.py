from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for binary segmentation.

    Input:  [B, 3, H, W]
    Output: logits [B, 1, H, W]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ResNet18UNet(nn.Module):
    """Memory-efficient U-Net with an ImageNet-pretrained ResNet-18 encoder."""

    def __init__(
        self,
        out_channels: int = 1,
        pretrained: bool = True,
        encoder_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if pretrained and encoder_state_dict is not None:
            raise ValueError("choose torchvision weights or an explicit encoder state, not both")
        encoder = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        if encoder_state_dict is not None:
            encoder.load_state_dict(encoder_state_dict, strict=True)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.up4 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up1 = Up(64 + 64, 64)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv(64, 32),
            OutConv(32, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = self.final_up(x)
        if x.shape[-2:] != input_size:
            x = nn.functional.interpolate(
                x, size=input_size, mode="bilinear", align_corners=False
            )
        return x


def architecture_metadata(name: str) -> dict[str, object]:
    if name == "unet":
        return {"name": "UNet", "in_channels": 3, "out_channels": 1, "base_channels": 64}
    if name == "resnet18_unet":
        return {
            "name": "ResNet18UNet",
            "in_channels": 3,
            "out_channels": 1,
            "encoder": "resnet18",
        }
    raise ValueError(f"Unknown segmentation architecture: {name}")


def architecture_name_from_metadata(metadata: object) -> str:
    if not isinstance(metadata, dict):
        return "unet"
    name = str(metadata.get("name", "UNet"))
    if name == "UNet":
        return "unet"
    if name == "ResNet18UNet":
        return "resnet18_unet"
    raise ValueError(f"Unsupported checkpoint architecture metadata: {metadata!r}")


def build_segmentation_model(
    name: str,
    pretrained: bool = False,
    *,
    encoder_state_dict: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    if name == "unet":
        if encoder_state_dict is not None:
            raise ValueError("plain U-Net does not accept a ResNet encoder state")
        return UNet(in_channels=3, out_channels=1, base_channels=64)
    if name == "resnet18_unet":
        return ResNet18UNet(
            out_channels=1,
            pretrained=pretrained,
            encoder_state_dict=encoder_state_dict,
        )
    raise ValueError(f"Unknown segmentation architecture: {name}")
