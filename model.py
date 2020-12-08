import torch
import torch.nn as nn
from blocks import ResBlocks, UpConv2dBlock, Conv2dBlock


class Transfer(nn.Module):
    def __init__(self, args):
        super(Transfer, self).__init__()
        c_up = args.c_up  # 64
        down = args.down  # 2
        self.model = [
            Conv2dBlock(7, c_up, 7, 1, 3, norm='in', pad_type='reflect'),  # RGB + mask + target
        ]
        for i in range(down):
            self.model.append(Conv2dBlock(c_up, 2 * c_up, 4, 2, 1, norm='in', pad_type='reflect'))
            c_up *= 2
        self.model.append(ResBlocks(5, c_up, norm='in', activation='relu', pad_type='reflect'))
        for i in range(down):
            self.model.append(UpConv2dBlock(c_up, norm='in', activation='relu', pad_type='reflect'))
            c_up //= 2
        self.model.append(Conv2dBlock(c_up, 3, 7, 1, padding=3, norm='none', activation='none', pad_type='reflect'))
        self.model = nn.Sequential(*self.model)

    def to_rgb(self, x):
        return 127.5 * (x + 1)

    def forward(self, x):
        """
        Args:
            x: (B, 7, ts, ts): RGB + mask + target

        Returns: (B, 3, ts, ts): RGB

        """
        return self.model(x)


# simple style transfer
class Refiner(nn.Module):
    def __init__(self, args):
        super(Refiner, self).__init__()
        c_up = args.c_up // 2  # 32
        down = args.down  # 2
        self.model = [
            Conv2dBlock(6, c_up, 7, 1, 3, norm='in', pad_type='reflect'),  # RGB + target
        ]
        for i in range(down):
            self.model.append(Conv2dBlock(c_up, 2 * c_up, 4, 2, 1, norm='in', pad_type='reflect'))
            c_up *= 2
        self.model.append(ResBlocks(5, c_up, norm='in', activation='relu', pad_type='reflect'))
        for i in range(down):
            self.model.append(UpConv2dBlock(c_up, norm='in', activation='relu', pad_type='reflect'))
            c_up //= 2
        self.model.append(Conv2dBlock(c_up, 3, 7, 1, padding=3, norm='none', activation='none', pad_type='reflect'))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """
        Args:
            x: (B, 6, ts, ts): RGB + target

        Returns: (B, 3, ts, ts): RGB

        """
        return self.model(x)


# Image gradient
class LaplacianFilter(nn.Module):
    def __init__(self):
        super(LaplacianFilter, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float().reshape_as(self.conv.weight)
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        # x: (B, 3, H, W) RGB image, [-1, 1]
        red_img_tensor = x[:, 0, :, :].unsqueeze(1)
        green_img_tensor = x[:, 1, :, :].unsqueeze(1)
        blue_img_tensor = x[:, 2, :, :].unsqueeze(1)

        red_gradient_tensor = self.conv(red_img_tensor).squeeze(1)
        green_gradient_tensor = self.conv(green_img_tensor).squeeze(1)
        blue_gradient_tensor = self.conv(blue_img_tensor).squeeze(1)

        return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor  # (B, H, W)
