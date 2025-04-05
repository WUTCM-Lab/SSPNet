import torch
import torch.nn as nn
from Visualization import cluster_and_visualize1
class FourierRelativePositionModel(nn.Module):
    def __init__(self, depth, height, width, number=5):
        super(FourierRelativePositionModel, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.number = number
    def forward(self, x):
        x = x.squeeze(2)

        fft_x = torch.fft.fftn(x, dim=[2, 3, 4])  # 对每个批次的图像进行3D傅里叶变换
        fft_x = torch.fft.fftshift(fft_x, dim=[2, 3, 4])  # 将零频移到中心
        output = fft_x
        low_freq, high_freq = self.separate_frequency(output)
        low_freq = torch.fft.ifftn(low_freq, dim=[2, 3, 4])
        high_freq = torch.fft.ifftn(high_freq, dim=[2, 3, 4])
        low_freq = low_freq.real  # 只保留实部
        high_freq = high_freq.real

        return low_freq, high_freq

    def separate_frequency(self, fft_x):
        batch_size, channels, depth, height, width = fft_x.shape

        center_d, center_h, center_w = depth // 2, height // 2, width // 2
        low_freq_mask = torch.zeros_like(fft_x)
        high_freq_mask = torch.zeros_like(fft_x)
        low_freq_mask[:, :, center_d - self.number:center_d + self.number, center_h - self.number:center_h + self.number, center_w - self.number:center_w + self.number] = 1
        high_freq_mask[:, :, :center_d - self.number, :, :] = 1
        high_freq_mask[:, :, center_d + self.number:, :, :] = 1
        high_freq_mask[:, :, :, :center_h - self.number, :] = 1
        high_freq_mask[:, :, :, center_h + self.number:, :] = 1
        high_freq_mask[:, :, :, :, :center_w - self.number] = 1
        high_freq_mask[:, :, :, :, center_w + self.number:] = 1
        low_freq = fft_x * low_freq_mask
        high_freq = fft_x * high_freq_mask

        return low_freq, high_freq

class TripleCrossAttentionHigh(nn.Module):
    def __init__(self, channels):
        super(TripleCrossAttentionHigh, self).__init__()

        self.softmax = nn.Softmax(-1)

        self.pool_d = nn.AdaptiveAvgPool3d((1, None, None))  # 低频特征使用较大的池化核
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, None))  # 低频特征使用较大的池化核
        self.pool_w = nn.AdaptiveAvgPool3d((None, None, 1))  # 低频特征使用较大的池化核

        self.gn = nn.GroupNorm(channels, channels)
        self.conv1x1 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, d, h, w = x.size()  # 获取输入特征张量的大小
        group_x = x.reshape(b, -1, d, h, w)  # 将输入特征张量按通道分组
        x_d = self.pool_d(group_x)
        x_h = self.pool_h(group_x).permute(0, 1, 3, 4, 2)
        x_w = self.pool_w(group_x).permute(0, 1, 4, 3, 2)
        dhw = self.conv1x1(torch.cat([x_d, x_h, x_w], dim=4))
        x_d, x_h, x_w = torch.split(dhw, [d, h, d], dim=4)
        x1 = self.gn(
            group_x * ((x_d.sigmoid().permute(0, 1, 2, 4, 3) @ x_h.sigmoid()).permute(0, 1, 3, 4, 2) @ x_w.sigmoid().permute(0, 1, 4, 2, 3)).sigmoid())
        return x1.reshape(b, c, d, h, w)

class TripleCrossAttentionLow(nn.Module):
    r""" SpatialCrossAttention for 3D tensors."""

    def __init__(self, channels):
        super(TripleCrossAttentionLow, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((1, 1, None))
        self.gn = nn.GroupNorm(channels, channels)
        self.conv1x1 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w, d = x.size()  # 获取输入特征张量的大小
        group_x = x.reshape(b, -1, h, w, d)  # 将输入特征张量按通道分组

        # 在高度、宽度、深度方向上分别进行平均池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 4, 2)
        x_d = self.pool_d(group_x).permute(0, 1, 4, 2, 3)

        # 将高度、宽度、深度方向上的池化结果拼接，并经过1x1卷积
        hwd = self.conv1x1(torch.cat([x_h, x_w, x_d], dim=2))
        x_h, x_w, x_d = torch.split(hwd, [h, w, d], dim=2)

        # 计算空间交叉注意力
        x1 = self.gn(
            group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 4, 2).sigmoid() * x_d.permute(0, 1, 4, 2, 3).sigmoid())

        return x1.reshape(b, c, h, w, d)

class FusionWithAttention(nn.Module):
    def __init__(self, channels):
        super(FusionWithAttention, self).__init__()
        self.spatial_attention_low_freq = TripleCrossAttentionLow(channels)
        self.spatial_attention_high_freq = TripleCrossAttentionHigh(channels)
        self.FRPM = FourierRelativePositionModel(30, 9, 9, 5)
    def forward(self, x):
        low_freq, high_freq = self.FRPM(x)
        low_freq_attention = self.spatial_attention_low_freq(low_freq)
        high_freq_attention = self.spatial_attention_high_freq(high_freq)
        fused_output = low_freq_attention + high_freq_attention + x
        return fused_output


# 示例用法
if __name__ == '__main__':
    batch_size, channels, depth, height, width = 64, 8, 30, 17, 17
    x = torch.randn(batch_size, channels, depth, height, width)

    model = FourierRelativePositionModel(depth, height, width)
    low_freq, high_freq = model(x)

    fusion_model = FusionWithAttention(channels)
    fused_output = fusion_model(low_freq, high_freq)
    print(fused_output.shape)  # 输出融合后的结果