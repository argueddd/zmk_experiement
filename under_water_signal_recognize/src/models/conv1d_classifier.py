import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DRowWiseClassifier(nn.Module):
    def __init__(self, num_rows, num_classes, conv_out_channels=64, kernel_size=5):
        super().__init__()
        self.num_rows = num_rows
        self.conv_out_channels = conv_out_channels

        # 1D 卷积用于提取每行序列特征
        self.conv = nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 每个通道池化为1个值

        # 全连接层：输入维度是 199 行 * 每行 64维特征
        self.fc = nn.Linear(num_rows * conv_out_channels, num_classes)

    def forward(self, x):
        # 输入: [B, 1, 199, 310]
        B = x.size(0)
        x = x.squeeze(1)  # [B, 199, 310]

        # 变成 [B*199, 1, 310]，每行是一个序列
        x = x.view(B * self.num_rows, 1, -1)

        # 卷积提特征 → [B*199, 64, 310]
        x = self.conv(x)

        # 池化压缩时间维 → [B*199, 64, 1]
        x = self.pool(x)

        # 去掉最后一维 → [B*199, 64]
        x = x.squeeze(-1)

        # 拼回 batch：每个样本是 [199 * 64]
        x = x.view(B, self.num_rows * self.conv_out_channels)

        # 分类输出
        x = self.fc(x)  # [B, num_classes]
        # 使用softmax转换为概率分布
        x = F.log_softmax(x, dim=1)

        # 计算累积概率
        # x = torch.cumsum(x, dim=1)
        return x


def cumulative_bce_loss(output, target):
    """计算累积概率输出与目标累积标签之间的BCE损失"""
    return torch.mean(torch.sum(torch.abs(output - target), dim=1))


def bce_loss(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)


if __name__ == '__main__':
    pass
