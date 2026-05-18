# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


## fusion module block ##

# 加性融合
class FusionModuleAdd(nn.Module):
    """光SAR特征融合模块"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, "Input must be a list or tuple of 2 tensors"
        return x[0] + x[1]

# 拼接融合
class FusionModuleConcat(nn.Module):
    """Concatenate tensors along a given dimension and reduce channels via Conv."""

    def __init__(self, dimension=1):
        """
        Args:
            dimension (int): Dimension to concatenate on (default is 1, channel dim).
        """
        super().__init__()
        self.d = dimension
        self.conv = None  # 延迟初始化

    def forward(self, x):
        """
        Args:
            x (List[Tensor]): List of 2 tensors to concatenate.
        Returns:
            Tensor: Concatenated and projected output.
        """
        assert isinstance(x, (list, tuple)) and len(x) == 2, "Input must be a list/tuple of 2 tensors"
        x_cat = torch.cat(x, dim=self.d)  # [B, C1+C2, H, W]

        # 延迟初始化卷积层（只在第一次 forward 时创建）
        if self.conv is None:
            in_channels = x_cat.shape[1]
            out_channels = in_channels // 2  # 默认压缩回输入均值通道数
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1).to(x_cat.device)

        return self.conv(x_cat)


# 互补融合模块
class GWF(nn.Module):
    def __init__(self, in_channels):
        super(GWF, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xRGB, xSAR = x[0], x[1]
        out = torch.cat([xRGB, xSAR], dim=1)
        G = self.gate(out)

        PG = xRGB * G
        FG = xSAR * (1 - G)

        return (PG + FG).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class CLSP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return attn


class DFSC(nn.Module):
    def __init__(self, in_channels) -> None:
        super(DFSC, self).__init__()
        self.lsk = CLSP(in_channels)
        self.res_v = ResidualBlock(in_channels, in_channels)
        self.res_t = ResidualBlock(in_channels, in_channels)

    def forward(self, v_feat, t_feat):
        d_feat1 = v_feat - t_feat
        d_feat2 = t_feat - v_feat

        d_vector1 = self.lsk(d_feat1)
        d_vector2 = self.lsk(d_feat2)

        vd_feat = v_feat * d_vector1
        td_feat = t_feat * d_vector2

        v_feat_ = v_feat + td_feat
        t_feat_ = t_feat + vd_feat

        v_feat = v_feat + self.res_v(v_feat_)
        t_feat = t_feat + self.res_t(t_feat_)

        return v_feat, t_feat


class CompFusionModule(nn.Module):
    """光学+SAR特征融合模块，整合GWF+DFSC"""
    def __init__(self, in_channels):
        super(CompFusionModule, self).__init__()
        self.gwf = GWF(in_channels)
        self.dfsc = DFSC(in_channels)

    def forward(self, x):
        x_optical, x_sar = x
        
        v_feat, t_feat = self.dfsc(x_optical, x_sar)

        fused = self.gwf([v_feat, t_feat])

        return fused



## -------------- 跨模态注意力机制 ---------------------##
class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Query, Key, Value投影矩阵
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # 输出变换
        self.output_conv = nn.Conv2d(in_channels, in_channels, 1)
        # self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数

    def forward(self, query_feat, key_feat, value_feat):
        """
        query_feat: 作为查询的特征 (B, C, H, W)
        key_feat: 作为键的特征 (B, C, H, W)  
        value_feat: 作为值的特征 (B, C, H, W)
        """
        batch_size, _, height, width = query_feat.size()
        
        # 投影到低维空间
        query = self.query_conv(query_feat)  # (B, C/r, H, W)
        key = self.key_conv(key_feat)        # (B, C/r, H, W)
        value = self.value_conv(value_feat)  # (B, C, H, W)
        
        # 重塑为矩阵形式
        query = query.view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, N, C/r)
        key = key.view(batch_size, -1, height * width)                        # (B, C/r, N)
        value = value.view(batch_size, -1, height * width)                    # (B, C, N)
        
        # 计算注意力权重 缩放点积注意力机制
        scale = (query.size(-1)) ** 0.5  # sqrt(d_k)
        energy = torch.bmm(query, key) / scale  # (B, N, N)
        attention = F.softmax(energy, dim=-1)  # 沿key维度softmax
        
        # 应用注意力权重
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, -1, height, width)  # (B, C, H, W)
        
        # 残差连接
        out = self.output_conv(out)
        return query_feat + out


# 计算模态相关性 #
class CrossModalCorrelation(nn.Module):
    """
    余弦 + 幅值 的跨模态相关性
    输入为未归一化的 backbone 特征，用于保留幅值与能量信息
    """

    def __init__(self, alpha=0.5, eps=1e-6, detach=True):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.detach = detach

    def forward(self, feat_a, feat_b):
        # feat_a, feat_b: (B,C,H,W)

        if self.detach:
            feat_a = feat_a.detach()
            feat_b = feat_b.detach()

        # -------- 1. 方向相关性（余弦）--------
        a_norm = F.normalize(feat_a, dim=1)
        b_norm = F.normalize(feat_b, dim=1)
        cos_corr = torch.sum(a_norm * b_norm, dim=1, keepdim=True)
        cos_corr = (cos_corr + 1) / 2  # → [0,1]

        # -------- 2. 幅值相关性（能量）--------
        mag_a = torch.norm(feat_a, dim=1, keepdim=True)
        mag_b = torch.norm(feat_b, dim=1, keepdim=True)
        mag_corr = 1 - torch.abs(mag_a - mag_b) / (mag_a + mag_b + self.eps)
        mag_corr = torch.clamp(mag_corr, 0, 1)

        # -------- 3. 融合 --------
        corr = self.alpha * cos_corr + (1 - self.alpha) * mag_corr

        return corr.contiguous()


class DirectionalCrossModalCorrelation(nn.Module):
    """
    方向感知的跨模态互补相关性
    corr(src → tgt): tgt 是否需要 src 来补充
    """

    def __init__(self, alpha=0.6, eps=1e-6, detach=True):
        super().__init__()
        self.alpha = alpha   # cos / deficit 融合比例
        self.eps = eps
        self.detach = detach

    def forward(self, feat_src, feat_tgt):
        """
        feat_src: 提供信息的一方 (key / value)
        feat_tgt: 接收信息的一方 (query)
        return: (B,1,H,W)  越大表示越需要补偿
        """

        if self.detach:
            feat_src = feat_src.detach()
            feat_tgt = feat_tgt.detach()

        # -------- 1. 方向一致性（结构是否可对齐）--------
        src_n = F.normalize(feat_src, dim=1)
        tgt_n = F.normalize(feat_tgt, dim=1)
        cos = torch.sum(src_n * tgt_n, dim=1, keepdim=True)
        cos = (cos + 1) / 2.0   # [0,1]

        # -------- 2. 能量不足（谁弱谁补）--------
        mag_src = torch.norm(feat_src, dim=1, keepdim=True)
        mag_tgt = torch.norm(feat_tgt, dim=1, keepdim=True)

        # tgt 相对于 src 的“信息缺失比例”
        deficit = torch.clamp(
            (mag_src - mag_tgt) / (mag_src + self.eps),
            min=0.0,
            max=1.0
        )

        # -------- 3. 融合：结构可对齐 & tgt 确实更弱 --------
        corr = self.alpha * cos + (1 - self.alpha) * deficit

        return corr

# 条件化跨模态注意力 空间维度# 
class ConditionalCrossModalAttention(nn.Module):
    """
    相关性引导的跨模态注意力
    """
    def __init__(self, in_channels, level="P3", reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.output_conv = nn.Conv2d(in_channels, in_channels, 1)

        # ⭐ 关键：可学习交互强度（建议初始化为 0）
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.corr_thresh = nn.Parameter(torch.tensor(0.5))

        # ⭐ 训练进度（不参与梯度）
        self.register_buffer("epoch_ratio", torch.zeros(1))
        self.level = level

        # ⭐ 日志控制
        self.log_interval = 1000    # 每 100 个 forward 记录一次
        self._step = 0
        # self.level = level

        # self.topk_ratio = topk_ratio  # 训练后期时，只在“高相关区域”交互
        # if level == "P3":
        #     self.topk_ratio = 0.05
        # elif level == "P4":
        #     self.topk_ratio = 0.1
        # else:
        #     self.topk_ratio = 0.2

    def _log_gate(self, gate):
        try:
            log_path = "/home/ciomp/project/hzx/MOD/multiYolo-old/runs/v8/m4sar/proposed-log/dcma_gate_log.txt"
            with open(log_path, "a") as f:
                # print("--log--\n")
                f.write(
                    f"{self.level}, "
                    f"DCMA "
                    f"epoch_ratio={self.epoch_ratio.item():.3f}, "
                    f"gamma={self.gamma.item():.4f}, "
                    f"gate_mean={gate.mean().item():.4f}, "
                    f"gate_max={gate.max().item():.4f}\n"
                )
        except Exception:
            pass

    def forward(self, query_feat, key_feat, value_feat, corr_map):
        """
        corr_map: (B,1,H,W)  跨模态相关性门控
        """
        B, _, H, W = query_feat.size()

        Q = self.query_conv(query_feat)
        K = self.key_conv(key_feat)
        V = self.value_conv(value_feat)

        Q = Q.view(B, -1, H * W).permute(0, 2, 1)   # (B,N,C')
        K = K.view(B, -1, H * W)                    # (B,C',N)
        V = V.view(B, -1, H * W)                    # (B,C,N)

        scale = Q.size(-1) ** 0.5
        attn = torch.bmm(Q, K) / scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(V, attn.permute(0, 2, 1))
        out = out.view(B, -1, H, W)
        out = self.output_conv(out)

        # 是否添加相关性引导
        if corr_map != None:
        # # 互补区域 (1 - corr) + 训练后期 (epoch_ratio ↑)
            # 在这里在传入时统一调整 不修改里面了
            # gate = (1 - corr_map) * self.epoch_ratio
            # 配合方向性互补相关性时
            gate = corr_map * self.epoch_ratio
            # 无epoch-aware
            # gate = corr_map
            # # ⭐ 条件化交互（核心）
            out = out * gate

        # # ✅ 只在“高相关区域”交互
        # gate = torch.sigmoid((corr_gate - self.corr_thresh) * 10.0)
        # gate = gate * self.epoch_ratio

        # # ===== Top-K Spatial Gate =====
        # corr_flat = corr_map.view(B, -1)  # (B, H*W)
        # k = max(1, int(self.topk_ratio * H * W))

        # # 每个 batch 单独算 top-k 阈值
        # topk_vals, _ = torch.topk(corr_flat, k, dim=1)
        # thresh = topk_vals[:, -1].view(B, 1, 1, 1)

        # gate = (corr_map >= thresh).to(query_feat.dtype)
        # gate = gate * self.epoch_ratio

            
        # out = torch.clamp(self.gamma, min=0) * out * gate

        # ================== 日志部分 ==================
        # if self.training:
        #     self._step += 1
        #     if self._step % self.log_interval == 0:
        #         self._log_gate(gate)
        # =================================================

        return (query_feat + self.gamma * out).contiguous()


class EnhancedDFSC(nn.Module):
    """改进的差异特征选择与补偿模块（使用跨模态注意力）"""
    def __init__(self, in_channels, level) -> None:
        super(EnhancedDFSC, self).__init__()
        
        self.corr = CrossModalCorrelation(alpha=0.9)
        # 使用方向性互补相关性
        # self.corr = DirectionalCrossModalCorrelation(alpha=0.2)
        # 条件引导的双向跨模态注意力
        self.opt2sar = ConditionalCrossModalAttention(in_channels, level=level)
        self.sar2opt = ConditionalCrossModalAttention(in_channels, level=level)
        
        # 双向跨模态注意力
        # self.optical_to_sar_attn = CrossModalAttention(in_channels)
        # self.sar_to_optical_attn = CrossModalAttention(in_channels)

        
        # # 残差块用于特征增强
        # self.res_optical = ResidualBlock(in_channels, in_channels)
        # self.res_sar = ResidualBlock(in_channels, in_channels)

        # # 层级交互强度
        # if level == "P3":
        #     self.scale = 0.3
        # elif level == "P4":
        #     self.scale = 0.6
        # else:  # P5
        #     self.scale = 1.0

    # optical_feat, sar_feat 用于计算相关性 optical_feat_nomal, sar_feat_nomal 用于后续的双向交互增强以及融合
    def forward(self, optical_feat, sar_feat, optical_feat_nomal, sar_feat_nomal):
        """
        双向跨模态注意力融合：
        1. Optical引导SAR信息提取
        2. SAR引导Optical信息提取
        """
        
        # # Optical → SAR 注意力：用光学特征引导SAR信息提取
        # # Query: optical, Key/Value: sar
        # sar_enhanced = self.optical_to_sar_attn(optical_feat, sar_feat, sar_feat)
        
        # # SAR → Optical 注意力：用SAR特征引导光学信息提取  
        # # Query: sar, Key/Value: optical
        # optical_enhanced = self.sar_to_optical_attn(sar_feat, optical_feat, optical_feat)

        # 计算跨模态相关性
        corr = self.corr(optical_feat, sar_feat)  # (B,1,H,W)

        # 计算方向互补相关性
        # # Optical → SAR：SAR 哪些地方更弱
        # corr_opt2sar = self.corr(
        #     feat_src=optical_feat,
        #     feat_tgt=sar_feat
        # )

        # corr_sar2opt = self.corr(
        #     feat_src=sar_feat,
        #     feat_tgt=optical_feat
        # )

        # Optical → SAR（互补区域更强）
        sar_dcma = self.opt2sar(
            query_feat=sar_feat_nomal,
            key_feat=optical_feat_nomal,
            value_feat=optical_feat_nomal,
            corr_map=1-corr
        )
        # sar_dcma = sar_feat_nomal

        # SAR → Optical
        opt_dcma  = self.sar2opt(
            query_feat=optical_feat_nomal,
            key_feat=sar_feat_nomal,
            value_feat=sar_feat_nomal,
            corr_map=1-corr
        )
        # opt_dcma = optical_feat_nomal

        # # 层级残差调制（防 P3 崩）
        # sar_out = sar_feat_nomal + self.scale * (sar_dcma - sar_feat_nomal)
        # opt_out = optical_feat_nomal + self.scale * (opt_dcma - optical_feat_nomal)
    
        return opt_dcma, sar_dcma


class EnhancedCompFusionModule(nn.Module):
    """改进的综合融合模块（使用跨模态注意力）"""
    def __init__(self, in_channels):
        super(EnhancedCompFusionModule, self).__init__()
        self.gwf = GWF(in_channels)
        self.dfsc = EnhancedDFSC(in_channels)  # 使用改进的DFSC
        # 拼接后通道数变成 2 * in_channels，用 1x1 conv 压缩回 in_channels
        self.fuse_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x_optical, x_sar = x
        
        # 第一步：跨模态注意力增强
        enhanced_optical, enhanced_sar = self.dfsc(x_optical, x_sar)
        
        # 第二步：门控加权融合
        fused = self.gwf([enhanced_optical, enhanced_sar])
        
        return fused


class P3_Cross_Gate(nn.Module):
    """改进的综合融合模块（使用跨模态注意力）"""
    def __init__(self, in_channels):
        super(P3_Cross_Gate, self).__init__()
        self.in_channels = in_channels
        self.gwf = GWF(in_channels)
        # self.degwf = DecoupledGWF(in_channels)
        self.dfsc = EnhancedDFSC(in_channels, "P3")  # 使用改进的DFSC
    
        # 特征分布对齐（动态适配特征尺寸）
        self.sar_norm = nn.LayerNorm(in_channels)
        self.optical_norm = nn.LayerNorm(in_channels)


    def forward(self, x):
        x_optical, x_sar = x
        
        # 动态初始化归一化层（适配特征尺寸），确保光 SAR 特征分布一致
        x_sar_norm = self.sar_norm(x_sar.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_optical_norm = self.optical_norm(x_optical.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # # ✅ SAR结构增强（在norm之后）
        # x_sar_denoise = self.sar_enhance(x_sar_norm)

        # # ----- 双向交互注意力增强 + 相关性门控 ----- #
        # # 第一步：跨模态注意力增强
        enhanced_optical, enhanced_sar = self.dfsc(x_optical, x_sar, x_optical_norm, x_sar_norm)

        
        # # 第二步：门控加权融合
        fused = self.gwf([enhanced_optical, enhanced_sar])

        # 消融实验1：无门控直接相加 # 
        # enhanced_optical, enhanced_sar = self.dfsc(x_optical_norm, x_sar_norm)
        # fused = enhanced_optical + enhanced_sar

        # ----- 消融实验2： 双向交互注意力增强 + 通道拼接 ----- #
        # enhanced_optical, enhanced_sar = self.dfsc(x_optical, x_sar, x_optical_norm, x_sar_norm)
        # fused = torch.cat([enhanced_optical, enhanced_sar], dim=1)
        # fused = self.channel_compress(fused)
        
        # ----- 消融实验3： 相关性门控----- #
        # fused = self.gwf([enhanced_optical, enhanced_sar])

        return fused


class P4_Cross_Gate(nn.Module):
    """改进的综合融合模块（使用跨模态注意力）"""
    def __init__(self, in_channels):
        super(P4_Cross_Gate, self).__init__()
        self.in_channels = in_channels
        self.gwf = GWF(in_channels)

        self.dfsc = EnhancedDFSC(in_channels, "P4")  # 使用改进的DFSC
        # 特征分布对齐（动态适配特征尺寸）
        self.sar_norm = nn.LayerNorm(in_channels)
        self.optical_norm = nn.LayerNorm(in_channels)


    def forward(self, x):
        x_optical, x_sar = x

        # x_sar_clean = self.sar_deno_enhance(x_sar)
        
        # 动态初始化归一化层（适配特征尺寸），确保光 SAR 特征分布一致
        x_sar_norm = self.sar_norm(x_sar.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_optical_norm = self.optical_norm(x_optical.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # # # ----- 双向交互注意力增强 + 相关性门控 ----- #
        # # 第一步：跨模态注意力增强
        enhanced_optical, enhanced_sar = self.dfsc(x_optical, x_sar, x_optical_norm, x_sar_norm)

        
        # 第二步：门控加权融合
        fused = self.gwf([enhanced_optical, enhanced_sar])

        # ----- 消融实验1： 双向交互注意力增强 + 直接相加 ----- #
        # enhanced_optical, enhanced_sar = self.dfsc(x_optical_norm, x_sar_norm)
        # fused = enhanced_optical + enhanced_sar

        # # ----- 消融实验2： 双向交互注意力增强 + 通道拼接 ----- #
        # enhanced_optical, enhanced_sar = self.dfsc(x_optical_norm, x_sar_norm)
        # fused = torch.cat([enhanced_optical, enhanced_sar], dim=1)
        # fused = self.channel_compress(fused)

        # # ----- 消融实验3： 相关性门控----- #
        # fused = self.gwf([x_optical_norm, x_sar_norm])

        return fused
    
class P5_Cross_Gate(nn.Module):
    """改进的综合融合模块（使用跨模态注意力）"""
    def __init__(self, in_channels):
        super(P5_Cross_Gate, self).__init__()
        self.in_channels = in_channels
        self.gwf = GWF(in_channels)
        self.dfsc = EnhancedDFSC(in_channels, level= "P5")  # 使用改进的DFSC
        
        # # 特征分布对齐（动态适配特征尺寸）
        self.sar_norm = nn.LayerNorm(in_channels)
        self.optical_norm = nn.LayerNorm(in_channels)


    def forward(self, x):
        x_optical, x_sar = x

        # x_sar_clean = self.sar_deno_enhance(x_sar)
        
        # 动态初始化归一化层（适配特征尺寸），确保光 SAR 特征分布一致
        x_sar_norm = self.sar_norm(x_sar.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_optical_norm = self.optical_norm(x_optical.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # # ----- 双向交互注意力增强 + 相关性门控 ----- #
        # # 第一步：跨模态注意力增强
        enhanced_optical, enhanced_sar = self.dfsc(x_optical, x_sar, x_optical_norm, x_sar_norm)

        
        # # 第二步：门控加权融合
        fused = self.gwf([enhanced_optical, enhanced_sar])

        # # ----- 消融实验1： 双向交互注意力增强 + 直接相加 ----- #
        # # enhanced_optical, enhanced_sar = self.dfsc(x_optical_norm, x_sar_norm)
        # fused = enhanced_optical + enhanced_sar

        # # # ----- 消融实验2： 双向交互注意力增强 + 通道拼接 ----- #
        # # # enhanced_optical, enhanced_sar = self.dfsc(x_optical_norm, x_sar_norm)
        # fused = torch.cat([enhanced_optical, enhanced_sar], dim=1)
        # fused = self.channel_compress(fused)

        # ----- 消融实验3： 相关性门控----- #
        # fused = self.gwf([x_optical_norm, x_sar_norm])

        return fused



class Add(nn.Module):
    def __init__(self, in_channels):
        super(Add, self).__init__()

    def forward(self, x):
        x_opt, x_sar = x
        x_fused = x_opt + x_sar

        return x_fused


# ---------------------------------------------------------------------------- #
#                                   ICAFusion                                  #
# ---------------------------------------------------------------------------- #
class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [rgb_fea_flat, ir_fea_flat]



class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#
        
        return new_fea
# ---------------------------------------------------------------------------- #
#                                      end                                     #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                 single-cross                                 #
# ---------------------------------------------------------------------------- #
class SingleCross(nn.Module):
    """改进的差异特征选择与补偿模块（使用跨模态注意力）"""
    def __init__(self, in_channels) -> None:
        super(SingleCross, self).__init__()
        
        # 单向跨模态注意力
        self.optical_to_sar_attn = CrossModalAttention(in_channels)

        # 门控融合
        self.gate = GWF(in_channels)
        # self.sar_to_optical_attn = CrossModalAttention(in_channels)
        
        # # 残差块用于特征增强
        # self.res_optical = ResidualBlock(in_channels, in_channels)
        # self.res_sar = ResidualBlock(in_channels, in_channels)

    def forward(self, x):
        """
        单向跨模态注意力融合：
        1. Optical引导SAR信息提取
        
        """
        optical_feat, sar_feat = x
        # Optical → SAR 注意力：用光学特征引导SAR信息提取
        # Query: optical, Key/Value: sar
        sar_enhanced = self.optical_to_sar_attn(optical_feat, sar_feat, sar_feat)
        
        feature_fused = self.gate([optical_feat, sar_enhanced])
    
        return feature_fused


## -------- 对比实验 -------- ##
class MSM_GatedFusion(nn.Module):
    """
    基于模态平衡遮蔽（MSM）的光SAR特征融合模块
    设计灵感：Information Fusion 126 (2026) 103576 
    """
    def __init__(self, channels, mask_ratio=0.5):
        super(MSM_GatedFusion, self).__init__()
        self.mask_ratio = mask_ratio
        
        # 通道注意力：用于识别光学特征中的“强特征”通道 
        self.channel_attn_opt = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 融合卷积层 
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x_opt, x_sar = x
        # 1. 计算强模态（光学）的通道注意力权重 
        attn_opt = self.channel_attn_opt(x_opt)
        
        # 2. MSM 策略：遮蔽最具判别性的光学通道 
        # 对权重进行排序，找到前 mask_ratio 比例的高权重索引
        batch, c, _, _ = x_opt.shape
        if self.training: # 仅在训练阶段应用遮蔽以增强弱模态学习 
            # 简化版实现：对每个Batch取平均权重
            avg_attn = attn_opt.mean(dim=(0, 2, 3)) 
            _, indices = torch.sort(avg_attn, descending=True)
            mask_num = int(c * self.mask_ratio)
            mask_idx = indices[:mask_num]
            
            # 执行遮蔽：将强特征通道置零 
            x_opt_masked = x_opt.clone()
            x_opt_masked[:, mask_idx, :, :] = 0.0
        else:
            x_opt_masked = x_opt

        # 3. 按照论文公式执行融合:
        # F_fused = Conv(F_r ⊗ mask(f_r) ⊕ F_d ⊗ F_d)
        # 这里我们将 SAR 特征与经过遮蔽的光学特征拼接
        feat_combined = torch.cat([x_opt_masked, x_sar], dim=1)
        fused_feat = self.fusion_conv(feat_combined)
        
        return fused_feat

## --------- 对比实验 CIM 模块 --------- ##
class CBG(nn.Module):
    """Conv + BN + GeLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() #

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LayerNorm2d(nn.Module):
    """针对 4D 特征图的 LayerNorm 实现"""
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)

class CIM(nn.Module):
    """
    Cross-modal Integration Module (CIM)
    复现自 Pattern Recognition 2026: LESOD
    """
    def __init__(self, channels):
        super(CIM, self).__init__()
        
        # 初始深度特征对齐
        self.align_depth = nn.Conv2d(channels, channels, kernel_size=1) 
        
        # --- Branch 1 逻辑组件 ---
        # branch1 = CBG(Cat(ri, Conv11(di)))
        self.cbg = CBG(channels * 2, channels, kernel_size=3, padding=1)
        # branch11 = Linear(branch1)
        self.linear = nn.Conv2d(channels, channels, kernel_size=1) 
        # branch12 = LN(Conv11(branch1))
        self.conv1x1_b1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.ln_b1 = LayerNorm2d(channels) 
        
        # --- Branch 2 逻辑组件 ---
        # branch2 = Conv11(LN(D(ri ⊗ Conv11(di))) + ri)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels) 
        self.ln_b2 = LayerNorm2d(channels)
        self.conv1x1_b2 = nn.Conv2d(channels, channels, kernel_size=1) 
        
        # --- 最终融合组件 ---
        # fusei = Conv11(Cat(branch11 ⊗ branch12, branch2))
        self.fuse_conv = nn.Conv2d(channels * 2, channels, kernel_size=1) 

    def forward(self, x):
        r_i, d_i = x
        # 深度特征维度对齐
        d_aligned = self.align_depth(d_i)
        
        # 1. Branch 1 逻辑
        b1_feat = self.cbg(torch.cat([r_i, d_aligned], dim=1)) 
        branch11 = self.linear(b1_feat) 
        branch12 = self.ln_b1(self.conv1x1_b1(b1_feat)) 
        b1_output = branch11 * branch12 # 元素级乘法 
        
        # 2. Branch 2 逻辑
        b2_mult = r_i * d_aligned # 元素级乘法 (ri ⊗ di)
        b2_output = self.dw_conv(b2_mult) # 深度可分离卷积 D(.)
        b2_output = self.ln_b2(b2_output) # 层归一化
        branch2 = self.conv1x1_b2(b2_output + r_i) # 残差连接 + 1x1 卷积
        
        # 3. 最终融合 
        fuse_i = self.fuse_conv(torch.cat([b1_output, branch2], dim=1)) 
        
        return fuse_i.contiguous()

## --------------------- 对比实验3：DeNet ----------------- ##
# --- 基础组件 ---
class CBP(nn.Module):
    """Conv1x1 + BN + PReLU"""
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CBR(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# --- 核心融合模块 ---
class DAM(nn.Module):
    """
    Depth Adapter Module (DAM) - 深度适配器模块 (适配光-SAR)
    复现自 DENet (Pattern Recognition 2026)
    输入顺序: x = [x_opt, x_sar]
    """
    def __init__(self, c1):
        super(DAM, self).__init__()
        # 通道注意力 (CA): 对应公式 (1)(2) 
        self.ca = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.Sigmoid()
        )
        # CBP 子网络: Conv1x1 + BN + PReLU, 对应公式 (3) 
        self.cbp = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.PReLU()
        )
        
        # --- 分支 1: 修正流 (Correction Stream) ---
        # 对应公式 (4), 融合光学与 SAR 
        self.conv_b1_init = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.Conv2d(c1, c1, 3, padding=1)
        )
        # 对应公式 (7), 投影输出 
        self.conv_b1_fuse = nn.Conv2d(c1 * 2, c1, 1)
        
        # --- 分支 2: 选择流 (Selection Stream) ---
        # 对应公式 (8), 捕捉显著细节区域 
        self.spatial_attn = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), # 
            nn.Conv2d(c1, c1, kernel_size=7, padding=3),      # 
            nn.Conv2d(c1, c1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 光学特征在前，SAR 特征在后
        x_opt, x_sar = x 
        
        # 2. 对 SAR 特征进行通道去噪与增强 
        x_sar_c = self.ca(x_sar) * x_sar
        x_sar_p = self.cbp(x_sar_c)
        
        # --- 分支 1 逻辑: 跨模态残差修正 ---
        # 拼接修正后的 SAR 和 原始光学特征 
        x_sar_opt = self.conv_b1_init(torch.cat([x_sar_p, x_opt], dim=1))
        # 元素级乘法与加法 
        x_dr1 = x_sar_opt * x_opt 
        x_dr2 = x_sar_opt + x_opt
        # 降维得到分支 1 输出 
        x_d1 = self.conv_b1_fuse(torch.cat([x_dr1, x_dr2], dim=1))
        
        # --- 分支 2 逻辑: 空间显著性选择 ---
        # 利用修正后的 SAR 自身生成空间权重 
        spatial_weights = self.spatial_attn(x_sar_p)
        x_d2 = spatial_weights * x_sar_p
        
        # 最终输出：两个分支相加 
        return (x_d1 + x_d2).contiguous()

class ERGM(nn.Module):
    """
    Edge Reinforcement Guidance Module (ERGM)
    输入顺序: x = [f1, f2] (不同尺度的融合特征)
    """
    def __init__(self, c1, c2):
        super(ERGM, self).__init__()
        # 对应公式 (10) 
        self.cbr_f2 = nn.Sequential(
            nn.Conv2d(c2, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        # 多分支提取边缘, 对应公式 (12-14) 
        self.cbr1 = nn.Sequential(nn.Conv2d(c1 * 2, c1, 1), nn.BatchNorm2d(c1), nn.ReLU())
        self.cbr3 = nn.Sequential(nn.Conv2d(c1 * 2, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU())
        self.cbr5 = nn.Sequential(nn.Conv2d(c1 * 2, c1, 5, padding=2), nn.BatchNorm2d(c1), nn.ReLU())

    def forward(self, x):
        f1, f2 = x # 解包输入的两个特征层
        
        # 对 F2 进行双线性上采样和增强 
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f2_p = F.max_pool2d(self.cbr_f2(f2_up), 3, 1, 1)
        
        # 拼接后通过多分支卷积提取最终边缘引导特征 F_eg 
        f_lf = torch.cat([f1, f2_p], dim=1)
        f_eg = self.cbr1(f_lf) + self.cbr3(f_lf) + self.cbr5(f_lf)
        
        return f_eg.contiguous()

class DENetRefine(nn.Module):
    """
    利用边缘特征 F_eg 精炼高级特征
    输入顺序: x = [f_high, f_eg]
    """
    def __init__(self, c1, c2):
        super(DENetRefine, self).__init__()
        self.proj = nn.Conv2d(c2, c1, 1)

    def forward(self, x):
        f_high, f_eg = x # 解包
        
        # 对齐边缘特征尺寸并执行精炼逻辑 
        f_eg_up = F.interpolate(f_eg, size=f_high.shape[2:], mode='bilinear', align_corners=True)
        f_eg_p = self.proj(f_eg_up)
        # 边缘引导增强 
        return (f_high * f_eg_p + f_high).contiguous()

## ------------- 对比实验4：CATNet ----------------- ##
# --- 基础注意力组件 ---
class SpatialAttention_CAT(nn.Module):
    """空间注意力: SA(x) = Sigmoid(Conv3(CGMP(x))) """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x_max, _ = torch.max(x, dim=1, keepdim=True) # CGMP: 通道向最大池化 
        return torch.sigmoid(self.conv(x_max))

class ChannelAttention_CAT(nn.Module):
    """通道注意力: CA(x) = Sigmoid(Conv1(GMP(x))) """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_pool = F.adaptive_max_pool2d(x, 1) # GMP: 全局最大池化 
        return torch.sigmoid(self.conv(x_pool))

# --- CATNet 核心特征融合模块 ---
class CMFM(nn.Module):
    """
    跨模态融合模块 (CMFM) 
    输入: x = [f_opt, f_sar]
    """
    def __init__(self, c1):
        super().__init__()
        # DFEM: 增强深度特征 
        self.sa_d = SpatialAttention_CAT()
        self.ca_d = ChannelAttention_CAT(c1)
        # RFEM: 增强RGB特征 
        self.sa_r = SpatialAttention_CAT()
        self.ca_r = ChannelAttention_CAT(c1)
        # FB: 融合块 
        self.fb_conv = nn.Conv2d(c1 * 2, c1, 1)

    def forward(self, x):
        f_r, f_d = x # 解包: 光学在前, SAR在后
        
        # DFEM 逻辑: 利用 RGB 语义增强深度特征 
        f_d_e = f_d * self.ca_d(f_d + (f_d * self.sa_d(f_d * f_r)))
        
        # RFEM 逻辑: 利用深度空间信息增强 RGB 特征 
        f_r_e = f_r * self.ca_r(f_r + (f_r * self.sa_r(f_r * f_d)))
        
        # Fusion Block (FB) 
        return self.fb_conv(torch.cat([f_r_e, f_d_e], dim=1)).contiguous()

class MSAM(nn.Module):
    """
    Multi-Scale Aggregation Module (MSAM)
    修正版：支持通道数调整
    """
    def __init__(self, c1, c2):
        super(MSAM, self).__init__()
        # c1: 低级特征通道, c2: 高级特征通道
        # BConv3 必须将高级特征 c2 调整为 c1，以便后续与 f_l (c1) 进行元素级乘法 
        self.bconv = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.ca_g = ChannelAttention(c1) # 门控信号在融合后的 c1 通道上生成 

    def forward(self, x):
        f_l, f_h = x # f_l (低级, c1), f_h (高级, c2)
        
        # 1. 高级特征上采样并调整通道数 
        f_h_up = F.interpolate(f_h, size=f_l.shape[2:], mode='bilinear', align_corners=True)
        f_h_proc = self.bconv(f_h_up) # 现在 f_h_proc 通道数为 c1
        
        # 2. 初始特征混合 
        f_prime = f_l * f_h_proc
        
        # 3. 生成通道级门控信号 CA_G 
        gate = self.ca_g(f_prime)
        
        # 4. 选择性聚合 
        # 注意：公式中的 f_h 应该使用调整通道后的 f_h_proc
        f_u_end = f_l * gate + (1 - gate) * f_h_proc
        return f_u_end.contiguous()

## -------------------- 对比实验4 -----------------------##
class CAM(nn.Module):
    """跨模态注意力融合模块 (CAM) - 适配不同通道数"""
    def __init__(self, c1, c2_hi):
        super().__init__()
        # c1: 当前层通道, c2_hi: 高一层通道
        self.conv_align_r = nn.Sequential(
            nn.Conv2d(c1 + c2_hi, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.conv_align_d = nn.Sequential(
            nn.Conv2d(c1 + c2_hi, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        
        self.ca = CoordinateAttention(c1 * 2) 
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.conv_out = nn.Sequential(nn.Conv2d(c1 * 2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(inplace=True))

    def forward(self, x):
        f_i_r, f_i_d, f_hi_r, f_hi_d = x # 从 list 解包
        # 按照 CPNet 公式 (1)-(5) 执行融合逻辑...
        f_hi_r_up = F.interpolate(f_hi_r, size=f_i_r.shape[2:], mode='bilinear', align_corners=True)
        F_i_r = self.conv_align_r(torch.cat([f_hi_r_up, f_i_r], dim=1))
        f_hi_d_up = F.interpolate(f_hi_d, size=f_i_d.shape[2:], mode='bilinear', align_corners=True)
        F_i_d = self.conv_align_d(torch.cat([f_hi_d_up, f_i_d], dim=1))
        
        F_i = torch.cat([F_i_r, F_i_d], dim=1)
        F_i_ca = self.ca(F_i)
        avg_out = torch.mean(F_i_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(F_i_ca, dim=1, keepdim=True)
        sa_map = self.sa(torch.cat([avg_out, max_out], dim=1))
        return self.conv_out(F_i * sa_map)

class CoordinateAttention(nn.Module):
    """坐标注意力实现"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Sigmoid() 
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))
    
class RCM(nn.Module):
    """
    Residual Convolutional Module (RCM) - 精炼特征并过滤噪声 
    """
    def __init__(self, c1):
        super(RCM, self).__init__()
        # 深度可分离卷积 7x7 
        self.dw = nn.Conv2d(c1, c1, kernel_size=7, padding=3, groups=c1, bias=False)
        self.pw1 = nn.Conv2d(c1, c1 * 4, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(c1 * 4, c1, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        if isinstance(x, list): x = x[0]
        identity = x
        
        # 对应论文公式 (7): P = f + PW2(GELU(PW1(LN(DW(f))))) 
        out = self.dw(x)
        # 使用简单的均值归一化模拟 LayerNorm，适配 YOLO 的动态 H/W
        out = out - out.mean(dim=(2, 3), keepdim=True) 
        
        out = self.pw1(out)
        out = self.act(out)
        out = self.pw2(out)
        return (identity + out).contiguous()