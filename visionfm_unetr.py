import torch
import torch.nn as nn
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x):
        return self.deconv(x)


class UnetrHead(nn.Module):
    # default for vit-base model
    def __init__(
        self,
        embed_dim,
        num_classes,
        img_dim=224,
        patch_dim=16,
        dropout_rate=0.3,
    ):
        super(UnetrHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.dropout_rate = dropout_rate
        self.patch_dim = patch_dim
        self.name = "unetr"

        self.base_channel = 32
        print(f"base channel: {self.base_channel}")

        # add at 09/15
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate)
        else:
            self.dropout = None

        """ CNN Decoder """
        ## Decoder 1
        self.d1 = DeconvBlock(self.embed_dim, self.base_channel * 4)
        self.s1 = nn.Sequential(
            DeconvBlock(self.embed_dim, self.base_channel * 4),
            ConvBlock(self.base_channel * 4, self.base_channel * 4),
        )
        self.c1 = nn.Sequential(
            ConvBlock(self.base_channel * 8, self.base_channel * 4),
            ConvBlock(self.base_channel * 4, self.base_channel * 4),
        )

        ## Decoder 2
        self.d2 = DeconvBlock(self.base_channel * 4, self.base_channel * 2)
        self.s2 = nn.Sequential(
            DeconvBlock(self.embed_dim, self.base_channel * 2),
            ConvBlock(self.base_channel * 2, self.base_channel * 2),
            DeconvBlock(self.base_channel * 2, self.base_channel * 2),
            ConvBlock(self.base_channel * 2, self.base_channel * 2),
        )
        self.c2 = nn.Sequential(
            ConvBlock(self.base_channel * 4, self.base_channel * 2),
            ConvBlock(self.base_channel * 2, self.base_channel * 2),
        )

        ## Decoder 3
        self.d3 = DeconvBlock(self.base_channel * 2, self.base_channel)
        self.s3 = nn.Sequential(
            DeconvBlock(self.embed_dim, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel),
            DeconvBlock(self.base_channel, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel),
            DeconvBlock(self.base_channel, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel),
        )
        self.c3 = nn.Sequential(
            ConvBlock(self.base_channel * 2, self.base_channel),
            ConvBlock(self.base_channel, self.base_channel),
        )

        ## Decoder 4
        self.d4 = DeconvBlock(self.base_channel, self.base_channel // 2)
        self.s4 = nn.Sequential(
            ConvBlock(3, self.base_channel // 2),
            ConvBlock(self.base_channel // 2, self.base_channel // 2),
        )
        self.c4 = nn.Sequential(
            ConvBlock(self.base_channel, self.base_channel // 2),
            ConvBlock(self.base_channel // 2, self.base_channel // 2),
        )

        """ Output """
        self.cls = nn.Conv2d(
            self.base_channel // 2, self.num_classes, kernel_size=1, padding=0
        )
        self.apply(self._init_weights)

    def _reshape_output(self, x):
        # [1, 197, 1024], [1, 197, 768], [B, seq_len, embed_dim] -> [B, C, 14, 14]
        h = w = int(x.shape[1] ** 0.5)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x

    def forward(self, feats: list, inputs):
        z3, z6, z9, z12 = feats

        ## Reshapings
        z0 = inputs  # [B, C, H, W]
        z3, z6, z9, z12 = (
            self._reshape_output(z3),
            self._reshape_output(z6),
            self._reshape_output(z9),
            self._reshape_output(z12),
        )

        ## Decoder 1
        x = self.d1(z12)  # x2, deconv
        s = self.s1(z9)  # deconv + conv
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)  # conv

        ## Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        ## Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        ## Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        """ Output """
        if self.dropout is not None:
            x = self.dropout(x)  # add dropout
        output = self.cls(x)  # [B, C, H, W]

        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # 初始化模型
    model = UnetrHead(
        embed_dim=768,  # 对应ViT-base的embed维度
        num_classes=2,  # 二分类任务
        img_dim=224,  # 输入图像尺寸
        patch_dim=16,  # patch大小
        dropout_rate=0.3,  # 使用dropout
    )

    # 生成模拟输入数据
    batch_size = 1
    embed_dim = 768

    # 生成各阶段特征 (假设ViT输出已移除cls token)
    feats = [
        torch.randn(batch_size, 196, embed_dim),  # z3
        torch.randn(batch_size, 196, embed_dim),  # z6
        torch.randn(batch_size, 196, embed_dim),  # z9
        torch.randn(batch_size, 196, embed_dim),  # z12
    ]

    # 生成原始输入图像
    inputs = torch.randn(batch_size, 3, 224, 224)  # 随机图像

    # 前向传播
    output = model(feats, inputs)

    # 验证输出维度
    print("Output shape:", output.shape)  # 预期输出 [1, 2, 224, 224]

    # 验证参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 测试梯度反向传播
    output.mean().backward()
    print("Gradient check passed.")
