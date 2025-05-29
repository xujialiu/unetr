import math
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from utils import trunc_normal_
from collections import namedtuple
from packaging import version


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


Config = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


class FlashAttention(nn.Module):
    # code from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_flash_attn_vit.py
    def __init__(self):
        super().__init__()

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def forward(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = nn.functional.scaled_dot_product_attention(q, k, v)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_flash = use_flash  # whether to use the flash attention
        if self.use_flash and (
            version.parse(torch.__version__) < version.parse("2.0.0")
        ):
            print(
                f"in order to use flash attention, you must be using pytorch 2.0 or above, but current version is: {version.parse(torch.__version__)}"
            )
            print(f"will disable the flash attention")
            self.use_flash = False

        if self.use_flash:
            self.flash_attn = FlashAttention()
            print(f"will use the Flash Attention.")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_flash:
            out = self.flash_attn(q, k, v)
            attn = None
            x = rearrange(out, "b h n d -> b n (h d)")
        else:

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_flash=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash=use_flash,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer
    add param: use_flash: whether to use the flash attention, which requires the torch version greater than 2.0.0
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=False,
        use_flash=False,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_flash=use_flash,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        # Classifier head
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # pos_embed的实现, 如果大小和输入的特征图大小一致, 则直接返回, 否则进行resize
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # return [2*B, 197, 384], two global crops
        x = x.mean(dim=1)
        return x

    def forward_cls(self, x):
        x = self.prepare_tokens(x)
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward_features_with_intermediate(self, x):
        # we return the output tokens from the `n` last blocks
        list_intermediate_output = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            list_intermediate_output.append(self.norm(x))

        x = self.norm(x)
        x = x.mean(dim=1)
        return x, list_intermediate_output

    def forward(self, x):
        x = self.prepare_tokens(x)
        x, list_intermediate_output = self.forward_features_with_intermediate(x)
        x = self.head(x)
        # torch.Size([1, 197, 768])
        return x, list_intermediate_output


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    import os

    # 设置随机种子
    torch.manual_seed(42)

    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    print("Creating ViT-Base model...")
    model = vit_large(
        img_size=224,
        patch_size=16,
        num_classes=1000,  # ImageNet classes
        use_flash=False,
    )

    # 将模型移到设备上
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 创建随机输入
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    # 测试前向传播
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        # 测试只返回最终输出
        output = model.forward_cls(dummy_input)
        print(f"Output shape (classification): {output.shape}")

        # 测试返回中间层输出
        output, intermediates = model(dummy_input)
        print(f"Output shape with intermediates: {output.shape}")
        print(f"Number of intermediate outputs: {len(intermediates)}")
        if intermediates:
            print(f"First intermediate output shape: {intermediates[0].shape}")

    # 加载checkpoint（如果存在）
    checkpoint_path = "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth"
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 处理不同的checkpoint格式
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 加载权重
        # msg = model.load_state_dict(state_dict, strict=False)
        # print(msg)

        print("Checkpoint loaded successfully!")

        # 如果checkpoint包含其他信息
        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if "best_acc" in checkpoint:
            print(f"Best accuracy: {checkpoint['best_acc']}")
    else:
        print(f"\nNo checkpoint found at {checkpoint_path}")
        print("You can save a checkpoint using:")
        print(
            "torch.save({'model': model.state_dict(), 'epoch': epoch}, checkpoint_path)"
        )
    
    from peft import LoraConfig, get_peft_model
    config_lora = LoraConfig(
            r=4,  # LoRA的秩
            lora_alpha=8,  # LoRA的alpha参数, scaling=alpha/r
            target_modules=["qkv"],  # 需要应用LoRA的模块名称
            # target_modules="all-linear",
            lora_dropout=0.1,
            bias=None,
            task_type="FEATURE_EXTRACTION",
            # task_type="CAUSAL_LM"
        )
    get_peft_model(model, config_lora)

    # 测试不同输入尺寸（测试位置编码插值）
    print("\nTesting different input sizes...")
    for size in [224, 384]:
        test_input = torch.randn(1, 3, size, size).to(device)
        with torch.no_grad():
            output = model.forward(test_input)
            print(f"Input size {size}x{size} -> Output shape: {output[0].shape}")

    # 保存示例checkpoint
    print("\nSaving example checkpoint...")
    example_checkpoint = {
        "model": model.state_dict(),
        "epoch": 0,
        "config": {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "num_classes": 1000,
        },
    }
    torch.save(example_checkpoint, "vit_base_example.pth")
    print("Example checkpoint saved as 'vit_base_example.pth'")
