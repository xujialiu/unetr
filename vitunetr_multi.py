import torch
import torch.nn as nn
from functools import partial
from typing import Optional, List, Tuple

from visionfm_model import VisionTransformer
from visionfm_unetr import UnetrHead

# Import the necessary components from your provided code
# In practice, these would be imported from their respective modules
# from vision_transformer import VisionTransformer, vit_base, vit_large
# from unetr_head import UnetrHead


class ClsHead(nn.Module):
    """
    Wu, Jianfang, et al. "Vision Transformer‐based recognition of diabetic retinopathy grade." Medical Physics 48.12 (2021): 7850-7863.
    """

    def __init__(self, embed_dim, num_classes, layers=3):
        super(ClsHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = (
            layers  # default is 3 layers, we test different layers for retfound
        )

        if self.layers == 3:
            channels = [
                self.embed_dim,
                self.embed_dim // 2,
                self.embed_dim // 4,
                self.num_classes,
            ]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[2], channels[3]),
            )
        elif self.layers == 2:
            channels = [self.embed_dim, self.embed_dim // 4, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]),
            )
        elif self.layers == 1:
            channels = [self.embed_dim, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
            )
        self.channel_bn = nn.BatchNorm2d(
            self.embed_dim,
            eps=1e-6,  # default 1e-6
            momentum=0.99,  # default: 0.99
        )
        self.init_weights()

    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        # flatten
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        # linear layer
        return self.classifier(x)


class ViTUNETR(nn.Module):
    """
    Vision Transformer with UNETR decoder head for segmentation tasks.

    This model combines a ViT encoder backbone with a UNETR decoder head,
    extracting features from multiple layers of the ViT and using them
    for hierarchical feature decoding in the UNETR head.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes_seg: int = 2,
        num_classes_cls: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        decoder_dropout: float = 0.3,
        use_flash: bool = False,
        # Which ViT layers to extract features from
        feature_layers: List[int] = [3, 6, 9, 12],
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size for ViT
            in_chans: Number of input channels
            num_classes: Number of segmentation classes
            embed_dim: Embedding dimension for ViT
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio for ViT blocks
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            decoder_dropout: Dropout rate for UNETR decoder
            use_flash: Whether to use flash attention
            feature_layers: List of layer indices to extract features from (1-indexed)
            pretrained_vit: Path to pretrained ViT weights
        """
        super().__init__()

        self.num_classes = num_classes_seg
        self.feature_layers = feature_layers

        # Ensure feature_layers are sorted and valid
        assert all(1 <= layer <= depth for layer in feature_layers), (
            f"feature_layers must be between 1 and {depth}"
        )
        assert len(feature_layers) == 4, "UNETR head expects exactly 4 feature maps"

        # Create ViT encoder
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # No classification head
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_flash=use_flash,
        )

        self.decoder_heads = nn.ModuleList(
            [
                UnetrHead(
                    embed_dim=embed_dim,
                    num_classes=1,  # Each head performs binary segmentation
                    img_dim=img_size,
                    patch_dim=patch_size,
                    dropout_rate=decoder_dropout,
                )
                for _ in range(num_classes_seg)
            ]
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.head = ClsHead(embed_dim=embed_dim, num_classes=num_classes_cls, layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT encoder and UNETR decoder.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            classification output [B, num_classes]
            Segmentation output [B, num_classes, H, W]
        """
        # Extract hierarchical features from ViT
        # features = self.extract_features(x)
        cls_token, list_intermediate_output = self.encoder(x)

        list_features = [
            list_intermediate_output[idx - 1][:, 1:] for idx in self.feature_layers
        ]

        # Process each class with its own head
        seg_outputs = []
        for decoder_head in self.decoder_heads:
            class_seg = decoder_head(list_features, x)  # Shape: [B, 1, H, W]
            seg_outputs.append(class_seg)

        # Combine outputs from all heads
        seg = torch.cat(seg_outputs, dim=1)  # Shape: [B, num_classes_seg, H, W]

        cls = self.head(cls_token)

        return cls, seg

    def forward_cls(self, x):
        cls_token, list_intermediate_output = self.encoder(x)
        cls = self.head(cls_token)
        return cls


def vit_unetr_base(
    num_classes_seg: int = 2, num_classes_cls: int = 2, **kwargs
) -> ViTUNETR:
    """Create ViT-Base UNETR model."""
    model = ViTUNETR(
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes_seg=num_classes_seg,
        num_classes_cls=num_classes_cls,
        **kwargs,
    )
    return model


def vit_unetr_large(
    num_classes_seg: int = 2, num_classes_cls: int = 2, **kwargs
) -> ViTUNETR:
    """Create ViT-Large UNETR model."""
    model = ViTUNETR(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes_seg=num_classes_seg,
        num_classes_cls=num_classes_cls,
        feature_layers=[6, 12, 18, 24],  # Adjust for larger model
        **kwargs,
    )
    return model


if __name__ == "__main__":
    # Test the combined model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = vit_unetr_base(
        num_classes_seg=2,  # Binary segmentation
        num_classes_cls=2,  # Binary classification
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output[0].shape, output[1].shape}")
