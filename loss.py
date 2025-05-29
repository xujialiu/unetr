from monai.losses.dice import DiceLoss, DiceFocalLoss, DiceCELoss, GeneralizedDiceLoss
import torch
from torch import nn


class MultiLabelSegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 对每个标签使用独立的二分类损失
        self.dice_loss = DiceFocalLoss(sigmoid=True, reduction="none")

    def forward(self, output, target):
        """
        output: [B, C, H, W] - C个独立的二分类预测
        target: [B, C, H, W] - 每个通道表示一个标签的存在与否
        """
        # 计算每个标签的损失
        losses = []
        for i in range(output.shape[1]):
            loss = self.dice_loss(output[:, i : i + 1], target[:, i : i + 1])
            losses.append(loss)

        # 返回平均损失
        return torch.stack(losses).mean()


if __name__ == "__main__":
    print("Testing MultiLabelSegmentationLoss class...")
    print("=" * 50)

    # 初始化损失函数
    try:
        loss_fn = MultiLabelSegmentationLoss()
        print("✓ Loss function initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing loss function: {e}")
        exit(1)

    # 测试1: 基本功能测试
    print("\nTest 1: Basic functionality")
    print("-" * 30)
    try:
        # 创建模拟数据
        batch_size, num_classes, height, width = 2, 3, 32, 32

        # 模拟网络输出 (logits, 未经过sigmoid)
        output = torch.randn(batch_size, num_classes, height, width)

        # 模拟ground truth (二值化标签)
        target = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()

        print(f"Output shape: {output.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Target unique values: {torch.unique(target)}")

        # 计算损失
        loss = loss_fn(output, target)
        print(f"Loss value: {loss.item():.6f}")
        print(f"Loss shape: {loss.shape}")
        print("✓ Basic forward pass successful")

    except Exception as e:
        print(f"✗ Error in basic test: {e}")

    # 测试2: 梯度反向传播测试
    print("\nTest 2: Gradient computation")
    print("-" * 30)
    try:
        # 创建需要梯度的输出
        output_grad = torch.randn(
            batch_size, num_classes, height, width, requires_grad=True
        )
        target_grad = torch.randint(
            0, 2, (batch_size, num_classes, height, width)
        ).float()

        loss = loss_fn(output_grad, target_grad)
        loss.backward()

        print(f"Output gradient shape: {output_grad.grad.shape}")
        print(f"Gradient norm: {output_grad.grad.norm().item():.6f}")
        print("✓ Gradient computation successful")

    except Exception as e:
        print(f"✗ Error in gradient test: {e}")

    # 测试3: 边界情况测试
    print("\nTest 3: Edge cases")
    print("-" * 30)

    # 测试3a: 完全匹配的情况
    try:
        perfect_output = (
            torch.ones(batch_size, num_classes, height, width) * 10
        )  # 高置信度正例
        perfect_target = torch.ones(batch_size, num_classes, height, width)
        perfect_loss = loss_fn(perfect_output, perfect_target)
        print(f"Perfect match loss: {perfect_loss.item():.6f}")

    except Exception as e:
        print(f"✗ Error in perfect match test: {e}")

    # 测试3b: 完全不匹配的情况
    try:
        worst_output = (
            torch.ones(batch_size, num_classes, height, width) * 10
        )  # 高置信度正例
        worst_target = torch.zeros(
            batch_size, num_classes, height, width
        )  # 实际全为负例
        worst_loss = loss_fn(worst_output, worst_target)
        print(f"Worst case loss: {worst_loss.item():.6f}")

    except Exception as e:
        print(f"✗ Error in worst case test: {e}")

    # 测试3c: 单个类别测试
    try:
        single_class_output = torch.randn(batch_size, 1, height, width)
        single_class_target = torch.randint(
            0, 2, (batch_size, 1, height, width)
        ).float()
        single_loss = loss_fn(single_class_output, single_class_target)
        print(f"Single class loss: {single_loss.item():.6f}")

    except Exception as e:
        print(f"✗ Error in single class test: {e}")

    # 测试4: 数值稳定性测试
    print("\nTest 4: Numerical stability")
    print("-" * 30)
    try:
        # 极端值测试
        extreme_output = torch.tensor(
            [[[[-1000.0, 1000.0], [-1000.0, 1000.0]]]]
        ).repeat(batch_size, num_classes, 1, 1)
        extreme_target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]).repeat(
            batch_size, num_classes, 1, 1
        )
        extreme_loss = loss_fn(extreme_output, extreme_target)

        print(f"Extreme values loss: {extreme_loss.item():.6f}")
        print(f"Loss is finite: {torch.isfinite(extreme_loss)}")
        print(
            "✓ Numerical stability test passed"
            if torch.isfinite(extreme_loss)
            else "✗ Numerical instability detected"
        )

    except Exception as e:
        print(f"✗ Error in numerical stability test: {e}")

    # 测试5: 检查DiceFocalLoss的reduction参数影响
    print("\nTest 5: Loss reduction behavior")
    print("-" * 30)
    try:
        # 手动验证reduction="none"的行为
        test_output = torch.randn(2, 3, 8, 8)
        test_target = torch.randint(0, 2, (2, 3, 8, 8)).float()

        # 使用你的损失函数
        your_loss = loss_fn(test_output, test_target)

        # 手动计算每个通道的损失来验证
        manual_losses = []
        dice_focal = DiceFocalLoss(sigmoid=True, reduction="none")
        for i in range(test_output.shape[1]):
            channel_loss = dice_focal(
                test_output[:, i : i + 1], test_target[:, i : i + 1]
            )
            manual_losses.append(channel_loss)

        manual_mean = torch.stack(manual_losses).mean()

        print(f"Your loss function result: {your_loss.item():.6f}")
        print(f"Manual calculation result: {manual_mean.item():.6f}")
        print(f"Difference: {abs(your_loss.item() - manual_mean.item()):.8f}")
        print(
            "✓ Loss calculation verified"
            if abs(your_loss.item() - manual_mean.item()) < 1e-6
            else "⚠ Loss calculation mismatch"
        )

    except Exception as e:
        print(f"✗ Error in reduction behavior test: {e}")

    print("\n" + "=" * 50)
    print("Testing completed!")

    # 总结潜在问题
    print("\nPotential considerations for your loss function:")
    print(
        "1. The loss uses sigmoid=True, so make sure your model outputs raw logits (not probabilities)"
    )
    print(
        "2. Each channel is treated independently - overlapping labels are handled correctly"
    )
    print(
        "3. The reduction='none' parameter means loss is computed per sample, then averaged"
    )
    print(
        "4. Consider if you want to weight different classes differently based on their importance"
    )
