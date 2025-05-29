# Train a segmentation decoder
import math
import sys
from tabnanny import check
import time

from torch.utils.data import sampler
from torch.utils.tensorboard import SummaryWriter
import lr_sched
from peft import LoraConfig, get_peft_model

sys.path.append("./")
import os
import argparse
import json
import copy
import torch
from torch import nn
import torch.backends.cudnn as cudnn

import utils
from pathlib import Path


# import transforms as self_transforms
# from loader import ImageFolder
from dataset import MyDataset, build_transform, get_weighted_sampler

from loss import MultiLabelSegmentationLoss

import ast
import numpy as np
import misc

from vitunetr import vit_unetr_base, vit_unetr_large
from misc import NativeScalerWithGradNormCount as NativeScaler


def main(args):
    misc.setup_print()
    result_path = Path(args.result_root_path) / args.result_name
    output_path = result_path / args.output_dir

    print(f"{args}".replace(", ", ",\n"))

    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)

    log_path = result_path / args.log_dir
    log_writer = SummaryWriter(log_dir=log_path)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # mean, std = utils.get_stats(args.modality)
    mean, std = [
        (0.423737496137619, 0.2609460651874542, 0.128403902053833),
        (0.29482534527778625, 0.20167365670204163, 0.13668020069599152),
    ]

    # ------------
    train_transform = build_transform(
        is_train="train", img_size=args.input_size, mean=mean, std=std
    )
    val_transform = build_transform(
        is_train="val", img_size=args.input_size, mean=mean, std=std
    )

    dataset_train = MyDataset(
        csv_path=args.csv_path,
        data_path=args.data_path,
        mask_folder_names=args.mask_folder_names,
        is_train="train",
        transform=train_transform,
    )
    dataset_val = MyDataset(
        csv_path=args.csv_path,
        data_path=args.data_path,
        mask_folder_names=args.mask_folder_names,
        is_train="val",
        transform=train_transform,
    )
    print(f"Class counts: {dataset_train.class_counts}")

    sampler_train = get_weighted_sampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )

    # ============ building network ... ============
    if args.arch == "vit_unetr_base":
        model_arch = vit_unetr_base
    elif args.arch == "vit_unetr_large":
        model_arch = vit_unetr_large
    else:
        raise ValueError(f"Unsupported model architecture: {args.arch}")

    # working xujia
    model = model_arch(
        num_classes_cls=args.nb_classes_cls,
        num_classes_seg=args.nb_classes_seg,
    )

    model.to(device=args.device)

    # apply lora

    if args.lora_position == "qkv":
        target_modules = ["qkv"]

    if not args.finetune:
        raise ValueError("Please specify a pre-trained model for finetuning")

    # finetune
    if args.finetune and (not args.eval):
        checkpoint = torch.load(args.finetune, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        msg = model.encoder.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")
        config_lora = LoraConfig(
            r=args.lora_rank,  # LoRA的秩
            lora_alpha=args.lora_alpha,  # LoRA的alpha参数, scaling=alpha/r
            target_modules=target_modules,  # 需要应用LoRA的模块名称
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            # task_type="FEATURE_EXTRACTION",
        )
        get_peft_model(model, config_lora)

    # eval
    elif args.eval:
        config_lora = LoraConfig(
            r=args.lora_rank,  # LoRA的秩
            lora_alpha=args.lora_alpha,  # LoRA的alpha参数, scaling=alpha/r
            target_modules=target_modules,  # 需要应用LoRA的模块名称
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            # task_type="FEATURE_EXTRACTION",
        )
        get_peft_model(model, config_lora)

        checkpoint = torch.load(args.finetune, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        msg = model.encoder.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")

    param_groups = model.parameters()

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    encoder_params = []
    decoder_params = []
    cls_head_params = []

    for name, param in model.named_parameters():
        if "decoder" in name:
            decoder_params.append(param)
        elif "head" in name:  # 分类头
            cls_head_params.append(param)
        else:
            encoder_params.append(param)

    # 为不同部分设置不同的学习率
    param_groups = [
        {"params": encoder_params, "lr": args.lr},
        {"params": decoder_params, "lr": args.lr * 10},  # 分割解码器使用更高的学习率
        {"params": cls_head_params, "lr": args.lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = MultiLabelSegmentationLoss()
    device = args.device

    if args.eval:
        # (
        #     test_stats,
        #     test_auc_roc,
        #     test_acc,
        #     (test_output_loss_total, test_output_loss_un, test_output_loss_ce),
        # ) = evaluate(
        #     args=args,
        #     data_loader=data_loader_test,
        #     model=model,
        #     device=device,
        #     epoch=0,
        #     mode="test",
        #     num_class=args.nb_classes,
        # )

        # exit(0)
        pass

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.50)

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train(
            model,
            criterion_cls,
            criterion_seg,
            train_loader,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
            scheduler=scheduler,
        )


def train(
    model,
    criterion_cls,
    criterion_seg,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm=0,
    log_writer=None,
    args=None,
    scheduler=None,
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter

    for data_iter_step, (inputs, masks, labels, file_name) in enumerate(
        metric_logger.log_every(
            data_loader,
            print_freq,
            header,
        )
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            # lr_sched.adjust_learning_rate(
            #     optimizer, data_iter_step / len(data_loader) + epoch, args
            # )
            scheduler.step()

        # move to gpu
        inputs = inputs.to(device=device, non_blocking=True)
        masks = masks.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs_cls, outputs_seg = model(inputs)
            loss_cls = criterion_cls(outputs_cls, labels)
            loss_seg = criterion_seg(outputs_seg, masks)
            # loss = loss_cls + 100 * loss_seg
            loss = loss_seg

        loss_value = loss.item()
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(
            loss=loss_value, loss_cls=loss_cls.item(), loss_seg=loss_seg.item()
        )
        min_lr = 10.0
        max_lr = 0.0

        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network_all(val_loader, model, linear_classifier):
    # compute the metrics on all data, instead of batch style
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    predictions, targets, all_positives, all_img_paths = [], [], [], []
    for inp, target, extras in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(dim=1)

        # forward
        with torch.no_grad():
            n = len(model.blocks)  # get all the layers
            if n == 12:
                selected_levels = [3, 5, 7, 11]  # for default vit-base model
            elif n == 24:  # for vit-large
                selected_levels = [5, 11, 17, 23]
            else:
                raise NotImplementedError  # please set suitable selected_levels
            intermediate_output = model.get_intermediate_layers(inp, n)
            # only retain the patch token in 4 levels
            features = [intermediate_output[idx][:, 1:] for idx in selected_levels]

            output = linear_classifier(features, inp)  # [B, 1, H, W]
            # output = linear_classifier(output)  # [B, 1, H, W]
            num_classes = output.shape[1]
            if num_classes == 1:  # for binary segmentation task
                loss = DiceFocalLoss(sigmoid=True)(output, target)  # B1HW and B1HW
            else:
                loss = DiceFocalLoss(softmax=True, to_onehot_y=True)(
                    output, target
                )  # BCHW and BCHW

        predictions.append(output.cpu())
        targets.append(target.cpu())
        all_img_paths += extras["img_path"]

    predictions = torch.cat(predictions, dim=0)  # [N, 4, H, W]
    targets = torch.cat(targets, dim=0)  # [N, C, H, W]
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[1]

    if num_classes == 1:
        dices = utils.dice(
            torch.sigmoid(predictions.squeeze(dim=1)),
            targets.squeeze(dim=1),
            return_ori=True,
        )
        metric_dice = dices.mean()
    else:
        dices_ori = utils.dice_mc(
            torch.softmax(input=predictions, dim=1),
            targets.squeeze(dim=1),
            n_classes=num_classes,
            return_ori=True,
        )  # [N, 4]
        metric_dice = dices_ori.mean(axis=0).mean()

    # batch_size = inp.shape[0]
    metric_logger.update(loss=loss.item())
    metric_logger.meters["metric_dice"].update(metric_dice, n=batch_size)

    print(
        "* Dice {metric_dice.global_avg:.4f}  Loss {loss.global_avg:.4f}".format(
            metric_dice=metric_logger.metric_dice, loss=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Train a segmentation decoder on top of VisionFM encoder"
    )

    # model
    parser.add_argument(
        "--device", default="cuda:0", help="device to use for training / testing"
    )
    parser.add_argument(
        "--arch",
        default="vit_unetr_base",
        type=str,
        choices=["vit_unetr_base", "vit_unetr_large"],
        help="Architecture.",
    )
    parser.add_argument(
        "--finetune",
        default="",
        type=str,
        help="finetune from checkpoint",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers per GPU.",
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # dataset
    parser.add_argument(
        "--nb_classes_cls",
        default=5,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument(
        "--nb_classes_seg", default=4, type=int, help="number of the segmentation types"
    )

    parser.add_argument(
        "--mask_folder_names",
        default=["EX", "MA", "HE", "SE"],
        type=ast.literal_eval,
        help="the folder names of the masks",
    )
    parser.add_argument(
        "--output_dir", default="./results", help="Path to save logs and checkpoints"
    )
    parser.add_argument("--csv_path", type=str, help="path where csv file is located")
    parser.add_argument("--data_path", type=str, help="path where images are located")
    parser.add_argument(
        "--input_size",
        default=224,
        type=int,
        help="model input size",
    )

    # lora
    parser.add_argument(
        "--lora_position",
        default="qkv",
        choices=["qkv", "all-linear"],
        type=str,
        help="position of lora layer (default: qkv)",
    )
    parser.add_argument(
        "--lora_rank",
        default=8,
        type=int,
        help="lora rank (default: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        default=16,
        type=int,
        help="lora alpha (default: 16)",
    )
    parser.add_argument(
        "--lora_bias",
        default="lora_only",
        type=str,
        help="bias of lora layer (none, all, lora_only, default: lora_only)",
    )
    parser.add_argument(
        "--lora_dropout",
        default=0.1,
        type=float,
        help="dropout rate of lora layer (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        metavar="NORM",
        help="Clip gradient norm (default: 1.0)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: 1e-3)",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # results output parameters
    parser.add_argument(
        "--result_root_path",
        default="./results",
        help="path where results will be saved",
    )
    parser.add_argument(
        "--result_name",
        default="SLO",
        help="path where results will be saved",
    )
    parser.add_argument(
        "--metrics_folder",
        default="metrics",
        help="path where to save metrics, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="log", help="path where to tensorboard log"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
