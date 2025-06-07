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
import torch.nn.functional as F
from einops import asnumpy
import pandas as pd
from sklearn import metrics


# import transforms as self_transforms
# from loader import ImageFolder
from dataset import MyDataset, build_transform, build_transform_2, get_weighted_sampler


import ast
import numpy as np
import misc


from misc import NativeScalerWithGradNormCount as NativeScaler

from monai.losses.dice import DiceLoss, DiceFocalLoss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

COLOR = (125, 125, 125), (255, 0, 0), (0, 255, 0), (0, 0, 255)


def plot_cm(ground_truths, predictions, save_path):
    cm = confusion_matrix(ground_truths, predictions)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Percentage)")

    fig.savefig(save_path)
    plt.close()


def main(args):
    misc.setup_print()
    result_path = Path(args.result_root_path) / args.result_name
    output_path = result_path / args.checkpoint_dir
    args.metrics_path = result_path / args.metrics_folder

    dict_args = args.__dict__.copy()
    for key in sorted(dict_args.keys()):
        print(f'"{key}": "{dict_args[key]}"')

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
    train_transform = build_transform_2(
        is_train="train", img_size=args.input_size, mean=mean, std=std
    )
    val_transform = build_transform_2(
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
        transform=val_transform,
    )
    print(f"Class counts: {dataset_train.class_counts}")

    sampler_train = get_weighted_sampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_val = torch.utils.data.RandomSampler(dataset_val)

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

    if args.multiple_unetr:
        from vitunetr_multi import vit_unetr_base, vit_unetr_large
    else:
        from vitunetr import vit_unetr_base, vit_unetr_large

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
        config_lora_1 = LoraConfig(
            r=args.lora_rank,  # LoRA的秩
            lora_alpha=args.lora_alpha,  # LoRA的alpha参数, scaling=alpha/r
            target_modules=target_modules,  # 需要应用LoRA的模块名称
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            # task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, config_lora_1, adapter_name="lora_1", mixed=True)
        print(model)

    # eval
    elif args.eval:
        config_lora_1 = LoraConfig(
            r=args.lora_rank,  # LoRA的秩
            lora_alpha=args.lora_alpha,  # LoRA的alpha参数, scaling=alpha/r
            target_modules=target_modules,  # 需要应用LoRA的模块名称
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            # task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, config_lora_1, adapter_name="lora_1", mixed=True)

        checkpoint = torch.load(args.finetune, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")

    print(model)
    for name, param in model.named_parameters():
        if "encoder" in name:
            if "lora" in name:
                param.requires_grad = True
            elif "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif "decoder" in name:
            param.requires_grad = False

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

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
        {"params": decoder_params, "lr": args.lr},  # 分割解码器使用更高的学习率
        {"params": cls_head_params, "lr": args.lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = DiceFocalLoss(
        sigmoid=True,
        reduction="mean",
    )
    device = args.device

    if args.eval:
        test(
            model,
            criterion_cls,
            criterion_seg,
            val_loader,
            device,
            mode="val",
            log_writer=log_writer,
            args=args,
        )

        # exit(0)
        pass

    print(f"Start training for {args.epochs} epochs")

    for epoch in range(args.start_epoch, args.epochs):
        train(
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
        )

        val_loss_total, val_loss_cls, val_loss_seg = evaluate(
            model,
            criterion_cls,
            criterion_seg,
            val_loader,
            device,
            epoch,
            mode="val",
            log_writer=log_writer,
            args=args,
        )

        # 在每个epoch后添加
        print(f"Epoch {epoch}: Current LR = {optimizer.param_groups[2]['lr']}")

        # 记录当前学习率到tensorboard
        current_lr = optimizer.param_groups[0]["lr"]
        if log_writer is not None:
            log_writer.add_scalar("lr/current_lr", current_lr, epoch)

        if args.checkpoint_dir:
            to_save = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }
            checkpoint_path = output_path / f"checkpoint_{epoch}.pth"
            torch.save(to_save, checkpoint_path)


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
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter

    image_count = 0
    max_images = 5

    for data_iter_step, (inputs, masks, labels, file_names) in enumerate(
        metric_logger.log_every(
            data_loader,
            print_freq,
            header,
        )
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # move to device
        inputs = inputs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        list_loss_cls = []
        list_loss_seg = []
        list_loss_totals = []

        if torch.cuda.is_bf16_supported():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs_cls, outputs_seg = model(inputs)

                loss_cls = criterion_cls(outputs_cls, labels)
                loss_seg = criterion_seg(outputs_seg, masks)
                # loss = loss_cls + 100 * loss_seg
                loss = loss_cls

        else:
            outputs_cls, outputs_seg = model(inputs)

            loss_cls = criterion_cls(outputs_cls, labels)
            loss_seg = criterion_seg(outputs_seg, masks)
            # loss = loss_cls + 100 * loss_seg
            loss = loss_cls

        # ------------------------------------

        if (labels.max() >= 3) and (image_count < max_images):  # 保存前5个batch的结果
            idx = torch.argmax(labels)

            save_path = args.metrics_path / f"train_visualizations_epoch_{epoch}"
            save_path.mkdir(exist_ok=True, parents=True)

            # 保存原图、真实mask、预测mask
            fig, axes = plt.subplots(2, args.nb_classes_seg + 1, figsize=(20, 8))

            # 原图
            img = inputs[idx].cpu().numpy().transpose(1, 2, 0)
            mean, std = [
                (0.423737496137619, 0.2609460651874542, 0.128403902053833),
                (0.29482534527778625, 0.20167365670204163, 0.13668020069599152),
            ]
            img = (img * std + mean).clip(0, 1)
            axes[0, 0].imshow(img)
            axes[0, 0].set_title("Input Image")

            # 显示4个类别的mask
            for i in range(args.nb_classes_seg):
                axes[0, i + 1].imshow(masks[idx, i].cpu(), cmap="gray")
                axes[0, i + 1].set_title(f"GT Mask {args.mask_folder_names[i]}")

                pred = torch.sigmoid(outputs_seg[idx, i]).cpu() > 0.5
                axes[1, i + 1].imshow(pred, cmap="gray")
                axes[1, i + 1].set_title(f"Pred Mask {args.mask_folder_names[i]}")

            plt.tight_layout()
            plt.savefig(save_path / f"sample_{file_names[idx]}.png")
            plt.close()
            image_count += 1
        # ------------------------------------

        list_loss_cls.append(asnumpy(loss_cls))
        list_loss_seg.append(asnumpy(loss_seg))
        list_loss_totals.append(asnumpy(loss))

        loss_value = loss.item()

        metric_logger.update(
            loss=loss_value, loss_cls=loss_cls.item(), loss_seg=loss_seg.item()
        )

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

    output_loss_total = np.mean(list_loss_totals)
    output_loss_cls = np.mean(list_loss_cls)
    output_loss_seg = np.mean(list_loss_seg)
    print(
        f"train Epoch {epoch}: Loss: {output_loss_total}, Loss_cls: {output_loss_cls}, Loss_seg: {output_loss_seg}\n"
    )


@torch.no_grad()
def test(
    model,
    criterion_cls,
    criterion_seg,
    data_loader,
    device,
    mode="val",
    log_writer=None,
    args=None,
):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    list_names = []
    list_labels = []
    list_outputs_cls_prob = []
    list_outputs_cls = []

    list_loss_cls = []
    list_loss_seg = []
    list_loss_totals = []

    criterion_cls = nn.CrossEntropyLoss(reduction="none")
    criterion_seg = DiceFocalLoss(
        sigmoid=True,
        reduction="none",
    )

    # switch to evaluation mode

    for data_iter_step, (inputs, masks, labels, file_names) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        ouput_path = Path(args.result_root_path) / args.result_name / "image_with_mask"

        if not ouput_path.exists():
            ouput_path.mkdir(parents=True, exist_ok=True)

        inputs = inputs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output_cls, ouputs_seg = model(inputs)

        batch_size = inputs.shape[0]
        for idx in range(batch_size):
            file_path = Path(args.data_path) / "image" / file_names[idx]

            import cv2
            from skimage import io

            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for mask_idx, mask_name in enumerate(args.mask_folder_names):
                mask = ouputs_seg[idx, mask_idx]
                mask = torch.sigmoid(mask)

                mask = asnumpy(mask).astype(np.float32)

                mask = cv2.resize(
                    mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
                )
                mask = mask > 0.5
                mask = (mask * 255).astype(np.uint8)
                # _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img, contours, -1, COLOR[mask_idx], 2)

            img_output_path = ouput_path / file_names[idx]
            io.imsave(img_output_path, img)


@torch.no_grad()
def evaluate(
    model,
    criterion_cls,
    criterion_seg,
    data_loader,
    device,
    epoch,
    mode="val",
    log_writer=None,
    args=None,
):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    list_names = []
    list_labels = []
    list_outputs_cls_prob = []
    list_outputs_cls = []

    list_loss_cls = []
    list_loss_seg = []
    list_loss_totals = []

    criterion_cls = nn.CrossEntropyLoss(reduction="none")
    criterion_seg = DiceFocalLoss(
        sigmoid=True,
        reduction="mean",
    )

    image_count = 0
    max_images = 5

    # switch to evaluation mode

    for data_iter_step, (inputs, masks, labels, file_names) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        inputs = inputs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        list_batch_loss_cls = []
        list_batch_loss_seg = []
        list_batch_loss_totals = []

        # compute output
        with torch.cuda.amp.autocast():
            outputs_cls, outputs_seg = model(inputs)
            outputs_cls_prob = F.softmax(outputs_cls, dim=-1)

        # ------------------------------------
        if (labels.max() >= 3) and (image_count < max_images):
            idx = torch.argmax(labels)

            save_path = args.metrics_path / f"visualizations_epoch_{epoch}"
            save_path.mkdir(exist_ok=True, parents=True)

            # 保存原图、真实mask、预测mask
            fig, axes = plt.subplots(2, args.nb_classes_seg + 1, figsize=(20, 8))

            # 原图
            img = asnumpy(inputs[idx]).transpose(1, 2, 0)
            mean, std = [
                (0.423737496137619, 0.2609460651874542, 0.128403902053833),
                (0.29482534527778625, 0.20167365670204163, 0.13668020069599152),
            ]
            img = (img * std + mean).clip(0, 1)
            axes[0, 0].imshow(img)
            axes[0, 0].set_title("Input Image")

            # 显示4个类别的mask
            for i in range(args.nb_classes_seg):
                axes[0, i + 1].imshow(masks[idx, i].cpu(), cmap="gray")
                axes[0, i + 1].set_title(f"GT Mask {args.mask_folder_names[i]}")

                pred = torch.sigmoid(outputs_seg[idx, i]).cpu() > 0.5
                axes[1, i + 1].imshow(pred, cmap="gray")
                axes[1, i + 1].set_title(f"Pred Mask {args.mask_folder_names[i]}")

            plt.tight_layout()
            plt.savefig(save_path / f"sample_{file_names[idx]}.png")
            plt.close()
            image_count += 1
        # ------------------------------------

        batch_size = inputs.shape[0]

        for idx in range(batch_size):
            label = labels[idx]
            mask = masks[idx]
            output_cls_prob = outputs_cls_prob[idx]

            output_cls = outputs_cls[idx]
            output_seg = outputs_seg[idx]

            loss_cls = criterion_cls(output_cls, label)
            loss_seg = criterion_seg(output_seg, mask)
            loss = loss_seg + loss_cls
            output_cls_pred = output_cls.argmax(dim=-1)

            list_names.append(file_names[idx])
            list_labels.append(asnumpy(label))
            list_outputs_cls.append(asnumpy(output_cls_pred))
            list_outputs_cls_prob.append(asnumpy(output_cls_prob))
            list_loss_cls.append(asnumpy(loss_cls))
            list_loss_seg.append(asnumpy(loss_seg))
            list_loss_totals.append(asnumpy(loss))

            list_batch_loss_cls.append(asnumpy(loss_cls))
            list_batch_loss_seg.append(asnumpy(loss_seg))
            list_batch_loss_totals.append(asnumpy(loss))

        acc = metrics.accuracy_score(list_labels, list_outputs_cls)

        metric_logger.update(
            loss=float(np.mean(list_batch_loss_totals)),
            loss_cls=float(np.mean(list_batch_loss_cls)),
            loss_seg=float(np.mean(list_batch_loss_seg)),
        )
        metric_logger.meters["acc"].update(acc, n=batch_size)

    if not args.metrics_path.exists():
        args.metrics_path.mkdir(parents=True, exist_ok=True)

    # save results, xujia
    pd.DataFrame(
        {
            "names": list_names,
            "labels": list_labels,
            "outputs": list_outputs_cls,
            "outputs_prob": list_outputs_cls_prob,
            "loss_cls": list_loss_cls,
            "loss_seg": list_loss_seg,
            "loss_total": list_loss_totals,
        }
    ).to_csv(args.metrics_path / f"results_{epoch}.csv")

    list_outputs_cls = list_outputs_cls
    if args.nb_classes_cls > 2:
        try:
            auc_macro = metrics.roc_auc_score(
                list_labels, list_outputs_cls_prob, multi_class="ovr", average="macro"
            )
        except ValueError as e:
            print(f"Error calculating AUC macro: {e}")
            auc_macro = 0.0

        kappa = metrics.cohen_kappa_score(
            list_labels, list_outputs_cls, weights="quadratic"
        )
        accuracy = metrics.accuracy_score(list_labels, list_outputs_cls)
        f1 = metrics.f1_score(list_labels, list_outputs_cls, average="macro")
        precision = metrics.precision_score(
            list_labels, list_outputs_cls, average="macro"
        )
        sensitivity = metrics.recall_score(
            list_labels, list_outputs_cls, average="macro"
        )
    else:
        list_outputs_cls_prob = [output[1] for output in list_outputs_cls_prob]

        try:
            auc_macro = metrics.roc_auc_score(
                list_labels,
                list_outputs_cls_prob,
            )
        except ValueError:
            auc_macro = 0.0

        kappa = metrics.cohen_kappa_score(list_labels, list_outputs_cls)
        accuracy = metrics.accuracy_score(list_labels, list_outputs_cls)
        f1 = metrics.f1_score(list_labels, list_outputs_cls)
        precision = metrics.precision_score(list_labels, list_outputs_cls)
        sensitivity = metrics.recall_score(list_labels, list_outputs_cls)

    output_loss_total = np.mean(list_loss_totals)
    output_loss_cls = np.mean(list_loss_cls)
    output_loss_seg = np.mean(list_loss_seg)

    cm_path = args.metrics_path / f"cm_{epoch}.png"
    plot_cm(list_labels, list_outputs_cls, cm_path)

    with open(
        os.path.join(args.metrics_path, "metrics.csv"),
        "a+",
    ) as txt:
        if epoch == 0:
            txt.write(
                f"Mode,Epoch,AUC_macro,F1,Kappa,Accuracy,Precision,Sensitivity,Loss,Loss_cls,loss_seg\n"
            )

        txt.write(
            f"{mode},{epoch},{auc_macro},{f1},{kappa},{accuracy},{precision},{sensitivity},{output_loss_total},{output_loss_cls},{output_loss_seg}\n"
        )

    print(
        f"{mode} Epoch {epoch}: AUC macro: {auc_macro}, F1: {f1}, Kappa: {kappa}, Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Loss: {output_loss_total}, Loss_cls: {output_loss_cls}, Loss_seg: {output_loss_seg}\n"
    )
    torch.cuda.empty_cache()

    if log_writer is not None:
        log_writer.add_scalar("perf/val_acc", accuracy, epoch)
        log_writer.add_scalar("perf/val_auc", auc_macro, epoch)

        log_writer.add_scalar("perf/val_total_loss", output_loss_total, epoch)
        log_writer.add_scalar("perf/val_loss_cls", output_loss_cls, epoch)
        log_writer.add_scalar("perf/val_loss_seg", output_loss_seg, epoch)

    return output_loss_total, output_loss_cls, output_loss_seg


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
        "--multiple_unetr",
        action="store_true",
        help="use multiple unetr heads.",
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
        "--checkpoint_dir", default="./ckpts", help="Path to save logs and checkpoints"
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

    # parser.add_argument(
    #     "--lr_decay_factor",
    #     type=float,
    #     default=0.5,
    #     help="Factor by which the learning rate will be reduced (default: 0.5)"
    # )
    # parser.add_argument(
    #     "--lr_patience",
    #     type=int,
    #     default=5,
    #     help="Number of epochs with no improvement after which learning rate will be reduced (default: 5)"
    # )
    # parser.add_argument(
    #     "--lr_threshold",
    #     type=float,
    #     default=1e-4,
    #     help="Threshold for measuring the new optimum (default: 1e-4)"
    # )
    # parser.add_argument(
    #     "--lr_cooldown",
    #     type=int,
    #     default=2,
    #     help="Number of epochs to wait before resuming normal operation after lr has been reduced (default: 2)"
    # )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
