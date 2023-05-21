from pytorch_lightning import Trainer
from pl_module import SegmentationModule
import argparse
from data import BuildingDataset
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import VisualizeCallback
from clearml import Task
import torch

torch.set_float32_matmul_precision("medium")


def make_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"train_logs/{args.desc}/checkpoints",
        monitor="val_iou",
        mode="max",
        filename="best",
    )
    visualize_callback = VisualizeCallback(args=args)
    return [checkpoint_callback, visualize_callback]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--root_folder", type=str, default="dataset/chips-1024-full")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--desc", type=str)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--visualize_input", default="/mnt/data/RasterMask_v11")
    parser.add_argument("--visualize_output", default="visualize")
    parser.add_argument("--tags", type=lambda s: s.split(","))
    args = parser.parse_args()
    args.img_size = (args.img_size, args.img_size)
    return args


if __name__ == "__main__":
    args = get_args()
    task = Task.init(
        project_name="DroneSegmentation/Building", task_name=args.desc, tags=args.tags
    )
    print(args)
    train_dataset = BuildingDataset(
        root_folder=args.root_folder, img_size=args.img_size, is_training=True
    )
    val_dataset = BuildingDataset(
        root_folder=args.root_folder, img_size=args.img_size, is_training=False
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16)

    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    L = SegmentationModule(model=model, args=args)
    logger = TensorBoardLogger("train_logs", name="segformer", version=args.desc)
    callbacks = make_callbacks(args)
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate,
    )
    trainer.fit(
        model=L, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
