#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
from argparse import Namespace

import numpy as np
import torch

# Shakhboz's imports
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from solo.data.pretrain_dataloader import (
    build_transform_pipeline,
    prepare_n_crop_transform,
)

# Solo-Learn import
from solo.methods import (  # imports the method class
    BYOL,
    MAE,
    BarlowTwins,
    MoCoV2Plus,
    MoCoV3,
    SimCLR,
    SimSiam,
)
from solo.utils.checkpointer import Checkpointer

import benthic_data_classes.datasets

# and some utilities to perform data loading for the method itself, including augmentation pipelines


METHODS = {
    "bt": BarlowTwins,
    "simclr": SimCLR,
    "mocov2+": MoCoV2Plus,
    "mocov3": MoCoV3,
    "mae": MAE,
    "simsiam": SimSiam,
    "byol": BYOL,
}


def main():
    parser = argparse.ArgumentParser(
        description="Parameters for SSL benthic habitat project"
    )

    # Required parameters
    parser.add_argument(
        "--ssl_cfg", type=str, required=True, help="set cfg file for SSL"
    )
    parser.add_argument(
        "--aug_stack_cfg",
        type=str,
        required=True,
        help="set cfg file for augmentations",
    )
    parser.add_argument("--nodes", type=int, required=True, help="number of nodes")
    parser.add_argument(
        "--gpus", type=int, required=True, help="number of gpus per node"
    )
    parser.add_argument("--method", type=str, required=True, help="type of SSL method")
    # Other parameters
    parser.add_argument("--mini", type=bool, default=False, help="use mini-dataset")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--name",
        type=str,
        default="self-supervised_learning",
        help="set name for the run",
    )

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    root_dir = "/project/rrg-ttt/become/benthicnet-compiled/compiled_250s_512px/"
    csv_file_name = "dataset_2022-04-22.csv"
    if args.mini:
        csv_file_name = "mini-" + csv_file_name
    csv_file_ssl = root_dir + csv_file_name
    tar_dir = "/project/rrg-ttt/become/benthicnet-compiled/compiled_labelled_512px/tar/"
    lab_csv_file = "./Catami/WFdataset_subd3.csv"

    (
        _,
        validation_data,
        test_same_data,
        test_other_data,
    ) = benthic_data_classes.datasets.get_dataset_by_station_split(lab_csv_file)

    # common parameters for all methods
    # some parameters for extra functionally are missing, but don't mind this for now.

    ssl_cfg_name = args.ssl_cfg

    with open("./ssl_cfgs/" + ssl_cfg_name) as f:
        ssl_cfg = f.read()

    kwargs = json.loads(ssl_cfg)
    cfg = OmegaConf.create(kwargs)

    model = METHODS[args.method](cfg)

    # we first prepare our single transformation pipeline

    aug_stack_name = args.aug_stack_cfg

    with open("./ssl_cfgs/aug_stacks/" + aug_stack_name) as f:
        aug_stack_cfg = f.read()

    transform_kwargs = json.loads(aug_stack_cfg)
    transform_cfg = OmegaConf.create(transform_kwargs)

    transform = build_transform_pipeline("custom", transform_cfg)

    # then, we wrap the pipepline using this utility function
    # to make it produce an arbitrary number of crops
    transform = prepare_n_crop_transform(
        [transform], num_crops_per_aug=[kwargs["data"]["num_large_crops"]]
    )

    train_dataset = benthic_data_classes.datasets.BenthicNetDatasetSSL(
        root_dir, csv_file_ssl, transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=kwargs["optimizer"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=kwargs["num_workers"],
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),  # resize shorter
            transforms.CenterCrop(224),  # take center crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
        ]
    )

    val_dataset = benthic_data_classes.datasets.BenthicNetDataset(
        tar_dir, validation_data, val_transforms
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=kwargs["optimizer"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=kwargs["num_workers"],
    )

    os.environ["WANDB_NOTEBOOK_NAME"] = "./solo_learn_train-bentho.ipynb"

    run_name = args.name

    wandb_logger = pl_loggers.WandbLogger(
        name=run_name,  # name of the experiment
        project="self-supervised_benthoscape",  # name of the wandb project
        entity=None,
        offline=False,
    )
    wandb_logger.watch(model, log="gradients", log_freq=100)

    callbacks = []

    # automatically log our learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # checkpointer can automatically log your parameters,
    # but we need to wrap it on a Namespace object

    ckpt_args = OmegaConf.create({"name": run_name})
    # saves the checkout after every epoch

    ckpt = Checkpointer(
        ckpt_args,
        logdir="./checkpoints/" + run_name,
        frequency=kwargs["max_epochs"] - 1,
    )
    callbacks.append(ckpt)

    trainer_args = Namespace(**kwargs)

    trainer = Trainer.from_argparse_args(
        trainer_args,
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="cuda",
        num_nodes=args.nodes,
        devices=args.gpus,
        log_every_n_steps=200,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
