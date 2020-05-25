import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.trainer import Trainer
from utils.utils import initialize_config

def main(config, resume):
    """
    训练脚本的入口函数

    Args:
        config (dict): 配置项
        resume (bool): 是否加载最近一次存储的模型断点
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_dataset = initialize_config(config["train_dataset"])
    validation_dataset = initialize_config(config["validation_dataset"])
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"]
    )

    collate_all_data = lambda data_list: data_list
    valid_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config["validation_dataloader"]["batch_size"],
        num_workers=config["validation_dataloader"]["num_workers"],
        collate_fn=collate_all_data
    )

    generator = initialize_config(config["generator_model"])
    discriminator = initialize_config(config["discriminator_model"])

    generator_optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=config["optimizer"]["G_lr"]
    )
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=config["optimizer"]["D_lr"],
        betas=(config["optimizer"]["beta1"], 0.999)
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        G=generator,
        D=discriminator,
        optim_G=generator_optimizer,
        optim_D=discriminator_optimizer,
        loss_function=loss_function,
        train_dl=train_data_loader,
        validation_dl=valid_data_loader,
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pix2pix for Speech Enhancement')
    parser.add_argument("-C", "--config", required=True, type=str, help="训练配置文件（*.json）")
    parser.add_argument('-D', '--device', default=None, type=str, help="本次实验使用的 GPU 索引，e.g. '1,2,3'")
    parser.add_argument("-R", "--resume", action="store_true", help="是否从最近的一个断点处继续训练")
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # load config file
    config = json.load(open(args.config))
    config["train_config_path"] = args.config

    main(config, resume=args.resume)
