import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb
import hydra 
import random
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from hydra.utils import instantiate
from datetime import datetime

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel
from utils import set_random_seed


# @hydra.main(version_base=None, config_path="configs", config_name="config") # As I use hydra & DVC
def main():
    config = OmegaConf.load("params.yaml")
    set_random_seed()
    device = config.trainer.device

    project_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, project_config)

    os.makedirs('samples', exist_ok=True)
    # Логируем полный конфиг как артефакт
    config_filename = "config.yaml"
    OmegaConf.save(config, config_filename)
    artifact = wandb.Artifact("config", type="config")
    artifact.add_file(config_filename)
    wandb.log_artifact(artifact)

    eps_model = instantiate(config.unet)
    ddpm = instantiate(config.model, eps_model=eps_model)
    ddpm.to(device)

    # seems useless, but I did not see it before:)) 
    wandb.watch(ddpm, log="all", log_freq=100)

    transform_list = []
    if config.augmentation.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5)) # can preprocess mean and std on the train set of CIFAR, but not necessary 
    ])
    train_transforms = transforms.Compose(transform_list)

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=False,
        transform=train_transforms,
    )
    
    dataloader = instantiate(config.dataloader, dataset=dataset, shuffle=True)
    optim = instantiate(config.optimizer, params=ddpm.parameters())

    n_epochs = config.trainer.get("n_epochs", 100)
    for epoch in range(n_epochs):
        writer.set_step(epoch, mode="train")

        loss_ema, lr, first_batch = train_epoch(ddpm, dataloader, optim, device)
        writer.add_image(image_name='input_batch', image=make_grid(first_batch))
        writer.add_scalar(scalar_name='train_loss', scalar=loss_ema)
        writer.add_scalar(scalar_name='lr', scalar=lr)
        
        grid = generate_samples(ddpm, device, f"samples/{epoch:02d}.png")
        writer.add_image(image_name='output_batch', image=grid)

    weights_filename = "ddpm_weights.pth"
    torch.save(ddpm.state_dict(), weights_filename)
    print(f"Model weights saved successfully to {weights_filename}")


if __name__ == "__main__":
    main()
