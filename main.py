import torch
import hydra 
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from hydra.utils import instantiate

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel
from utils import set_random_seed


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config):
    set_random_seed()

    device = config.trainer.device

    project_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, project_config)

    # log the full config like an artifact
    config_filename = "config.yaml"
    OmegaConf.save(config, config_filename)
    artifact = wandb.Artifact("config", type="config")
    artifact.add_file(config_filename)
    wandb.log_artifact(artifact)

    eps_model = instantiate(config.unet)
    ddpm = instantiate(config.model, eps_model=eps_model)

    ddpm.to(device)
    transform_list = []
    if config.augmentation.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_transforms = transforms.Compose(transform_list)

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )
    
    dataloader = instantiate(config.dataloader, dataset=dataset, shuffle=True)
    optim = instantiate(config.optimizer, params=ddpm.parameters())

    for i in range(config.trainer.get("n_epochs")):
        train_epoch(ddpm, dataloader, optim, device, writer)
        grid = generate_samples(ddpm, device, f"samples/{i:02d}.png")
        writer.add_image(image_name='output_batch', image=grid)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()
