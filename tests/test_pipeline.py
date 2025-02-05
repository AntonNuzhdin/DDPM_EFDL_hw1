import os
import sys
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from PIL import Image

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import set_random_seed


@pytest.fixture(scope="session", autouse=True)
def seed():
    set_random_seed(123)


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(device, train_dataset):
    def train(ddpm, optim, index):
        ddpm = DiffusionModel(
            eps_model=UnetModel(3, 3, hidden_size=32),
            betas=(1e-4, 0.02),
            num_timesteps=1000,
        )
        ddpm.to(device)

        train_subset = Subset(train_dataset, list(range(0, 8)))

        dataloader = DataLoader(train_subset, batch_size=4, shuffle=True)
        losses = []
        for _ in range(0, 2):
            loss, _, _ = train_epoch(ddpm, dataloader, optim, device)
            losses.append(loss)
        os.makedirs('test_samples', exist_ok=True)
        generate_samples(ddpm, device, os.path.join('test_samples', f'test_img_{index}.png'))
        return sum(losses) / len(losses)

    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim1 = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    optim2 = torch.optim.Adam(ddpm.parameters(), lr=3e-4)

    loss1 = train(ddpm, optim1, 1)
    loss2 = train(ddpm, optim2, 2)
    assert loss1 != loss2

    img1 = Image.open('test_samples/test_img_1.png').convert('RGB')
    img2 = Image.open('test_samples/test_img_2.png').convert('RGB')

    assert img1 != img2



