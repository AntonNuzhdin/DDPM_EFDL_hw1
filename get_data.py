import os
from torchvision.datasets import CIFAR10


if __name__ == "__main__":
    os.makedirs("cifar10", exist_ok=True)
    CIFAR10(root="cifar10", train=True, download=True)
    CIFAR10(root="cifar10", train=False, download=True)
