import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str, writer):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    losses = []
    first_batch_logged = False 
    for x, _ in pbar:
        if not first_batch_logged:
            writer.add_image(image_name='input_batch', image=make_grid(x))
            first_batch_logged = True

        train_loss = train_step(model, x, optimizer, device)
        writer.add_scalar(scalar_name='train_loss', scalar=train_loss)
        losses.append(train_loss)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar(scalar_name='learning_rate', scalar=current_lr)
    return sum(losses) / len(losses)


def generate_samples(model: DiffusionModel, device: str, path: str):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, nrow=4)
        save_image(grid, path)
    return grid
