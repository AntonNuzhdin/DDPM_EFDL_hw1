import torch
import wandb
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image


class WandbWriter:
    def __init__(
        self,
        project_config,
        project_name,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name
                is given.
            mode (str): if online, log data to the remote server. If
                offline, log locally.
        """

        wandb.login()

        self.run_id = run_id

        wandb.init(
            project=project_name,
            config=project_config,
            name=run_name,
            resume="allow",  # resume the run if run_id existed
            id=self.run_id,
            mode=mode,
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name). Used to separate metrics
        from different partitions.

        Args:
            object_name (str): current object name.
        Returns:
            object_name (str): updated object name.
        """
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.

        The checkpoints will be available in the files section
        inside the run_name dir.

        Args:
            checkpoint_path (str): path to the checkpoint file.
            save_dir (str): path to the dir, where checkpoint is saved.
        """
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )


    def add_image(self, image_name, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            image = image.numpy()
            if np.issubdtype(image.dtype, np.floating):
                if image.min() < 0:
                    image = (image + 1.0) / 2.0 
                image = (image * 255).astype(np.uint8)
        self.wandb.log(
            {self._object_name(image_name): self.wandb.Image(image)},
            step=self.step
        )
 