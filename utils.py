import random
import torch
import numpy as np


# for exps reproducibility from https://github.com/open-mmlab/mmengine/blob/85c83ba61689907fb1775713622b1b146d82277b/mmengine/runner/utils.py#L48
def set_random_seed(seed: int = 42,
                    deterministic: bool = False) -> int:
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return seed