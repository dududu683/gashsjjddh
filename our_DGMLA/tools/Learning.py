import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    """Cosine annealing with warmup period"""
    def __init__(self, optimizer, T_max, warmup_epochs=5, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * (1 + torch.cos(torch.tensor(progress * np.pi))) / 2
                    for base_lr in self.base_lrs]

