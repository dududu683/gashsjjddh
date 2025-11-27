import torch
import os


class CheckpointManager:
    """Save/load model checkpoints with best performance tracking"""

    def __init__(self, save_dir='checkpoints', metric='loss', mode='min'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metric = metric
        self.mode = mode  # 'min' for loss, 'max' for accuracy
        self.best_metric = float('inf') if mode == 'min' else -float('inf')

    def save(self, epoch, model, optimizer, metric_value, scheduler=None):
        """Save checkpoint if metric improves"""
        is_best = (metric_value < self.best_metric) if self.mode == 'min' else (metric_value > self.best_metric)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': self.best_metric,
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest.pth'))

        # Save best checkpoint if improved
        if is_best:
            self.best_metric = metric_value
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pth'))

    def load(self, model, optimizer=None, scheduler=None, checkpoint_path='best.pth'):
        """Load checkpoint"""
        checkpoint = torch.load(os.path.join(self.save_dir, checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint['best_metric']

