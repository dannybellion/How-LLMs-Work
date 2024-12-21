from dataclasses import dataclass
from typing import Tuple, Dict
import torch

@dataclass
class BatchLoader:
    """Handles data loading and batch generation"""
    data: torch.Tensor
    block_size: int
    batch_size: int

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a random batch of data"""
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y
    

class Evaluator:
    def __init__(self, model, train_loader: BatchLoader, val_loader: BatchLoader):
        self.model = model
        self.loaders = {'train': train_loader, 'val': val_loader}

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int) -> Dict[str, float]:
        """Estimate loss across multiple iterations"""
        out = {}
        self.model.eval()
        
        for split, loader in self.loaders.items():
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = loader()
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            
        self.model.train()
        return out