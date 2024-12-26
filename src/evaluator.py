import torch
from typing import Dict
from src.utils import BatchLoader

class Evaluator:
    def __init__(self, model, train_loader: BatchLoader, val_loader: BatchLoader, vocab_size: int):
        self.model = model
        self.loaders = {'train': train_loader, 'val': val_loader}
        self.vocab_size = vocab_size

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int) -> Dict[str, float]:
        """Estimate loss metrics across multiple iterations"""
        out = {}
        self.model.eval()
        
        for split, loader in self.loaders.items():
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = loader()
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            avg_loss = losses.mean()
            
            # Calculate metrics
            perplexity = torch.exp(avg_loss).item()
            
            out[f"{split}_loss"] = avg_loss
            out[f"{split}_perplexity"] = perplexity
            
        self.model.train()
        return out