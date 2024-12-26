from dataclasses import dataclass
from typing import Tuple, Dict
import torch
import paramiko

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

    
def create_encoders(chars):
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode