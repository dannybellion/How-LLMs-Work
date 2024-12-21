from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, train_loader, evaluator, max_iters, eval_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.max_iters = max_iters
        self.eval_interval = eval_interval

    def train(self):
        pbar = tqdm(range(self.max_iters), desc="Training")
        for i in pbar:
            # Training step
            xb, yb = self.train_loader()
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            # Evaluation step
            if i % self.eval_interval == 0:
                losses = self.evaluator.estimate_loss(eval_iters=200)
                pbar.set_postfix({
                    'train_loss': f"{losses['train']:.4f}",
                    'val_loss': f"{losses['val']:.4f}"
                })
        
        return losses
