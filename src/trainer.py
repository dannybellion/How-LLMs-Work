import wandb

class Trainer:
    def __init__(self, model, optimizer, train_loader, evaluator, max_iters, eval_interval=100, use_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.use_wandb = use_wandb

    def train(self):
        for i in range(self.max_iters):
            # Training step
            xb, yb = self.train_loader()
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            # Evaluation step
            if i % self.eval_interval == 0:
                metrics = self.evaluator.estimate_loss(eval_iters=self.eval_interval)
                print(f"step {i}: perplexity: {metrics['train_perplexity']:.1f}, ")
                
                if self.use_wandb:
                    wandb.log({
                        "train_loss": metrics['train_loss'],
                        "val_loss": metrics['val_loss'],
                        "train_perplexity": metrics['train_perplexity'],
                        "val_perplexity": metrics['val_perplexity'],
                        "step": i
                    })
        
        return metrics
