class Trainer:
    def __init__(self, model, optimizer, train_loader, evaluator, max_iters, eval_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.max_iters = max_iters
        self.eval_interval = eval_interval

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
                print(f"step {i}: "
                      #f"train loss {metrics['train_loss']:.4f} "
                      f"perplexity: {metrics['train_perplexity']:.1f}, "
                      #f"val loss {metrics['val_loss']:.4f} "
                      )
        
        return metrics
