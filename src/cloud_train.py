import torch
from src import BigramLanguageModel, BatchLoader, Evaluator, Trainer
import json

def train_model(config=None):
    # Default parameters
    default_config = {
        "batch_size": 64,
        "block_size": 256,
        "max_iters": 5000,
        "eval_interval": 500,
        "learning_rate": 3e-4,
        "n_embed": 384,
        "n_heads": 6,
        "n_layer": 6,
        "dropout": 0.2
    }
    
    # Update with provided config if any
    if config:
        if isinstance(config, str):
            config = json.loads(config)
        default_config.update(config)

    # Extract parameters
    batch_size = default_config["batch_size"]
    block_size = default_config["block_size"]
    max_iters = default_config["max_iters"]
    eval_interval = default_config["eval_interval"]
    learning_rate = default_config["learning_rate"]
    n_embed = default_config["n_embed"]
    n_heads = default_config["n_heads"]
    n_layer = default_config["n_layer"]
    dropout = default_config["dropout"]

    # Load data
    with open("data/input.txt", "r") as f:
        text = f.read()

    # Create character mapping
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    # Prepare data
    data = torch.tensor(encode(text))
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Setup data loaders
    train_loader = BatchLoader(train_data, block_size=block_size, batch_size=batch_size)
    val_loader = BatchLoader(val_data, block_size=block_size, batch_size=batch_size)

    # Initialize model
    model = BigramLanguageModel(
        vocab_size, 
        n_embed, 
        block_size, 
        n_layer, 
        n_heads, 
        dropout
    ).cuda()  # Move to GPU

    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    evaluator = Evaluator(model, train_loader, val_loader, vocab_size)
    trainer = Trainer(model, optimizer, train_loader, evaluator, max_iters, eval_interval)

    # Train the model
    final_losses = trainer.train()
    
    # Save the model
    torch.save(model.state_dict(), 'model_checkpoint.pt')
    
    return final_losses

if __name__ == "__main__":
    import sys
    config_str = sys.argv[1] if len(sys.argv) > 1 else None
    train_model(config_str)