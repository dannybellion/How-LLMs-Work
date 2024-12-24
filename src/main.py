import torch
from utils.models import BigramLanguageModel, SimpleBigramLanguageModel
from utils.utils import BatchLoader, Evaluator, create_encoders
from utils.trainer import Trainer

# hyperparameters
batch_size = 32
block_size = 16
max_iters = 2000
eval_interval = 100
learning_rate = 3e-3
eval_iters = 200
n_embed = 32
n_heads = 4
n_layer = 4
dropout = 0.2

# Load and process data
with open("../data/input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
encode, decode = create_encoders(chars)
data = torch.tensor(encode(text))

# Split data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Setup data and model
torch.manual_seed(1337)
train_loader = BatchLoader(train_data, block_size=block_size, batch_size=batch_size)
val_loader = BatchLoader(val_data, block_size=block_size, batch_size=batch_size)
# model = SimpleBigramLanguageModel(vocab_size, n_embed, block_size)
model = BigramLanguageModel(vocab_size, n_embed, block_size, n_layer, n_heads, dropout)

# Setup training components
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
evaluator = Evaluator(model, train_loader, val_loader)
trainer = Trainer(model, optimizer, train_loader, evaluator, max_iters, eval_interval)

# Train the model
final_losses = trainer.train()

# Generate some text
context = torch.zeros((1, 1), dtype=torch.long)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print("\nGenerated text:")
print(generated_text)
