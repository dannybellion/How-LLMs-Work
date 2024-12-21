import torch
#from utils.simple_bigram import BigramLanguageModel
from utils.bigram_dropout import BigramLanguageModel
from utils.utils import BatchLoader, Evaluator

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel
block_size = 256 # what is the maximum context length for predictions
max_iters = 1000 # number of training iterations
eval_interval = 500 # how often to evaluate the model
learning_rate = 3e-4 # learning rate
eval_iters = 200 # number of iterations to evaluate loss
n_embed = 384 # embedding dimension
n_heads = 6
n_layer = 6
dropout = 0.2


with open("../data/input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

def create_encoders(chars):
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

encode, decode = create_encoders(chars)

data = torch.tensor(encode(text))

# split into train and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# create a dataset
torch.manual_seed(1337)

# Setup data loaders
train_loader = BatchLoader(train_data, block_size=block_size, batch_size=batch_size)
val_loader = BatchLoader(val_data, block_size=block_size, batch_size=batch_size)

model = BigramLanguageModel(vocab_size, n_embed, block_size, n_layer, n_heads, dropout)

# Setup evaluator
evaluator = Evaluator(model, train_loader, val_loader)

# Use in training loop
losses = evaluator.estimate_loss(eval_iters)

# print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# now we train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    xb, yb = train_loader()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        losses = evaluator.estimate_loss(eval_iters)
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
