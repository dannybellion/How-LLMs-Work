import torch

with open("../data/input.txt", "r") as f:
    text = f.read()

print(text[:100])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(' '.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("ABCD"))
print(decode([13,14,15,16]))

data = torch.tensor(encode(text))
print(data.shape)
print(data[:100])

# split into train and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data.shape)
print(val_data.shape)

block_size = 8

# 9 items will have 8 predicion examples
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]

# useful so the transformer is used to seeing different lengths of data
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")