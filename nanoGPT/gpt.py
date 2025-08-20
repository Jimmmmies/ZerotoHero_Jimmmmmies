import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
batch_size = 4
block_size = 8

with open("input.txt", "r") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create encoder & decoder
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Initial dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
