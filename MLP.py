import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s : i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i : s for s, i in stoi.items()}

#Build dataset
def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        chs = w + '.'
        for ch in chs:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y
        
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1]) # Training set
Xdev, Ydev = build_dataset(words[n1:n2]) # Dev set
Xte, Yte = build_dataset(words[n2:]) # Test set

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)
w1 = torch.randn((30, 300), generator=g)
b1 = torch.randn(300, generator=g)
w2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, w1, b1, w2, b2]
for p in parameters:
    p.requires_grad = True

# Training loop
lossi = []
stepi = []
for _ in range(200000):
    # Create minibatch
    i = torch.randint(0, Xtr.shape[0], (32,))
    # Forward pass
    emb = C[Xtr[i]]
    h = torch.tanh(emb.view(-1, 30) @ w1 + b1)
    logits = h @ w2 + b2
    # counts = logits.exp()
    # probs = counts / counts.sum(1, keepdims=True)
    # loss = -probs[torch.arange(counts.shape[0]), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[i]) #same as the above 3 lines code, but more efficient
    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # Update
    lr = 0.1 if _ < 100000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad
    # Track stats
    stepi.append(_)
    lossi.append(loss.log10().item())
plt.plot(stepi, lossi)
plt.show()

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ w1 + b1)
logits = h @ w2 + b2
loss = F.cross_entropy(logits, Ytr)
print(loss.item())

# Evaluate
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ w1 + b1)
logits = h @ w2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

# Sampling
g = torch.Generator().manual_seed(2147483647 + 10)
for i in range(20):
    out = []
    context = [0] * 3
    while True:
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(-1, 30) @ w1 + b1)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        context = context[1:] + [ix]
        if ix == 0:
            break
    print(''.join(out))