import torch
import torch.nn.functional as F

words = open('/Users/jimmmmmies/Desktop/Coding/Py/makemore_Jimmy/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s : i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i : s for s, i in stoi.items()}

#create dataset
xs = []
ys = []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for chs1, chs2 in zip(chs, chs[1:]):
        xs.append(stoi[chs1])
        ys.append(stoi[chs2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

#initialize network
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for i in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True) + 0.01*(W**2).mean()
    loss = -probs[torch.arange(num), ys].log().mean()
    
    #back propagation
    W.grad = None
    loss.backward()
    W.data -= 50 * W.grad
    
#sampling
g = torch.Generator().manual_seed(2147483647)
for i in range(100):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))