"""通道筛选 - 超简版"""
import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib; import matplotlib.pyplot as plt
matplotlib.use('Agg')

DROOT, RDIR, GPATH = "./data", "./results/channel_selection", "./good_subjects.json"
FS, DEV = 2000, torch.device("cpu")
CCOUNTS, NSUBJ = [12, 8, 4], 3

os.makedirs(RDIR, exist_ok=True)

class LSTM(nn.Module):
    def __init__(self, inp=12): 
        super().__init__()
        self.l = nn.LSTM(inp, 32, 1, batch_first=True)
        self.f = nn.Linear(32, 18)
    def forward(self, x):
        o, _ = self.l(x)
        return self.f(o[:, -1, :])

class DS(torch.utils.data.Dataset):
    def __init__(self, sid):
        wl = int(300 * FS / 1000)
        st = int(150 * FS / 1000)
        self.d = np.load(f"{DROOT}/S{sid}_data.npy")
        self.l = np.load(f"{DROOT}/S{sid}_label.npy")
        self.d = (self.d - self.d.mean(0)) / (self.d.std(0) + 1e-6)
        self.ns = (len(self.d) - wl) // st + 1
    def __len__(self): return min(self.ns, 100)
    def __getitem__(self, i):
        s, e = i * int(150*FS/1000), i * int(150*FS/1000) + int(300*FS/1000)
        x = torch.from_numpy(self.d[s:e, :]).float()
        y = int(np.bincount(self.l[s:e]).argmax()) if e < len(self.l) else 0
        return x, torch.tensor(y)

class FDS(torch.utils.data.Dataset):
    def __init__(self, ds, ch): self.ds, self.ch = ds, ch
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        return x[:, self.ch], y

def run(nc, sid, ch=None):
    ds = DS(sid)
    if ch: ds = FDS(ds, ch)
    tl, vl = int(0.7*len(ds)), len(ds)-int(0.7*len(ds))
    trd, vdd = random_split(ds, [tl, vl])
    tl, vl = DataLoader(trd, 32, True), DataLoader(vdd, 32)
    m = LSTM(nc).to(DEV)
    for _ in range(2):
        for x, y in tl:
            o = m(x); loss = nn.CrossEntropyLoss()(o, y.to(DEV).long())
            loss.backward(); torch.optim.Adam(m.parameters()).step()
    c, t = 0, 0
    with torch.no_grad():
        for x, y in vl:
            p = m(x).argmax(1); c += (p == y.to(DEV)).sum().item(); t += y.size(0)
    return c/t

with open(GPATH) as f: subs = json.load(f)[:NSUBJ]
print("通道筛选")
R = {c: [] for c in CCOUNTS}
for c in CCOUNTS:
    print(f"=== {c}通道 ===")
    for s in subs:
        np.random.seed(s*100+c)
        ch = sorted(np.random.choice(12, c, replace=False).tolist()) if c<12 else None
        a = run(c, s, ch); R[c].append(a); print(f"S{s}: {a:.3f}")
    print(f"M: {np.mean(R[c]):.3f}")

print("\n=== 结果 ===")
for c in CCOUNTS: print(f"{c}: {np.mean(R[c]):.3f}")

with open(f"{RDIR}/results.json", 'w') as f: json.dump(R, f, indent=2)
plt.figure(); plt.plot(CCOUNTS, [np.mean(R[c]) for c in CCOUNTS], 'o-'); plt.xlabel('Channels'); plt.ylabel('Accuracy'); plt.title('Channel Selection'); plt.ylim(0,1); plt.grid(True); plt.savefig(f"{RDIR}/pc.png", dpi=300); print("OK")
