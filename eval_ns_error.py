#!/usr/bin/env python3
"""
3-D Navier–Stokes ABC-flow error evaluation (u, v, w only)

• Works with MOL-PINN checkpoints that output (u,v,w) OR (u,v,w,p).
  If a 4-channel checkpoint is given, the 4th channel is ignored.
• Produces |error| heat-maps on the x–t plane at y = 0.5, z = 0.5.
• Each title shows slice-MSE and full-field MSE; global MSE is printed once.

Usage
-----
python eval_ns_error.py --ckpt PATH/your_model.pth
"""

from __future__ import annotations
import argparse, math, re
from pathlib import Path
import torch, torch.nn as nn
import matplotlib.pyplot as plt

PI = math.pi

# ── CLI ───────────────────────────────────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument('--ckpt',  required=True, help='model .pth with state_dict')
pa.add_argument('--nu',    type=float, default=0.01)
pa.add_argument('--res',   type=int,   default=200)   # heat-map resolution
pa.add_argument('--n_sys', type=int)                 # override N_SYS
args = pa.parse_args()

ck     = Path(args.ckpt)
ckpt   = torch.load(ck, map_location='cpu')
state  = ckpt['state_dict']
IS_VEC = any(k.startswith('mlp.') for k in state)
N_SYS  = args.n_sys or ckpt.get('meta', {}).get('n_sys', 21)

# ── analytic ABC flow (A = B = C = 1) ─────────────────────────────────────
def abc_exact(x, y, z, t):
    u = torch.sin(2*PI*z) + torch.cos(2*PI*y)
    v = torch.sin(2*PI*x) + torch.cos(2*PI*z)
    w = torch.sin(2*PI*y) + torch.cos(2*PI*x)
    return torch.stack((u, v, w), -1)          # (…,3)

# ── rebuild network ───────────────────────────────────────────────────────
def build(sd):
    layers, in_f = [], None
    for k, w in [(k, w) for k, w in sd.items() if k.endswith('.weight')]:
        in_f = w.shape[1] if in_f is None else in_f
        layers += [nn.Linear(in_f, w.shape[0]), nn.SiLU()]
        in_f = w.shape[0]
    layers.pop()
    net = nn.Sequential(*layers)
    net.load_state_dict(sd); net.eval()
    return net

if IS_VEC:
    inner = {re.sub(r'^mlp\.', '', k): v for k, v in state.items()
             if k.startswith('mlp.')}
    mlp = build(inner)
    x_grid = torch.linspace(0, 1, N_SYS)

    def net_pred(x, y, z, t):
        B   = x.shape[0]
        idx = (x * (N_SYS-1)).clamp(0, N_SYS-1-1e-6)
        i0  = idx.floor().long()
        i1  = (i0+1).clamp_max(N_SYS-1)
        wgt = (idx - i0).unsqueeze(1)

        yrep = y.unsqueeze(1).expand(-1, N_SYS)
        zrep = z.unsqueeze(1).expand(-1, N_SYS)
        trep = t.unsqueeze(1).expand_as(yrep)
        out  = mlp(torch.stack((x_grid.expand_as(yrep), yrep, zrep, trep), -1))

        z0, z1 = out[torch.arange(B), i0], out[torch.arange(B), i1]
        return (1-wgt)*z0 + wgt*z1          # (B, n_channels)
else:
    inner = {re.sub(r'^[^.]+\.', '', k): v for k, v in state.items()}
    mlp   = build(inner)
    def net_pred(x, y, z, t):
        return mlp(torch.stack((x, y, z, t), -1))

# ── grids ────────────────────────────────────────────────────────────────
R  = args.res
xv = torch.linspace(0, 1, R)
tv = torch.linspace(0, 1, R)
fields = {'u':0, 'v':1, 'w':2}

# ── global MSE on coarse grid ────────────────────────────────────────────
G = 64
g1 = torch.linspace(0, 1, G)
Xg, Yg, Zg, Tg = torch.meshgrid(g1, g1, g1, g1, indexing='ij')
coords = torch.stack((Xg.flatten(), Yg.flatten(), Zg.flatten(), Tg.flatten()), 1)

pred_chunks = []
BATCH = 4096
with torch.no_grad():
    for i in range(0, coords.shape[0], BATCH):      # ← coords.shape[0]
        xb, yb, zb, tb = coords[i:i+BATCH].T
        pred_chunks.append(
            net_pred(xb, yb, zb, tb)[:, :3]          # keep u,v,w only
        )

pred_all  = torch.cat(pred_chunks)                  # (N,3)
exact_all = abc_exact(Xg, Yg, Zg, Tg).view(-1, 3)
global_mse = (pred_all - exact_all).pow(2).mean().item()
print(f"Global MSE (u,v,w) = {global_mse:.3e}")

# ── heat-maps at y = z = 0.5 ─────────────────────────────────────────────
y_const = torch.full((R,), 0.5)
z_const = torch.full((R,), 0.5)

with torch.no_grad():
    for name, ch in fields.items():
        err_mat = torch.empty(R, R)                 # x rows, t cols
        for j, tval in enumerate(tv):
            pred  = net_pred(xv, y_const, z_const,
                             torch.full_like(xv, tval))[:, ch]
            exact = abc_exact(xv, y_const, z_const,
                               torch.full_like(xv, tval))[:, ch]
            err_mat[:, j] = (pred - exact).abs()

        mse_slice = err_mat.pow(2).mean().item()
        mse_field = (pred_all[:, ch] - exact_all[:, ch]).pow(2).mean().item()

        plt.figure(figsize=(6, 4))
        plt.pcolormesh(xv, tv, err_mat.T, shading='auto', cmap='viridis')
        plt.colorbar(label='|error|')
        plt.xlabel('x');  plt.ylabel('t')
        plt.title(f"{ck.name}  {name}  (y=0.5, z=0.5)\n"
                  f"MSE_slice={mse_slice:.2e}  |  MSE_{name}={mse_field:.2e}")
        plt.tight_layout()

        out = ck.with_name(f"abs_err_{name}_y0.5_z0.5_{ck.stem}.png")
        plt.savefig(out, dpi=200); plt.close()
        print(f"Saved → {out.name}   slice={mse_slice:.2e}  field={mse_field:.2e}")
