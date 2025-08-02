#!/usr/bin/env python3
"""
MOL-PINN for 3-D incompressible Navier–Stokes (ABC flow)
Outputs: u, v, w   (no pressure)
"""

from __future__ import annotations
import argparse, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

# ─── CLI ──────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('--n_sys',     type=int,   default=21)
ap.add_argument('--epochs',    type=int,   default=1)
ap.add_argument('--lr',        type=float, default=2e-3)
ap.add_argument('--batch_yzt', type=int,   default=128)
ap.add_argument('--nu',        type=float, default=0.01)
ap.add_argument('--lambda_bc', type=float, default=100.0)
ap.add_argument('--lambda_div',type=float, default=1.0)
ap.add_argument('--bc_dir',    type=str,   default='abc_bcs')
ap.add_argument('--out_dir',   type=str,   default='abc_run')
args = ap.parse_args()

N_SYS, NU = args.n_sys, args.nu
OUT = Path(args.out_dir); OUT.mkdir(exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)

# ─── boundary tiny nets  (u,v,w only) ─────────────────────────────────────
class BCNet(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        mods=[nn.Linear(3,hidden), nn.Tanh()]
        for _ in range(layers-1):
            mods += [nn.Linear(hidden,hidden), nn.Tanh()]
        mods.append(nn.Linear(hidden,1))
        self.net = nn.Sequential(*mods)
    def forward(self, xyz): return self.net(xyz)

def load_bc(var:str,t_mark:int):
    fn = Path(args.bc_dir)/f'{var}_t{t_mark}.pth'
    if not fn.exists(): raise FileNotFoundError(fn)
    net = BCNet().to(DEVICE)
    net.load_state_dict(torch.load(fn,map_location=DEVICE)['state_dict'])
    net.eval(); return net

t_slices = [0,1]
bc_nets  = {(v,t):load_bc(v,t) for t in t_slices for v in ('u','v','w')}

# ─── periodic grid in x ───────────────────────────────────────────────────
x_grid = torch.linspace(0,1,N_SYS,device=DEVICE)
dx = (x_grid[1]-x_grid[0]).item()

# ─── main network (3 outputs) ─────────────────────────────────────────────
class VecMOLPINN(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.register_buffer('x_code',torch.linspace(0,1,n).unsqueeze(0))
        self.mlp = nn.Sequential(
            nn.Linear(4,128),nn.SiLU(),
            nn.Linear(128,128),nn.SiLU(),
            nn.Linear(128,128),nn.SiLU(),
            nn.Linear(128,3))                # u,v,w
        self.n=n
    def forward(self,y,z,t):
        B=y.size(0)
        xr=self.x_code.expand(B,-1)
        inp=torch.stack((xr,
                         y.unsqueeze(1).expand_as(xr),
                         z.unsqueeze(1).expand_as(xr),
                         t.unsqueeze(1).expand_as(xr)),-1).view(-1,4)
        return self.mlp(inp).view(B,self.n,3)

model = VecMOLPINN(N_SYS).to(DEVICE)
opt   = optim.Adam(model.parameters(),lr=args.lr)
mse   = nn.MSELoss()

# ─── FD helpers & safe grad ───────────────────────────────────────────────
def fd_first(z):
    zp=torch.cat([z[:,1:2],z,z[:,-2:-1]],1)
    return (zp[:,2:]-zp[:,:-2])/(2*dx)
def fd_second(z):
    zp=torch.cat([z[:,1:2],z,z[:,-2:-1]],1)
    return (zp[:,:-2]-2*z+zp[:,2:])/dx**2
def safe_grad(out,inp,cg=True):
    g=torch.autograd.grad(out,inp,torch.ones_like(inp),
                          retain_graph=True,create_graph=cg,allow_unused=True)[0]
    return g if g is not None else torch.zeros_like(inp)

# ─── training loop ────────────────────────────────────────────────────────
loss_hist=[]; wall=0.
for ep in range(1,args.epochs+1):
    tic=time.time()
    batch=torch.rand(args.batch_yzt,3,device=DEVICE,requires_grad=True)
    yb,zb,tb=batch.T
    u,v,w = model(yb,zb,tb).unbind(-1)          # each (B,N_SYS)

    u_t = torch.stack([safe_grad(u[:,j],tb) for j in range(N_SYS)],1)
    v_t = torch.stack([safe_grad(v[:,j],tb) for j in range(N_SYS)],1)
    w_t = torch.stack([safe_grad(w[:,j],tb) for j in range(N_SYS)],1)

    zeros=torch.zeros_like(u)
    du_dy=dv_dy=dw_dy=zeros.clone()
    du_dz=dv_dz=dw_dz=zeros.clone()
    d2u_dy2=d2v_dy2=d2w_dy2=zeros.clone()
    d2u_dz2=d2v_dz2=d2w_dz2=zeros.clone()

    for j in range(N_SYS):
        du_dz[:,j]=safe_grad(u[:,j],zb); dv_dz[:,j]=safe_grad(v[:,j],zb); dw_dz[:,j]=safe_grad(w[:,j],zb)
        d2u_dz2[:,j]=safe_grad(du_dz[:,j],zb,False)
        d2v_dz2[:,j]=safe_grad(dv_dz[:,j],zb,False)
        d2w_dz2[:,j]=safe_grad(dw_dz[:,j],zb,False)

    for j in range(N_SYS):
        du_dy[:,j]=safe_grad(u[:,j],yb); dv_dy[:,j]=safe_grad(v[:,j],yb); dw_dy[:,j]=safe_grad(w[:,j],yb)
        d2u_dy2[:,j]=safe_grad(du_dy[:,j],yb,False)
        d2v_dy2[:,j]=safe_grad(dv_dy[:,j],yb,False)
        d2w_dy2[:,j]=safe_grad(dw_dy[:,j],yb,False)

    u_x,v_x,w_x = map(fd_first,(u,v,w))
    u_xx,v_xx,w_xx = map(fd_second,(u,v,w))

    lap_u=u_xx+d2u_dy2+d2u_dz2
    lap_v=v_xx+d2v_dy2+d2v_dz2
    lap_w=w_xx+d2w_dy2+d2w_dz2

    mom_x=u_t+u*u_x+v*du_dy+w*du_dz-NU*lap_u
    mom_y=v_t+u*v_x+v*dv_dy+w*dv_dz-NU*lap_v
    mom_z=w_t+u*w_x+v*dw_dy+w*dw_dz-NU*lap_w
    cont =u_x+dv_dy+dw_dz

    phy=(mom_x**2+mom_y**2+mom_z**2).mean()
    div=cont.pow(2).mean()

    # boundary anchors (u,v,w only)
    bc_loss=0.
    for ts in t_slices:
        tb=torch.full((args.batch_yzt,),float(ts),device=DEVICE)
        yz=torch.rand(args.batch_yzt,2,device=DEVICE); yb2,zb2=yz.T
        pu,pv,pw = model(yb2,zb2,tb).unbind(-1)
        xyz=torch.stack((x_grid.expand(args.batch_yzt,-1),
                         yb2.unsqueeze(1).expand(-1,N_SYS),
                         zb2.unsqueeze(1).expand(-1,N_SYS)),-1).view(-1,3)
        with torch.no_grad():
            bc_u=bc_nets[('u',ts)](xyz).view(args.batch_yzt,N_SYS)
            bc_v=bc_nets[('v',ts)](xyz).view(args.batch_yzt,N_SYS)
            bc_w=bc_nets[('w',ts)](xyz).view(args.batch_yzt,N_SYS)
        bc_loss += mse(pu,bc_u)+mse(pv,bc_v)+mse(pw,bc_w)

    total=phy+args.lambda_div*div+args.lambda_bc*bc_loss
    opt.zero_grad(set_to_none=True); total.backward(); opt.step()

    loss_hist.append(total.item())
    wall += time.time()-tic
    if ep%100==0 or ep==1:
        print(f"ep {ep:5d}/{args.epochs}  loss={total.item():.2e}  "
              f"phy={phy.item():.2e}  bc={bc_loss.item():.2e}")

# ─── save & loss curve ────────────────────────────────────────────────────
avg_t = wall/args.epochs
torch.save({'state_dict':model.state_dict(),
            'meta':{'n_sys':N_SYS,'nu':NU}}, OUT/'molpinn_abc_nop.pth')

plt.figure(); plt.semilogy(loss_hist,label=f"avg {avg_t:.3f} s/epoch")
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
plt.savefig(OUT/'loss_curve_nop.png',dpi=180); plt.close()

# ── heat-maps (u,v,w) at y=z=0.5 ────────────────────────────────────────
t_dense = torch.linspace(0, 1, 200, device=DEVICE)
fix_y   = torch.full((N_SYS,), 0.5, device=DEVICE)
fix_z   = torch.full((N_SYS,), 0.5, device=DEVICE)

for name, ch in {'u': 0, 'v': 1, 'w': 2}.items():
    U = torch.stack(
        [
            model(fix_y, fix_z, torch.full((N_SYS,), t, device=DEVICE))
                .detach()                     # ← detach here
                [:, :, ch].mean(0).cpu()
            for t in t_dense
        ],
        0
    )

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(x_grid.cpu(), t_dense.cpu(), U, shading='nearest')  # U is now grad-free
    plt.colorbar(label=f'{name}(x,t)  at y=z=0.5')
    plt.xlabel('x'); plt.ylabel('t')
    plt.title(f'NS MOL-PINN ({name})  N_SYS={N_SYS}')
    plt.tight_layout()
    plt.savefig(OUT / f'heatmap_{name}_ns_{N_SYS}.png', dpi=200)
    plt.close()
    print(f"Saved heatmap_{name}_ns_{N_SYS}.png")

print("✓ training & plots complete")
