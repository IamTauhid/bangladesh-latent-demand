"""
Stage 16 - sensitivity of the recovered latent demand to alpha_min.

alpha_min is the only free prior in the interval-censored model: it is the assumed
lower bound on the share of the true shortfall the operator admits. alpha_min = 1
recovers the censoring-blind model exactly; smaller values widen the interval and
permit a larger correction. A reviewer will ask how the conclusions move with it,
so we sweep it and report the recovered suppression per year for each value.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
torch.set_num_threads(4)
torch.manual_seed(0); np.random.seed(0)

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
NONLIN = ['CDD_at','CDD','HDD','T2M','T2M_MAX','RH2M','THI','apparent_temp','CDD_ma7',
          'CDD_lag1','T_ma3','dT','rain7','WS2M','ALLSKY_SFC_SW_DWN',
          'sin1','cos1','sin2','cos2','sin3','cos3',
          'is_friday','is_saturday','is_holiday','is_ramadan','eid_prox',
          'covid_lockdown','july_unrest']
SQRT2 = float(np.sqrt(2.0))
ALPHAS = [0.20, 0.30, 0.50, 0.70, 1.00]

class DCG(nn.Module):
    def __init__(self, p_nl, h=96):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.body = nn.Sequential(nn.Linear(p_nl, h), nn.SiLU(), nn.Dropout(0.10),
                                  nn.Linear(h, h//2), nn.SiLU())
        self.mu = nn.Linear(h//2, 1); self.ls = nn.Linear(h//2, 1)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias, -3.0)
    def forward(self, xnl, xtr):
        z = self.body(xnl)
        return self.lin(xtr).squeeze(-1)+self.mu(z).squeeze(-1), self.ls(z).squeeze(-1).clamp(-7, 1)

def _Phi(z): return 0.5*torch.erfc(-z/SQRT2)

def nll(mu, ls, lo, hi, c):
    sig = ls.exp()
    zl = (lo-mu)/sig
    exact = -ls - 0.5*np.log(2*np.pi) - 0.5*zl**2
    zh = (hi-mu)/sig
    cens = torch.log(torch.clamp(_Phi(zh)-_Phi(zl), min=1e-10))
    return -(((1-c)*exact + c*cens).mean())

Xnl = X[NONLIN].values.astype('float32'); Xtr = X[['t']].values.astype('float32')
mn, sn = Xnl.mean(0), Xnl.std(0)+1e-8; mt, st = Xtr.mean(0), Xtr.std(0)+1e-8
served = X.gen_mwh.values.astype('float64')
adm    = X.ls_mwh.values.astype('float64')
c_np   = X.censored.values.astype('float32')
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
xn, xt, cc = T((Xnl-mn)/sn), T((Xtr-mt)/st), T(c_np)

def fit(lo_np, hi_np, seed):
    torch.manual_seed(seed); m = DCG(len(NONLIN))
    with torch.no_grad():
        m.lin.bias.fill_(float(lo_np.mean())); m.lin.weight.fill_(0.)
    opt = torch.optim.AdamW(m.parameters(), lr=8e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1000)
    lo_t, hi_t = T(lo_np), T(hi_np)
    for _ in range(1000):
        m.train(); opt.zero_grad()
        mu, ls = m(xn, xt)
        nll(mu, ls, lo_t, hi_t, cc).backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): return m(xn, xt)[0].numpy()

rows = {}
print("="*84)
print("SENSITIVITY OF RECOVERED SUPPRESSION TO alpha_min  (3 seeds each)")
print("="*84)
for a in ALPHAS:
    lo = np.log(served + adm).astype('float32')
    hi = np.log(served + np.where(c_np > 0, adm/a, 0.0)).astype('float32')
    hi = np.maximum(hi, lo)
    lat = np.exp(np.mean([fit(lo, hi, s) for s in range(3)], 0))
    d = pd.DataFrame({'served': served, 'latent': lat}, index=X.index)
    yr = d.groupby(d.index.year).sum()
    supp = (yr.latent/yr.served - 1)*100
    unc = [2017, 2018, 2019, 2020, 2021]
    floor = supp.loc[unc].std()
    rows[a] = {int(k): float(v) for k, v in supp.items()}
    rows[a]['noise_floor_sd'] = float(floor)
    sig23 = (supp.loc[2023]-supp.loc[unc].mean())/floor
    sig24 = (supp.loc[2024]-supp.loc[unc].mean())/floor
    print(f"alpha_min={a:.2f} | 2022 {supp.loc[2022]:+.2f} | 2023 {supp.loc[2023]:+.2f} "
          f"({sig23:+.1f} sd) | 2024 {supp.loc[2024]:+.2f} ({sig24:+.1f} sd) | "
          f"2025 {supp.loc[2025]:+.2f} | floor sd {floor:.2f}")

json.dump(rows, open('results/alpha_sensitivity.json', 'w'), indent=2)
pd.DataFrame(rows).T.round(3).to_csv('results/alpha_sensitivity.csv')
print("\nsaved -> results/alpha_sensitivity.{json,csv}")
