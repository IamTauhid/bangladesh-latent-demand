"""
Stage 19 - extend the synthetic recovery sweep to the magnitudes actually reported.

Referee point: the original sweep spanned true suppression of 0.00-1.23 pp, but the
application reports 3-10 pp. Calibration was therefore never tested near the values
being claimed. Here the curtailment depth is scaled so that true suppression spans
roughly 0-5 pp, and the admission rate is swept more finely. The question is whether
the anti-correlation between the applied correction and the truth persists at the
magnitudes that matter, or is an artefact of the narrow original range.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
NONLIN = ['CDD_at','CDD','HDD','T2M','T2M_MAX','RH2M','THI','apparent_temp','CDD_ma7',
          'CDD_lag1','T_ma3','dT','rain7','WS2M','ALLSKY_SFC_SW_DWN',
          'sin1','cos1','sin2','cos2','sin3','cos3',
          'is_friday','is_saturday','is_holiday','is_ramadan','eid_prox',
          'covid_lockdown','july_unrest']
TREND = ['t']
SQRT2 = float(np.sqrt(2.0))
ALPHA_MIN = 0.30

class DCG(nn.Module):
    def __init__(self, p_nl, p_tr, h=96):
        super().__init__()
        self.lin = nn.Linear(p_tr, 1)
        self.body = nn.Sequential(nn.Linear(p_nl, h), nn.SiLU(), nn.Dropout(0.10),
                                  nn.Linear(h, h//2), nn.SiLU())
        self.mu = nn.Linear(h//2, 1); self.ls = nn.Linear(h//2, 1)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias, -3.0)
    def forward(self, xn, xt):
        z = self.body(xn)
        return self.lin(xt).squeeze(-1)+self.mu(z).squeeze(-1), self.ls(z).squeeze(-1).clamp(-4.0, 1.0)

def _Phi(z): return 0.5*torch.erfc(-z/SQRT2)

def nll(mu, ls, lo, hi, c, mode):
    sig = ls.exp(); zl = (lo-mu)/sig
    exact = -ls - 0.5*np.log(2*np.pi) - 0.5*zl**2
    if mode == 'blind':
        return -(exact.mean())
    if mode == 'tobit':
        cens = torch.log(torch.clamp(0.5*torch.erfc(zl/SQRT2), min=1e-10))
        return -(((1-c)*exact + c*cens).mean()) + 5.0*(torch.relu(mu-lo)**2*c).mean()
    zh = (hi-mu)/sig
    cens = torch.log(torch.clamp(_Phi(zh)-_Phi(zl), min=1e-10))
    return -(((1-c)*exact + c*cens).mean())

def fit(xn, xt, lo, hi, c, mode, seed, epochs=800):
    torch.manual_seed(seed); m = DCG(xn.shape[1], xt.shape[1])
    with torch.no_grad(): m.lin.bias.fill_(float(lo.mean())); m.lin.weight.fill_(0.)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        mu, ls = m(xn, xt); nll(mu, ls, lo, hi, c, mode).backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval(); return m

T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
pre  = X[(X.index >= '2016-04-23') & (X.index < '2022-01-01')].copy()
post = X[(X.index >= '2022-01-01') & (X.index < '2025-01-01')]
ratio = (post.ls_mwh/(post.gen_mwh+post.ls_mwh)); ratio = ratio[ratio > 0]
r_mean, r_std, p_shed = ratio.mean(), ratio.std(), post.censored.mean()

tr_m = np.asarray(pre.index < '2021-01-01')
te_m = np.asarray(pre.index >= '2021-01-01')
a_nl, b_nl = pre[NONLIN].values[tr_m].astype('float32'), pre[NONLIN].values[te_m].astype('float32')
a_tr, b_tr = pre[TREND].values[tr_m].astype('float32'), pre[TREND].values[te_m].astype('float32')
m1, s1 = a_nl.mean(0), a_nl.std(0)+1e-8; m2, s2 = a_tr.mean(0), a_tr.std(0)+1e-8
xn_tr, xn_te = T((a_nl-m1)/s1), T((b_nl-m1)/s1)
xt_tr, xt_te = T((a_tr-m2)/s2), T((b_tr-m2)/s2)

print("="*94)
print("EXTENDED SWEEP: does the anti-correlation persist at 3-5 pp true suppression?")
print("="*94)
rows = []
for mult in (1.0, 2.0, 3.0):
    for admit in (1.00, 0.60, 0.40, 0.20):
        true_supp = p_shed*r_mean*mult*(1-admit)*100
        acc = {k: [] for k in ('blind', 'tobit', 'interval')}
        for seed in range(3):
            rng = np.random.default_rng(500+seed); n = len(pre)
            shed = rng.random(n) < p_shed
            depth = np.clip(rng.normal(r_mean*mult, r_std*mult, n), 0.002, 0.60)*shed
            true = pre.gen_mwh.values
            G = true*(1-depth); Sad = true*depth*admit
            lo = np.log(G + Sad)
            hi = np.where(shed, np.log(G + Sad/ALPHA_MIN), lo)
            c = shed.astype('float32'); ytrue = np.log(true)
            for mode in acc:
                m = fit(xn_tr, xt_tr, T(lo[tr_m]), T(hi[tr_m]), T(c[tr_m]), mode, seed)
                with torch.no_grad(): mu, _ = m(xn_te, xt_te)
                p_, t_ = np.exp(np.clip(mu.numpy(), 10.5, 13.5)), np.exp(ytrue[te_m])
                acc[mode].append((p_-t_).mean()/t_.mean()*100)
        b = np.mean(acc['blind'])
        ct, ci = np.mean(acc['tobit'])-b, np.mean(acc['interval'])-b
        rows.append(dict(depth_mult=mult, admit=admit, true_supp=true_supp,
                         tobit_corr=ct, icg_corr=ci))
        print(f"  depth x{mult:.0f}, admit {admit:.2f} | TRUE {true_supp:5.2f} pp | "
              f"Tobit {ct:+6.2f} | ICG {ci:+6.2f}")

D = pd.DataFrame(rows)
ct_r = np.corrcoef(D.true_supp, D.tobit_corr)[0, 1]
ci_r = np.corrcoef(D.true_supp, D.icg_corr)[0, 1]
print(f"\nAcross all {len(D)} configurations (true suppression "
      f"{D.true_supp.min():.2f}-{D.true_supp.max():.2f} pp):")
print(f"  corr(true, Tobit correction) = {ct_r:+.3f}")
print(f"  corr(true, ICG   correction) = {ci_r:+.3f}")
print(f"  mean |calibration error|: Tobit {np.abs(D.tobit_corr-D.true_supp).mean():.2f} pp, "
      f"ICG {np.abs(D.icg_corr-D.true_supp).mean():.2f} pp")

D.round(3).to_csv('results/exp1d_magnitude_sweep.csv', index=False)
json.dump({'corr_true_tobit': float(ct_r), 'corr_true_icg': float(ci_r),
           'true_range_pp': [float(D.true_supp.min()), float(D.true_supp.max())],
           'mae_tobit_pp': float(np.abs(D.tobit_corr-D.true_supp).mean()),
           'mae_icg_pp': float(np.abs(D.icg_corr-D.true_supp).mean())},
          open('results/exp1d_summary.json', 'w'), indent=2)
print("\nsaved -> results/exp1d_magnitude_sweep.csv, exp1d_summary.json")
