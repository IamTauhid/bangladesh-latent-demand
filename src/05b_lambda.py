"""
Stage 5b - sensitivity of the DCG to the latent-inflation penalty lambda, and to the
size of the uncensored anchor set.

Experiment 1 gave a mixed result: with a 1670-day training window the censoring
correction removes 3.6-4.2 pp of bias, but with a 946-day window it OVER-corrects by
2.8-4.1 pp. The censored term in the Tobit likelihood saturates as mu -> inf, so mu is
anchored only by exact observations; when that anchor is thin the penalty lambda, not
the data, determines how far mu inflates. This script quantifies that dependence.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
torch.set_num_threads(4)
torch.manual_seed(0); np.random.seed(0)

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
NONLIN = ['CDD_at','CDD','HDD','T2M','T2M_MAX','RH2M','THI','apparent_temp','CDD_ma7','CDD_lag1',
          'T_ma3','dT','rain7','WS2M','ALLSKY_SFC_SW_DWN','sin1','cos1','sin2','cos2','sin3','cos3',
          'is_friday','is_saturday','is_holiday','is_ramadan','eid_prox','covid_lockdown','july_unrest']
TREND = ['t']
SQRT2 = float(np.sqrt(2.0))

class DCG(nn.Module):
    def __init__(self, p_nl, p_tr, h=96):
        super().__init__()
        self.lin = nn.Linear(p_tr, 1)
        self.body = nn.Sequential(nn.Linear(p_nl, h), nn.SiLU(), nn.Dropout(0.10),
                                  nn.Linear(h, h//2), nn.SiLU())
        self.mu = nn.Linear(h//2, 1); self.ls = nn.Linear(h//2, 1)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias, -3.0)
    def forward(self, xnl, xtr):
        z = self.body(xnl)
        return self.lin(xtr).squeeze(-1)+self.mu(z).squeeze(-1), self.ls(z).squeeze(-1).clamp(-7, 1)

def cnll(mu, ls, y, c, ridge):
    sig = ls.exp(); z = (y-mu)/sig
    le = -ls-0.5*np.log(2*np.pi)-0.5*z**2
    lc = torch.log(torch.clamp(0.5*torch.erfc(z/SQRT2), min=1e-10))
    loss = -(((1-c)*le+c*lc).mean())
    if ridge > 0: loss = loss + ridge*(torch.relu(mu-y)**2*c).mean()
    return loss

def fit(xnl, xtr, y, c, ridge, censored=True, epochs=900, seed=0):
    torch.manual_seed(seed); m = DCG(xnl.shape[1], xtr.shape[1])
    with torch.no_grad(): m.lin.bias.fill_(float(y.mean())); m.lin.weight.fill_(0.)
    opt = torch.optim.AdamW(m.parameters(), lr=8e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cu = c if censored else torch.zeros_like(c)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        mu, ls = m(xnl, xtr)
        cnll(mu, ls, y, cu, ridge if censored else 0.0).backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval(); return m

def prep(tr, te):
    a, b = tr[NONLIN].values.astype('float32'), te[NONLIN].values.astype('float32')
    c, d = tr[TREND].values.astype('float32'),  te[TREND].values.astype('float32')
    m1, s1 = a.mean(0), a.std(0)+1e-8; m2, s2 = c.mean(0), c.std(0)+1e-8
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    return T((a-m1)/s1), T((b-m1)/s1), T((c-m2)/s2), T((d-m2)/s2)

pre  = X[(X.index >= '2016-04-23') & (X.index < '2022-01-01')].copy()
post = X[(X.index >= '2022-01-01') & (X.index < '2025-01-01')]
ratio = (post.ls_mwh/(post.gen_mwh+post.ls_mwh)); ratio = ratio[ratio > 0]
r_mean, r_std, p_shed = ratio.mean(), ratio.std(), post.censored.mean()
T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
ADMIT = 0.60
LAMBDAS = [0.0, 1.0, 5.0, 20.0, 100.0]

print("="*94)
print("LAMBDA / ANCHOR-SIZE SENSITIVITY  (admit=0.60, 3 seeds)")
print("="*94)
print(f"P(shed)={p_shed:.3f}  curtailment depth mean={r_mean:.4f} sd={r_std:.4f}")
print(f"true mean suppression imposed = {p_shed*r_mean*(1-ADMIT)*100:.2f}% of served energy")

rows = []
for cut in ['2019-01-01', '2020-01-01', '2021-01-01']:
    tr_m = np.asarray(pre.index < cut)
    te_m = np.asarray((pre.index >= cut) & (pre.index < pd.Timestamp(cut)+pd.DateOffset(years=1)))
    xnl_tr, xnl_te, xtr_tr, xtr_te = prep(pre[tr_m], pre[te_m])
    n_anchor = int((1-p_shed)*tr_m.sum())
    print(f"\n--- train n={tr_m.sum()} (approx {n_anchor} uncensored anchor days), "
          f"holdout {cut[:4]}, test n={te_m.sum()} ---")
    for lam in LAMBDAS:
        acc = []
        for seed in range(3):
            rng = np.random.default_rng(200+seed); n = len(pre)
            shed = rng.random(n) < p_shed
            depth = np.clip(rng.normal(r_mean, r_std, n), 0.002, 0.35)*shed
            true_mwh = pre.gen_mwh.values
            obs_mwh = true_mwh*(1-depth) + true_mwh*depth*ADMIT
            y_obs, y_true, c_obs = np.log(obs_mwh), np.log(true_mwh), shed.astype('float32')
            m = fit(xnl_tr, xtr_tr, T(y_obs[tr_m]), T(c_obs[tr_m]), lam, seed=seed)
            with torch.no_grad(): mu, _ = m(xnl_te, xtr_te)
            p_, t_ = np.exp(mu.numpy()), np.exp(y_true[te_m])
            acc.append(((p_-t_).mean()/t_.mean()*100, np.mean(np.abs(p_-t_)/t_)*100))
        a = np.array(acc)
        rows.append(dict(holdout=cut[:4], train_n=int(tr_m.sum()), anchor_n=n_anchor,
                         lam=lam, bias=a[:,0].mean(), bias_sd=a[:,0].std(), mape=a[:,1].mean()))
        print(f"  lambda={lam:6.1f}   bias={a[:,0].mean():+6.2f} +-{a[:,0].std():.2f}%   MAPE={a[:,1].mean():5.2f}%")
    # censoring-blind reference for this split
    acc = []
    for seed in range(3):
        rng = np.random.default_rng(200+seed); n = len(pre)
        shed = rng.random(n) < p_shed
        depth = np.clip(rng.normal(r_mean, r_std, n), 0.002, 0.35)*shed
        true_mwh = pre.gen_mwh.values
        obs_mwh = true_mwh*(1-depth) + true_mwh*depth*ADMIT
        y_obs, y_true, c_obs = np.log(obs_mwh), np.log(true_mwh), shed.astype('float32')
        m = fit(xnl_tr, xtr_tr, T(y_obs[tr_m]), T(c_obs[tr_m]), 0.0, censored=False, seed=seed)
        with torch.no_grad(): mu, _ = m(xnl_te, xtr_te)
        p_, t_ = np.exp(mu.numpy()), np.exp(y_true[te_m])
        acc.append(((p_-t_).mean()/t_.mean()*100, np.mean(np.abs(p_-t_)/t_)*100))
    a = np.array(acc)
    rows.append(dict(holdout=cut[:4], train_n=int(tr_m.sum()), anchor_n=n_anchor,
                     lam='blind', bias=a[:,0].mean(), bias_sd=a[:,0].std(), mape=a[:,1].mean()))
    print(f"  censoring-blind  bias={a[:,0].mean():+6.2f} +-{a[:,0].std():.2f}%   MAPE={a[:,1].mean():5.2f}%")

pd.DataFrame(rows).round(3).to_csv('results/exp1b_lambda.csv', index=False)
print("\nsaved -> results/exp1b_lambda.csv")
