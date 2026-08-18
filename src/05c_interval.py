"""
Stage 5c - Interval-Censored Gaussian (ICG), replacing the right-censored Tobit.

DIAGNOSIS.  The Tobit specification of Stage 5 treats a shed day only as
"D_t >= y_t". That indicator carries almost no information about HOW MUCH demand was
suppressed, and the censored log-likelihood term saturates as mu -> inf, so the fitted
latent level is set by the penalty rather than by the data. Stage 5b confirms this:
with a true imposed suppression of 0.82% of energy the Tobit recovers +3.7%, and the
result is insensitive to the penalty weight.

FIX.  The operator publishes the admitted shortfall S_t, so we know more than an
indicator. If the operator admits a fraction alpha of the true shortfall, then

    true demand  D_t = G_t + S_t/alpha ,    alpha in [alpha_min, 1]

which bounds D_t in a closed interval whose WIDTH IS SET BY THE REPORTED SHORTFALL:

    lo_t = log(G_t + S_t)              (alpha = 1: the operator admitted everything)
    hi_t = log(G_t + S_t/alpha_min)    (alpha = alpha_min: maximal under-reporting)

giving an interval-censored likelihood

    log[ Phi((hi_t - mu)/sigma) - Phi((lo_t - mu)/sigma) ] .

This is bounded above, needs no ad-hoc penalty, scales the correction with the
observed severity of rationing, and reduces to the exact-observation term as
S_t -> 0. alpha_min is the single interpretable prior: alpha_min = 1 recovers the
censoring-blind model, small alpha_min admits large under-reporting.
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

def _Phi(z): return 0.5*torch.erfc(-z/SQRT2)

def nll(mu, ls, lo, hi, c, mode):
    """mode: 'blind' | 'tobit' | 'interval'.  lo = observed y; hi = upper bound."""
    sig = ls.exp()
    zl = (lo-mu)/sig
    exact = -ls - 0.5*np.log(2*np.pi) - 0.5*zl**2
    if mode == 'blind':
        return -(exact.mean())
    if mode == 'tobit':
        cens = torch.log(torch.clamp(0.5*torch.erfc(zl/SQRT2), min=1e-10))
        return -(((1-c)*exact + c*cens).mean()) + 5.0*(torch.relu(mu-lo)**2*c).mean()
    zh = (hi-mu)/sig
    cens = torch.log(torch.clamp(_Phi(zh)-_Phi(zl), min=1e-10))
    return -(((1-c)*exact + c*cens).mean())

def fit(xnl, xtr, lo, hi, c, mode, epochs=900, seed=0):
    torch.manual_seed(seed); m = DCG(xnl.shape[1], xtr.shape[1])
    with torch.no_grad(): m.lin.bias.fill_(float(lo.mean())); m.lin.weight.fill_(0.)
    opt = torch.optim.AdamW(m.parameters(), lr=8e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        mu, ls = m(xnl, xtr)
        nll(mu, ls, lo, hi, c, mode).backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval(); return m

def prep(tr, te):
    a, b = tr[NONLIN].values.astype('float32'), te[NONLIN].values.astype('float32')
    cc, dd = tr[TREND].values.astype('float32'), te[TREND].values.astype('float32')
    m1, s1 = a.mean(0), a.std(0)+1e-8; m2, s2 = cc.mean(0), cc.std(0)+1e-8
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    return T((a-m1)/s1), T((b-m1)/s1), T((cc-m2)/s2), T((dd-m2)/s2)

pre  = X[(X.index >= '2016-04-23') & (X.index < '2022-01-01')].copy()
post = X[(X.index >= '2022-01-01') & (X.index < '2025-01-01')]
ratio = (post.ls_mwh/(post.gen_mwh+post.ls_mwh)); ratio = ratio[ratio > 0]
r_mean, r_std, p_shed = ratio.mean(), ratio.std(), post.censored.mean()
T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
ALPHA_MIN = 0.30

print("="*98)
print("EXPERIMENT 1c: TOBIT vs INTERVAL CENSORING, KNOWN GROUND TRUTH")
print("="*98)
print(f"P(shed)={p_shed:.3f}  depth mean={r_mean:.4f} sd={r_std:.4f}  alpha_min={ALPHA_MIN}")

rows = []
for cut in ['2019-01-01', '2020-01-01', '2021-01-01']:
    tr_m = np.asarray(pre.index < cut)
    te_m = np.asarray((pre.index >= cut) & (pre.index < pd.Timestamp(cut)+pd.DateOffset(years=1)))
    xnl_tr, xnl_te, xtr_tr, xtr_te = prep(pre[tr_m], pre[te_m])
    for admit in [1.00, 0.60, 0.40]:
        true_supp = p_shed*r_mean*(1-admit)*100
        print(f"\n--- holdout {cut[:4]} (train n={tr_m.sum()}), admit={admit:.2f}, "
              f"TRUE suppression = {true_supp:.2f}% of served ---")
        acc = {k: [] for k in ['blind', 'tobit', 'interval']}
        for seed in range(3):
            rng = np.random.default_rng(300+seed); n = len(pre)
            shed  = rng.random(n) < p_shed
            depth = np.clip(rng.normal(r_mean, r_std, n), 0.002, 0.35)*shed
            true_mwh = pre.gen_mwh.values
            G   = true_mwh*(1-depth)                 # served
            Sad = true_mwh*depth*admit               # admitted shortfall
            obs = G + Sad
            lo  = np.log(obs)
            hi  = np.log(G + Sad/ALPHA_MIN)
            hi  = np.where(shed, hi, lo)
            c   = shed.astype('float32')
            y_true = np.log(true_mwh)
            for mode in ['blind', 'tobit', 'interval']:
                m = fit(xnl_tr, xtr_tr, T(lo[tr_m]), T(hi[tr_m]), T(c[tr_m]), mode, seed=seed)
                with torch.no_grad(): mu, _ = m(xnl_te, xtr_te)
                p_, t_ = np.exp(mu.numpy()), np.exp(y_true[te_m])
                acc[mode].append(((p_-t_).mean()/t_.mean()*100, np.mean(np.abs(p_-t_)/t_)*100))
        for mode in ['blind', 'tobit', 'interval']:
            a = np.array(acc[mode])
            rows.append(dict(holdout=cut[:4], train_n=int(tr_m.sum()), admit=admit,
                             true_suppression_pct=true_supp, model=mode,
                             bias=a[:,0].mean(), bias_sd=a[:,0].std(), mape=a[:,1].mean()))
            print(f"    {mode:9s} bias={a[:,0].mean():+6.2f} +-{a[:,0].std():.2f}%   MAPE={a[:,1].mean():5.2f}%")
        # relative-to-blind correction actually applied vs what was needed
        b_blind = np.array(acc['blind'])[:,0].mean()
        for mode in ['tobit', 'interval']:
            applied = np.array(acc[mode])[:,0].mean() - b_blind
            print(f"      -> {mode:9s} applied a correction of {applied:+.2f} pp "
                  f"(needed {true_supp:+.2f} pp)")

pd.DataFrame(rows).round(3).to_csv('results/exp1c_interval.csv', index=False)
print("\nsaved -> results/exp1c_interval.csv")
