"""
Stage 5 - Deep Censored Gaussian (DCG) latent-demand model.

Latent daily demand  D_t (log MWh) ~ N(mu_t, sigma_t^2)
    mu_t        = beta0 + beta1 * t~   +  g_theta(x_t)      (partially linear: trend extrapolates)
    log sigma_t = h_theta(x_t)                              (heteroscedastic)

The grid serves G_t = min(D_t, C_t) under rationing; the operator publishes
y_t = log(G_t + S_t), with S_t the ADMITTED shortfall. Since S_t understates true
unserved energy, y_t is RIGHT-CENSORED for D_t on shed days (c_t=1), exact otherwise.

NLL = -sum_t [ (1-c_t)*log N(y|mu,sigma) + c_t*log(1-Phi((y-mu)/sigma)) ]

The censored term saturates as mu->inf, so mu is anchored solely by exact
observations; a ridge penalty on the latent-inflation direction keeps the
optimisation well posed (Sec. III-D).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
torch.set_num_threads(4)
torch.manual_seed(0); np.random.seed(0)

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')

NONLIN = ['CDD_at','CDD','HDD','T2M','T2M_MAX','RH2M','THI','apparent_temp','CDD_ma7','CDD_lag1',
          'T_ma3','dT','rain7','WS2M','ALLSKY_SFC_SW_DWN',
          'sin1','cos1','sin2','cos2','sin3','cos3',
          'is_friday','is_saturday','is_holiday','is_ramadan','eid_prox',
          'covid_lockdown','july_unrest']
TREND = ['t']

class DCG(nn.Module):
    """mu = linear(trend) + MLP(covariates); log-sigma = MLP head. Partially linear."""
    def __init__(self, p_nl, p_tr, h=96):
        super().__init__()
        self.lin  = nn.Linear(p_tr, 1)                       # extrapolating trend
        self.body = nn.Sequential(nn.Linear(p_nl,h), nn.SiLU(), nn.Dropout(0.10),
                                  nn.Linear(h,h//2), nn.SiLU())
        self.mu   = nn.Linear(h//2,1)
        self.ls   = nn.Linear(h//2,1)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias,-3.0)
    def forward(self, xnl, xtr):
        z = self.body(xnl)
        return self.lin(xtr).squeeze(-1) + self.mu(z).squeeze(-1), self.ls(z).squeeze(-1).clamp(-7,1)

SQRT2 = float(np.sqrt(2.0))
def censored_nll(mu, logsig, y, c, ridge=0.0):
    sig = logsig.exp(); z = (y-mu)/sig
    ll_exact = -logsig - 0.5*np.log(2*np.pi) - 0.5*z**2
    ll_cens  = torch.log(torch.clamp(0.5*torch.erfc(z/SQRT2), min=1e-10))
    nll = -(((1-c)*ll_exact + c*ll_cens).mean())
    if ridge > 0:                     # penalise unbounded latent inflation on censored points
        nll = nll + ridge*(torch.relu(mu-y)**2 * c).mean()
    return nll

def fit(xnl, xtr, y, c, censored=True, epochs=1500, lr=8e-3, wd=1e-4, ridge=5.0, seed=0):
    torch.manual_seed(seed)
    m = DCG(xnl.shape[1], xtr.shape[1])
    with torch.no_grad():                                    # warm-start intercept at mean(y)
        m.lin.bias.fill_(float(y.mean())); m.lin.weight.fill_(0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    c_use = c if censored else torch.zeros_like(c)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        mu, ls = m(xnl, xtr)
        loss = censored_nll(mu, ls, y, c_use, ridge if censored else 0.0)
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); sch.step()
    m.eval(); return m

def prep(tr_df, te_df):
    a_nl, b_nl = tr_df[NONLIN].values.astype('float32'), te_df[NONLIN].values.astype('float32')
    a_tr, b_tr = tr_df[TREND].values.astype('float32'),  te_df[TREND].values.astype('float32')
    m1,s1 = a_nl.mean(0), a_nl.std(0)+1e-8
    m2,s2 = a_tr.mean(0), a_tr.std(0)+1e-8
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    return T((a_nl-m1)/s1), T((b_nl-m1)/s1), T((a_tr-m2)/s2), T((b_tr-m2)/s2)

# =====================================================================
# EXPERIMENT 1 - synthetic censoring recovery (multi-seed, two holdout years)
# =====================================================================
print("="*76); print("EXPERIMENT 1: SYNTHETIC CENSORING RECOVERY"); print("="*76)
pre  = X[(X.index>='2016-04-23') & (X.index<'2022-01-01')].copy()
post = X[(X.index>='2022-01-01') & (X.index<'2025-01-01')].copy()
ratio = (post.ls_mwh/(post.gen_mwh+post.ls_mwh)); ratio = ratio[ratio>0]
r_mean, r_std, p_shed = ratio.mean(), ratio.std(), post.censored.mean()
print(f"uncensored era n={len(pre)} | crisis era n={len(post)}  P(shed)={p_shed:.3f}  "
      f"curtailment depth mean={r_mean:.4f} sd={r_std:.4f}")

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)

rows=[]
for cut, label in [('2019-01-01','holdout 2019 (normal growth)'),
                   ('2021-01-01','holdout 2021 (post-COVID rebound)')]:
    tr_m  = pre.index < cut
    te_m  = (pre.index >= cut) & (pre.index < pd.Timestamp(cut)+pd.DateOffset(years=1))
    tr_df, te_df = pre[tr_m], pre[te_m]
    xnl_tr, xnl_te, xtr_tr, xtr_te = prep(tr_df, te_df)
    print(f"--- {label}: train n={tr_m.sum()}, test n={te_m.sum()} ---")
    for admit in [1.00, 0.60, 0.40]:
        acc = {k: [] for k in ['DCG','Naive']}
        for seed in range(5):
            rng = np.random.default_rng(100+seed); n=len(pre)
            shed  = rng.random(n) < p_shed
            depth = np.clip(rng.normal(r_mean, r_std, n), 0.002, 0.35)*shed
            true_mwh = pre.gen_mwh.values
            obs_mwh  = true_mwh*(1-depth) + true_mwh*depth*admit
            y_obs, y_true, c_obs = np.log(obs_mwh), np.log(true_mwh), shed.astype('float32')
            for tag, cflag in [('DCG',True), ('Naive',False)]:
                m = fit(xnl_tr, xtr_tr, T(y_obs[tr_m]), T(c_obs[tr_m]), censored=cflag, seed=seed)
                with torch.no_grad(): mu,_ = m(xnl_te, xtr_te)
                p_, t_ = np.exp(mu.numpy()), np.exp(y_true[te_m])
                acc[tag].append(((p_-t_).mean()/t_.mean()*100, np.mean(np.abs(p_-t_)/t_)*100))
        out={}
        for tag in ['DCG','Naive']:
            a=np.array(acc[tag]); out[tag]=(a[:,0].mean(), a[:,0].std(), a[:,1].mean(), a[:,1].std())
        d_bias = abs(out['Naive'][0]) - abs(out['DCG'][0])
        print(f"  admit={admit:.2f} | DCG   bias {out['DCG'][0]:+6.2f}+-{out['DCG'][1]:.2f}%  MAPE {out['DCG'][2]:5.2f}%")
        print(f"              | Naive bias {out['Naive'][0]:+6.2f}+-{out['Naive'][1]:.2f}%  MAPE {out['Naive'][2]:5.2f}%"
              f"   -> |bias| reduced {d_bias:+.2f} pp")
        rows.append(dict(holdout=label, admit=admit,
                         dcg_bias=out['DCG'][0], dcg_bias_sd=out['DCG'][1], dcg_mape=out['DCG'][2],
                         naive_bias=out['Naive'][0], naive_bias_sd=out['Naive'][1], naive_mape=out['Naive'][2],
                         bias_reduction_pp=d_bias))
    print()
pd.DataFrame(rows).round(3).to_csv('results/exp1_synthetic_censoring.csv', index=False)
print("saved -> results/exp1_synthetic_censoring.csv")
