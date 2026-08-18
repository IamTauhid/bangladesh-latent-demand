"""
Stage 6 - Operational forecasting benchmark.

Task A  day-ahead (h=1), autoregressive features permitted, fixed origin.
Task B  ex-ante long horizon: train <= 2024-12-31, forecast 2025-01-01..2026-03-08
        (433 days) using exogenous drivers only - no observed target in the test span.

Compared: seasonal naive, Ridge, Random Forest, XGBoost, MLP, LSTM,
and the proposed censoring-aware DCG. Significance by Diebold-Mariano.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json, warnings
torch.set_num_threads(4); warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy import stats
torch.manual_seed(0); np.random.seed(0)

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
y = np.log(X.gen_mwh.values)
c = X.censored.values.astype('float32')

EXOG = ['CDD_at','CDD','HDD','T2M','T2M_MAX','RH2M','THI','apparent_temp','CDD_ma7','CDD_lag1',
        'T_ma3','dT','rain7','WS2M','ALLSKY_SFC_SW_DWN','sin1','cos1','sin2','cos2','sin3','cos3',
        'is_friday','is_saturday','is_holiday','is_ramadan','eid_prox','covid_lockdown','july_unrest']
AR    = ['gen_lag1','gen_lag2','gen_lag3','gen_lag7','gen_lag14','gen_lag364','gen_ma7','gen_ma28','gen_std7']
TREND = ['t']

def mape(a, p): return float(np.mean(np.abs(p-a)/a)*100)
def rmse(a, p): return float(np.sqrt(np.mean((p-a)**2)))
def mae(a, p):  return float(np.mean(np.abs(p-a)))

def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared-error loss. H0: equal predictive accuracy."""
    d = e1**2 - e2**2
    n = len(d); dbar = d.mean()
    g = [np.sum((d[k:]-dbar)*(d[:n-k]-dbar))/n for k in range(h)]
    var = (g[0] + 2*sum(g[1:]))/n
    if var <= 0: return np.nan, np.nan
    stat = dbar/np.sqrt(var)
    return float(stat), float(2*(1-stats.norm.cdf(abs(stat))))

# ---------------- proposed model ----------------
SQRT2 = float(np.sqrt(2.0))

class DCG(nn.Module):
    """Partially linear heteroscedastic net: mu = linear(trend) + MLP(x); log-sigma = MLP(x)."""
    def __init__(self, p_nl, p_tr, h=96):
        super().__init__()
        self.lin  = nn.Linear(p_tr, 1)
        self.body = nn.Sequential(nn.Linear(p_nl, h), nn.SiLU(), nn.Dropout(0.10),
                                  nn.Linear(h, h//2), nn.SiLU())
        self.mu = nn.Linear(h//2, 1); self.ls = nn.Linear(h//2, 1)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias, -3.0)
    def forward(self, xnl, xtr):
        z = self.body(xnl)
        return self.lin(xtr).squeeze(-1) + self.mu(z).squeeze(-1), self.ls(z).squeeze(-1).clamp(-7, 1)

def cnll(mu, ls, yy, cc, ridge=5.0):
    sig = ls.exp(); z = (yy-mu)/sig
    le = -ls - 0.5*np.log(2*np.pi) - 0.5*z**2
    lc = torch.log(torch.clamp(0.5*torch.erfc(z/SQRT2), min=1e-10))
    return -(((1-cc)*le + cc*lc).mean()) + ridge*(torch.relu(mu-yy)**2*cc).mean()

def fit_dcg(xnl, xtr, yy, cc, censored=True, epochs=1200, seed=0):
    torch.manual_seed(seed)
    m = DCG(xnl.shape[1], xtr.shape[1])
    with torch.no_grad():
        m.lin.bias.fill_(float(yy.mean())); m.lin.weight.fill_(0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=8e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cu = cc if censored else torch.zeros_like(cc)
    rg = 5.0 if censored else 0.0
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        mu, ls = m(xnl, xtr)
        cnll(mu, ls, yy, cu, rg).backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval(); return m

class LSTMNet(nn.Module):
    def __init__(self, p, h=64):
        super().__init__(); self.l = nn.LSTM(p, h, batch_first=True); self.o = nn.Linear(h, 1)
    def forward(self, x): return self.o(self.l(x)[0][:, -1]).squeeze(-1)

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)

def run_task(name, FEATS, tr_mask, te_mask, seq=False):
    print("\n" + "="*76); print(name); print("="*76)
    Xa = X[FEATS].values.astype('float32')
    m_, s_ = Xa[tr_mask].mean(0), Xa[tr_mask].std(0)+1e-8
    Xs = (Xa-m_)/s_
    ytr, yte = y[tr_mask], y[te_mask]
    a = np.exp(yte)
    preds = {}

    preds['Seasonal naive (t-364)'] = X.gen_lag364.values[te_mask]
    preds['Ridge'] = np.exp(Ridge(alpha=1.0).fit(Xs[tr_mask], ytr).predict(Xs[te_mask]))
    preds['Random Forest'] = np.exp(RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1
        ).fit(Xs[tr_mask], ytr).predict(Xs[te_mask]))
    preds['XGBoost'] = np.exp(XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, random_state=0, n_jobs=4
        ).fit(Xs[tr_mask], ytr).predict(Xs[te_mask]))

    def mlp(seed):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(Xs.shape[1], 128), nn.SiLU(), nn.Dropout(0.1),
                          nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        with torch.no_grad(): m[-1].bias.fill_(float(ytr.mean()))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 700)
        xt, yt = T(Xs[tr_mask]), T(ytr)
        for _ in range(700):
            m.train(); opt.zero_grad()
            nn.functional.huber_loss(m(xt).squeeze(-1), yt, delta=1.0).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): return m(T(Xs[te_mask])).squeeze(-1).numpy()
    preds['MLP'] = np.exp(np.mean([mlp(s) for s in range(3)], 0))

    if seq:
        L = 14
        itr, ite = np.where(tr_mask)[0], np.where(te_mask)[0]
        ktr = [i for i in itr if i >= L]; kte = [i for i in ite if i >= L]
        Str = np.stack([Xs[i-L:i] for i in ktr]); ytr2 = y[ktr]
        Ste = np.stack([Xs[i-L:i] for i in kte])
        def lstm(seed):
            torch.manual_seed(seed); m = LSTMNet(Xs.shape[1])
            with torch.no_grad(): m.o.bias.fill_(float(ytr2.mean()))
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
            xt, yt = T(Str), T(ytr2)
            for _ in range(600):
                m.train(); opt.zero_grad()
                nn.functional.huber_loss(m(xt), yt, delta=1.0).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
            m.eval()
            with torch.no_grad(): return m(T(Ste)).numpy()
        lp = np.exp(np.mean([lstm(s) for s in range(3)], 0))
        full = np.full(int(te_mask.sum()), np.nan)
        pos = {k: i for i, k in enumerate(ite)}
        for j, k in enumerate(kte): full[pos[k]] = lp[j]
        preds['LSTM'] = pd.Series(full).ffill().bfill().values

    nl = [f for f in FEATS if f not in TREND]
    Xnl = X[nl].values.astype('float32'); Xtr_ = X[TREND].values.astype('float32')
    mn, sn = Xnl[tr_mask].mean(0), Xnl[tr_mask].std(0)+1e-8
    mt, st = Xtr_[tr_mask].mean(0), Xtr_[tr_mask].std(0)+1e-8
    Xnl, Xtr_ = (Xnl-mn)/sn, (Xtr_-mt)/st
    for tag, cf in [('DCG-blind (ablation)', False), ('DCG censoring-aware (proposed)', True)]:
        ps = []
        for s in range(3):
            m = fit_dcg(T(Xnl[tr_mask]), T(Xtr_[tr_mask]), T(ytr), T(c[tr_mask]), censored=cf, seed=s)
            with torch.no_grad(): mu, _ = m(T(Xnl[te_mask]), T(Xtr_[te_mask]))
            ps.append(mu.numpy())
        preds[tag] = np.exp(np.mean(ps, 0))

    out = {}
    for k, p in preds.items():
        out[k] = dict(MAPE=round(mape(a, p), 3), RMSE=round(rmse(a, p), 1), MAE=round(mae(a, p), 1))
        print(f"  {k:34s} MAPE={out[k]['MAPE']:6.3f}%  RMSE={out[k]['RMSE']:8.0f}  MAE={out[k]['MAE']:8.0f}")

    for k, p in preds.items():
        out[k]['mean_signed_dev_pct'] = round(float(((p-a)/a).mean()*100), 3)
    conv = [k for k in preds if 'DCG' not in k]
    best = min(conv, key=lambda k: out[k]['MAPE'])
    print(f"  [DM reference = best conventional baseline: {best}]")
    for k in preds:
        if k == best: continue
        s, pv = dm_test(preds[best]-a, preds[k]-a)
        if s == s:
            out[k]['DM_stat'] = round(s, 3); out[k]['DM_p'] = round(pv, 4)
            sig = '**' if pv < 0.01 else ('*' if pv < 0.05 else '')
            print(f"    DM {k:32s} stat={s:+7.3f}  p={pv:.4f} {sig}")
    return out, preds, a

TR = np.asarray(X.index < '2025-01-01'); TE = ~TR
resA, pA, aA = run_task("TASK A - day-ahead (h=1), AR features permitted",
                        EXOG+AR+TREND, TR, TE, seq=True)
resB, pB, aB = run_task("TASK B - ex-ante 433-day horizon, exogenous only (no target leakage)",
                        EXOG+TREND, TR, TE, seq=False)

json.dump({'task_A_day_ahead': resA, 'task_B_ex_ante': resB},
          open('results/l1_forecast.json', 'w'), indent=2)
pd.DataFrame(resA).T.to_csv('results/l1_task_a.csv')
pd.DataFrame(resB).T.to_csv('results/l1_task_b.csv')
np.savez('results/l1_preds.npz',
         **{'A_'+k: v for k, v in pA.items()}, **{'B_'+k: v for k, v in pB.items()},
         actual=aA, dates=X.index[TE].astype('int64').values)
print("\nsaved -> results/l1_forecast.json, l1_task_a.csv, l1_task_b.csv, l1_preds.npz")
