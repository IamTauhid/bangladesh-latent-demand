"""
Stage 7 - Compositional (simplex-constrained) fuel-mix forecasting.

The daily dispatch mix s_t = (gas, liquid, coal, renew, import) lives on the
4-simplex S^5: parts are non-negative and sum to one. Per-fuel regression respects
neither constraint. We work in Aitchison geometry.

(1) MERIT-ORDER ILR BASIS.  Instead of an arbitrary ALR reference we build an
    isometric log-ratio basis from a sequential binary partition that mirrors the
    dispatch hierarchy of this grid:

        b1 : {gas, liquid, coal}      vs {renew, import}   thermal vs non-thermal
        b2 : {gas}                    vs {liquid, coal}    cheap domestic gas vs rest of thermal
        b3 : {coal}                   vs {liquid}          baseload coal vs peaking oil
        b4 : {renew}                  vs {import}          indigenous RE vs cross-border

    Each balance is an economically interpretable quantity (a log price/merit ratio),
    and the basis is orthonormal, so Euclidean distance in ILR coordinates equals
    Aitchison distance on the simplex.

(2) DIRICHLET DEEP REGRESSION.  A network emits concentration parameters
    alpha(x) = exp(f(x)); the composition is modelled as s ~ Dir(alpha). This yields
    a valid composition by construction AND a calibrated predictive density over the
    mix - which no per-fuel baseline provides.

Essential/rounded zeros are handled by multiplicative replacement
(Martin-Fernandez et al., 2003) with delta_j = 0.65 * min positive share.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
torch.set_num_threads(4)
from sklearn.ensemble import HistGradientBoostingRegressor
torch.manual_seed(0); np.random.seed(0)

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
GRP = {'gas': ['gas'], 'liquid': ['liquid_fuel'], 'coal': ['coal'],
       'renew': ['hydro', 'solar', 'wind'],
       'import': ['india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']}
COMP = list(GRP)
E = pd.DataFrame({k: X[[c+'_mwh' for c in v]].sum(axis=1) for k, v in GRP.items()})
S = E.div(E.sum(axis=1), axis=0)

# ---- multiplicative zero replacement -------------------------------------
Z = (S <= 0)
delta = {c: 0.65*S.loc[S[c] > 0, c].min() for c in COMP}
Sr = S.copy()
for c in COMP:
    Sr.loc[Z[c], c] = delta[c]
Sr = Sr.div(Sr.sum(axis=1), axis=0)
print(f"zero cells replaced: {int(Z.values.sum())}/{S.size} ({Z.values.sum()/S.size*100:.2f}%)")

# ---- merit-order sequential binary partition -> ILR basis -----------------
SBP = [ (['gas', 'liquid', 'coal'], ['renew', 'import'], 'thermal|non-thermal'),
        (['gas'],                   ['liquid', 'coal'],  'gas|liquid+coal'),
        (['coal'],                  ['liquid'],          'coal|liquid'),
        (['renew'],                 ['import'],          'renew|import') ]
IDX = {c: i for i, c in enumerate(COMP)}

def build_psi():
    """Orthonormal ILR contrast matrix (4 x 5) from the SBP."""
    P = np.zeros((len(SBP), len(COMP)))
    for k, (plus, minus, _) in enumerate(SBP):
        r, s = len(plus), len(minus)
        a, b = np.sqrt(s/(r*(r+s))), -np.sqrt(r/(s*(r+s)))
        for c in plus:  P[k, IDX[c]] = a
        for c in minus: P[k, IDX[c]] = b
    return P
PSI = build_psi()
assert np.allclose(PSI @ np.ones(len(COMP)), 0, atol=1e-12), "contrasts must sum to zero"
assert np.allclose(PSI @ PSI.T, np.eye(len(SBP)), atol=1e-10), "basis must be orthonormal"
print("ILR basis verified: rows sum to zero, PSI PSI^T = I")

def ilr(s):     return np.log(np.asarray(s)) @ PSI.T
def ilr_inv(z):
    e = np.exp(np.clip(np.asarray(z) @ PSI, -30, 30))
    return e/e.sum(1, keepdims=True)

Y = pd.DataFrame(ilr(Sr.values), index=Sr.index, columns=[f'b{i+1}' for i in range(len(SBP))])
BAL = list(Y.columns)

def aitchison(a, b):
    la, lb = np.log(np.clip(a, 1e-12, None)), np.log(np.clip(b, 1e-12, None))
    ca, cb = la-la.mean(1, keepdims=True), lb-lb.mean(1, keepdims=True)
    return np.sqrt(((ca-cb)**2).sum(1))

# ---------------- design matrix -------------------------------------------
EX = ['CDD_at','T2M','RH2M','ALLSKY_SFC_SW_DWN','WS2M','rain7','PRECTOTCORR',
      'sin1','cos1','sin2','cos2','sin3','cos3','t',
      'is_friday','is_holiday','is_ramadan',
      'reg_fuelcrisis','cap_coal_p1','cap_coal_p2','rep_adani','cap_wind',
      'covid_lockdown','july_unrest']
D = X.join(Y)
D['log_demand'] = np.log(X.gen_mwh)          # coupling to L1: demand level drives merit order
for b in BAL:                                # AR structure of the balances (day-ahead task only)
    D[b+'_lag1'] = D[b].shift(1)
    D[b+'_lag7'] = D[b].shift(7)
    D[b+'_ma28'] = D[b].shift(1).rolling(28).mean()
AR = [b+s for b in BAL for s in ('_lag1', '_lag7', '_ma28')]
D = D.dropna(subset=EX+BAL+AR+['log_demand'])
Sr = Sr.loc[D.index]

TR = np.asarray(D.index < '2025-01-01'); TE = ~TR
Ste = Sr.values[TE]
print(f"train n={TR.sum()} ({D.index[TR].min().date()}..{D.index[TR].max().date()})  "
      f"test n={TE.sum()} ({D.index[TE].min().date()}..{D.index[TE].max().date()})")

T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)

class DirNet(nn.Module):
    """Dirichlet deep regression: s | x ~ Dir(alpha(x)), alpha = exp(f(x))."""
    def __init__(self, p, k, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(p, h), nn.SiLU(), nn.Dropout(0.10),
                               nn.Linear(h, h//2), nn.SiLU(),
                               nn.Linear(h//2, k))
        self.scale = nn.Parameter(torch.tensor(3.0))     # log total concentration
    def forward(self, x):
        logits = torch.log_softmax(self.f(x), dim=-1)
        return torch.exp(logits + self.scale).clamp(1e-3, 1e6)

def dirichlet_nll(alpha, s):
    a0 = alpha.sum(-1)
    return -(torch.lgamma(a0) - torch.lgamma(alpha).sum(-1)
             + ((alpha-1.0)*torch.log(s.clamp_min(1e-9))).sum(-1)).mean()

def run(task, FEATS, use_ar):
    print("\n" + "="*102); print(task); print("="*102)
    Xa = D[FEATS].values.astype('float32')
    m_, s_ = Xa[TR].mean(0), Xa[TR].std(0)+1e-8
    Xs = (Xa-m_)/s_
    Ytr = D[BAL].values[TR]
    Str = Sr.values[TR]
    res, preds = {}, {}

    preds['B1 persistence'] = np.repeat(Str[-1:], TE.sum(), 0)
    preds['B2 seasonal naive (t-364)'] = Sr.shift(364).loc[D.index[TE]].ffill().values

    raw = np.column_stack([HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
            random_state=0).fit(Xs[TR], E.loc[D.index[TR], c].values).predict(Xs[TE]) for c in COMP])
    res.setdefault('_viol', {})['B3 per-fuel GBM (levels)+renorm'] = float((raw < 0).any(1).mean()*100)
    preds['B3 per-fuel GBM (levels)+renorm'] = np.clip(raw, 1e-9, None)/np.clip(raw, 1e-9, None).sum(1, keepdims=True)

    rawS = np.column_stack([HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
            random_state=0).fit(Xs[TR], Str[:, i]).predict(Xs[TE]) for i in range(len(COMP))])
    res['_viol']['B4 per-share GBM+renorm'] = float(((rawS < 0).any(1) | (np.abs(rawS.sum(1)-1) > 0.01)).mean()*100)
    preds['B4 per-share GBM+renorm'] = np.clip(rawS, 1e-9, None)/np.clip(rawS, 1e-9, None).sum(1, keepdims=True)

    alr_ref = np.log(Sr.values[:, 1:]/Sr.values[:, :1])
    ag = np.column_stack([HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
            random_state=0).fit(Xs[TR], alr_ref[TR][:, j]).predict(Xs[TE]) for j in range(4)])
    ex = np.exp(np.clip(ag, -30, 30)); den = 1+ex.sum(1, keepdims=True)
    preds['C1 ALR-GBM (gas reference)'] = np.hstack([1/den, ex/den])

    ig = np.column_stack([HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
            random_state=0).fit(Xs[TR], Ytr[:, j]).predict(Xs[TE]) for j in range(len(BAL))])
    preds['C2 ILR-GBM (merit-order basis)'] = ilr_inv(ig)

    def dirnet(seed):
        torch.manual_seed(seed)
        m = DirNet(Xs.shape[1], len(COMP))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200)
        xt, st = T(Xs[TR]), T(Str)
        for _ in range(1200):
            m.train(); opt.zero_grad()
            dirichlet_nll(m(xt), st).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): return m(T(Xs[TE])).numpy()
    A = np.mean([dirnet(s) for s in range(5)], 0)
    preds['C3 Dirichlet-Net (proposed)'] = A/A.sum(1, keepdims=True)

    for k, p in preds.items():
        ad = aitchison(p, Ste); mae = np.abs(p-Ste).mean(0)*100
        res[k] = dict(aitchison=float(ad.mean()), aitchison_med=float(np.median(ad)),
                      mae_total=float(np.abs(p-Ste).mean()*100),
                      simplex_violation_pct=res.get('_viol', {}).get(k, 0.0),
                      **{'mae_'+c: float(v) for c, v in zip(COMP, mae)})
        print(f"  {k:34s} Aitch={ad.mean():.4f}  MAE%={res[k]['mae_total']:5.3f}  " +
              "  ".join(f"{c}={v:4.2f}" for c, v in zip(COMP, mae)) +
              f"  viol={res[k]['simplex_violation_pct']:.1f}%")
    res.pop('_viol', None)
    best_b = min([k for k in res if k.startswith('B')], key=lambda k: res[k]['aitchison'])
    for k in res:
        res[k]['aitch_gain_vs_'+best_b] = round((1-res[k]['aitchison']/res[best_b]['aitchison'])*100, 2)
    print(f"  -> best baseline = {best_b}; proposed C3 improves Aitchison by "
          f"{res['C3 Dirichlet-Net (proposed)']['aitch_gain_vs_'+best_b]:.1f}%, "
          f"C2 by {res['C2 ILR-GBM (merit-order basis)']['aitch_gain_vs_'+best_b]:.1f}%")
    return res, preds

resB, pB = run("TASK B - ex-ante 432-day mix forecast, exogenous drivers only", EX+['log_demand'], False)
resA, pA = run("TASK A - day-ahead mix forecast, balance lags permitted", EX+['log_demand']+AR, True)

json.dump({'ex_ante': resB, 'day_ahead': resA}, open('results/l2_composition.json', 'w'), indent=2)
pd.DataFrame(resB).T.round(4).to_csv('results/l2_ex_ante.csv')
pd.DataFrame(resA).T.round(4).to_csv('results/l2_day_ahead.csv')
np.savez('results/l2_preds.npz', actual=Ste,
         **{'B_'+k: v for k, v in pB.items()}, **{'A_'+k: v for k, v in pA.items()},
         dates=D.index[TE].astype('int64').values, comp=np.array(COMP, dtype=object))
print("\nsaved -> results/l2_composition.json, l2_ex_ante.csv, l2_day_ahead.csv, l2_preds.npz")
