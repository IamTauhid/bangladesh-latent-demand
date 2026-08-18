"""
Stage 9 - Forward projection to 2030 (demand + fuel mix + cost).

Demand: the L1 partially linear DCG separates an extrapolable trend from a
non-linear weather/calendar response, so out-of-sample projection is done by
(i) advancing the linear trend, (ii) driving the non-linear block with a synthetic
future weather year, (iii) reporting LOW / CENTRAL / HIGH scenarios rather than a
single path.

Future weather = day-of-year climatology over the last 10 observed years plus a
warming increment of +0.025 C/yr (IPCC AR6 SSP2-4.5 regional near-term rate for
South Asia), applied to temperature-derived covariates.

Fuel mix: L2 ILR/Dirichlet model driven by the projected demand and by an explicit
capacity-commissioning schedule (scenario input, stated in Table).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
torch.set_num_threads(4)
torch.manual_seed(0); np.random.seed(0)

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
HORIZON_END = '2030-12-31'
WARMING = 0.025          # deg C per year

# ---------------- build the future exogenous frame -------------------------
# Weather realisations: instead of day-of-year climatology (which smooths away the
# extremes that a convex cooling response depends on -- Jensen's inequality), we
# replay each of the last NREP observed weather years onto every future year and
# shift temperatures for warming. The spread across replays is the weather band.
NREP = 5
last = X.index.max()
fut_idx = pd.date_range(last + pd.Timedelta(days=1), HORIZON_END, freq='D')
WCOLS = ['T2M','T2M_MAX','T2M_MIN','RH2M','WS2M','ALLSKY_SFC_SW_DWN','PRECTOTCORR',
         'apparent_temp','CDD','HDD','CDD_at','THI']
src_years = [y for y in range(last.year-NREP, last.year) if (X.index.year == y).sum() > 350]
print(f"weather replay years: {src_years}")

def doy_key(idx):
    return np.where((idx.month == 2) & (idx.day == 29), 59, idx.dayofyear)

def build_frame(rep_year):
    """Future exogenous frame with weather replayed from `rep_year`."""
    src = X[X.index.year == rep_year]
    tbl = src.groupby(doy_key(src.index))[WCOLS].mean()
    F = pd.DataFrame(index=fut_idx)
    for c in WCOLS:
        F[c] = tbl[c].reindex(doy_key(F.index)).values
    F[WCOLS] = F[WCOLS].ffill().bfill()
    ya = (F.index.year - last.year) + (F.index.dayofyear/365.25)
    for c in ['T2M','T2M_MAX','T2M_MIN','apparent_temp']:
        F[c] = F[c] + WARMING*ya
    F['CDD']    = (F.T2M - 22.0).clip(lower=0)
    F['HDD']    = (22.0 - F.T2M).clip(lower=0)
    F['CDD_at'] = (F.apparent_temp - 22.0).clip(lower=0)
    F['CDD_ma7']  = F.CDD_at.rolling(7, min_periods=1).mean()
    F['CDD_lag1'] = F.CDD_at.shift(1).bfill()
    F['T_ma3']    = F.T2M.rolling(3, min_periods=1).mean()
    F['dT']       = F.T2M.diff().fillna(0)
    F['rain7']    = F.PRECTOTCORR.rolling(7, min_periods=1).sum()

    F['dow'] = F.index.dayofweek; F['doy'] = F.index.dayofyear
    F['is_friday'] = (F.dow == 4).astype(int); F['is_saturday'] = (F.dow == 5).astype(int)
    for k in (1, 2, 3):
        F[f'sin{k}'] = np.sin(2*np.pi*k*F.doy/365.25); F[f'cos{k}'] = np.cos(2*np.pi*k*F.doy/365.25)
    F['t'] = (F.index - X.index.min()).days
    F['is_ramadan'] = 0; F['eid_prox'] = 0.0
    for yy, sdate in RAMADAN.items():
        sdate = pd.Timestamp(sdate)
        F.loc[(F.index >= sdate) & (F.index < sdate+pd.Timedelta(days=30)), 'is_ramadan'] = 1
    for yy, ds in EIDS.items():
        for sdate in ds:
            sdate = pd.Timestamp(sdate)
            w = F.index[(F.index >= sdate-pd.Timedelta(days=3)) & (F.index <= sdate+pd.Timedelta(days=4))]
            F.loc[w, 'eid_prox'] = np.maximum(F.loc[w, 'eid_prox'], 1-np.abs((w-sdate).days)/5.0)
    F['is_holiday'] = 0
    for m, dd in [(2,21),(3,26),(4,14),(5,1),(8,15),(12,16),(12,25)]:
        F.loc[(F.index.month == m) & (F.index.day == dd), 'is_holiday'] = 1
    F['covid_lockdown'] = 0; F['july_unrest'] = 0
    F['reg_fuelcrisis'] = 0
    F['cap_coal_p1'] = 1; F['cap_coal_p2'] = 1; F['rep_adani'] = 1; F['cap_wind'] = 1
    return F

RAMADAN = {2026:'2026-02-19', 2027:'2027-02-08', 2028:'2028-01-28', 2029:'2029-01-16', 2030:'2030-01-06'}
EIDS = {2026:['2026-03-20','2026-05-27'], 2027:['2027-03-09','2027-05-16'],
        2028:['2028-02-26','2028-05-05'], 2029:['2029-02-14','2029-04-24'],
        2030:['2030-02-04','2030-04-13']}
FRAMES = {yy: build_frame(yy) for yy in src_years}
F = FRAMES[src_years[-1]]

# ---------------- L1: refit DCG on all data, project ----------------------
NONLIN = ['CDD_at','CDD','HDD','T2M','T2M_MAX','RH2M','THI','apparent_temp','CDD_ma7','CDD_lag1',
          'T_ma3','dT','rain7','WS2M','ALLSKY_SFC_SW_DWN','sin1','cos1','sin2','cos2','sin3','cos3',
          'is_friday','is_saturday','is_holiday','is_ramadan','eid_prox','covid_lockdown','july_unrest']
SQRT2 = float(np.sqrt(2.0))

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

ALPHA_MIN = 0.30      # operator is assumed to admit at least 30% of the true shortfall

def _Phi(z): return 0.5*torch.erfc(-z/SQRT2)

def cnll(mu, ls, lo, hi, c):
    """Interval-censored Gaussian NLL. On shed days the latent log-demand lies in
    [lo, hi] = [log(G+S), log(G+S/alpha_min)] -- the width is set by the REPORTED
    shortfall, so the correction scales with observed severity (Sec. III-B)."""
    sig = ls.exp()
    zl = (lo-mu)/sig
    exact = -ls - 0.5*np.log(2*np.pi) - 0.5*zl**2
    zh = (hi-mu)/sig
    cens = torch.log(torch.clamp(_Phi(zh)-_Phi(zl), min=1e-10))
    return -(((1-c)*exact + c*cens).mean())

Xnl = X[NONLIN].values.astype('float32'); Xtr = X[['t']].values.astype('float32')
mn, sn = Xnl.mean(0), Xnl.std(0)+1e-8; mt, st = Xtr.mean(0), Xtr.std(0)+1e-8
Fnl = ((F[NONLIN].values.astype('float32')-mn)/sn)
Ftr = ((F[['t']].values.astype('float32')-mt)/st)
served = X.gen_mwh.values.astype('float64')
adm    = X.ls_mwh.values.astype('float64')
c      = X.censored.values.astype('float32')
lo_np  = np.log(served + adm).astype('float32')                      # alpha = 1
hi_np  = np.log(served + np.where(c > 0, adm/ALPHA_MIN, 0.0)).astype('float32')
hi_np  = np.maximum(hi_np, lo_np)
y      = lo_np                                                        # for warm start
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)

def fit(seed):
    torch.manual_seed(seed); m = DCG(len(NONLIN))
    with torch.no_grad(): m.lin.bias.fill_(float(y.mean())); m.lin.weight.fill_(0.)
    opt = torch.optim.AdamW(m.parameters(), lr=8e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200)
    xn, xt = T((Xnl-mn)/sn), T((Xtr-mt)/st)
    lo_t, hi_t, cc = T(lo_np), T(hi_np), T(c)
    for _ in range(1200):
        m.train(); opt.zero_grad()
        mu, ls = m(xn, xt); cnll(mu, ls, lo_t, hi_t, cc).backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval(); return m

models = [fit(s) for s in range(5)]
with torch.no_grad():
    hist = np.mean([m(T((Xnl-mn)/sn), T((Xtr-mt)/st))[0].numpy() for m in models], 0)

X['latent_mwh'] = np.exp(hist)
X['suppression_pct'] = (X.latent_mwh/X.gen_mwh - 1)*100

# project under every weather replay x every model seed
paths = []
for ry, Fr in FRAMES.items():
    fnl = T((Fr[NONLIN].values.astype('float32')-mn)/sn)
    ftr = T((Fr[['t']].values.astype('float32')-mt)/st)
    with torch.no_grad():
        for m in models:
            paths.append(pd.Series(np.exp(m(fnl, ftr)[0].numpy()), index=Fr.index))
P = pd.concat(paths, axis=1)
central = P.mean(axis=1)

# complete 2026 with the already-observed part of the year
obs_2026 = X.gen_mwh[X.index.year == 2026].sum()

def annual(series):
    a = (series.astype('float64').groupby(series.index.year).sum())/1e6
    if 2026 in a.index: a.loc[2026] = a.loc[2026] + obs_2026/1e6
    return a

ann = pd.DataFrame({
    'weather_low' : annual(P.min(axis=1)),
    'central'     : annual(central),
    'weather_high': annual(P.max(axis=1)),
})
# growth-scenario band: +/-1.5 pp on the trend, applied to the central path
for tag, adj in [('low', -0.015), ('high', +0.015)]:
    scaled = central*np.power(1+adj, (central.index-last).days/365.25)
    ann['growth_'+tag] = annual(scaled)

base_cagr = float(np.exp(models[0].lin.weight.item()/st[0]*365.25)-1)
obs = (X.gen_mwh.groupby(X.index.year).sum()/1e6)
lat = (X.latent_mwh.groupby(X.index.year).sum()/1e6)

print("="*94); print("DEMAND PROJECTION (TWh/yr)"); print("="*94)
print(f"implied trend CAGR from the ICG linear term: {base_cagr*100:.2f}%/yr")
print(f"alpha_min = {ALPHA_MIN} (operator admits at least this share of true shortfall)")
print(f"weather replays: {list(FRAMES)}  x  {len(models)} model seeds = {P.shape[1]} paths")
h = pd.DataFrame({'served_TWh': obs.round(2), 'latent_TWh': lat.round(2)})
h['suppression_%'] = ((lat/obs-1)*100).round(2)
print(); print(h.tail(6).to_string())
print(); print(ann.round(2).to_string())
print(); print("implied CAGR of central path 2027->2030: "
      f"{(ann.central.loc[2030]/ann.central.loc[2027])**(1/3)*100-100:.2f}%/yr")

X[['gen_mwh','latent_mwh','suppression_pct','censored']].to_csv('results/latent_demand.csv')
ann.round(3).to_csv('results/projection_demand.csv')
pd.DataFrame({'central': central, 'lo': P.min(axis=1), 'hi': P.max(axis=1)}).to_csv('results/projection_daily.csv')
json.dump({'trend_cagr': base_cagr, 'warming_c_per_yr': WARMING, 'horizon_end': HORIZON_END,
           'weather_replay_years': list(FRAMES), 'n_paths': int(P.shape[1])},
          open('results/projection_meta.json','w'), indent=2)
print("saved -> results/latent_demand.csv, projection_demand.csv, projection_daily.csv")

