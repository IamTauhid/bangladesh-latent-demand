"""
Stage 17 - corrected Layer 1: partial identification of latent demand,
with the demand-destruction confound addressed explicitly.

WHY THE POINT ESTIMATOR WAS WITHDRAWN
-------------------------------------
The interval [G+S, G+S/alpha_min] is a VALID bound: it contains true demand whenever
the operator's admission rate alpha >= alpha_min. But the maximum-likelihood point
INSIDE it is not identified, and it moves the WRONG WAY. Interval width is set by the
REPORTED shortfall S, whereas true unreported suppression scales as S(1/alpha - 1).
When the operator admits everything (alpha=1) S is large, the interval is wide and the
estimator corrects strongly -- yet the truth needs no correction at all. On the
synthetic design corr(true suppression, applied correction) = -0.73. We therefore
report what the data actually identifies:

  (A) MODEL-FREE BOUNDS:  suppression in [ S/G , S/(alpha_min*G) ].
  (B) STRUCTURAL COUNTERFACTUAL: a transparent OLS demand model fitted on the
      UNCENSORED era (2016-2021) and extrapolated into the rationed era. It never
      touches S, so it is independent evidence. OLS is deliberate: the claim is an
      identification claim, and a linear model with an explicit trend is auditable,
      numerically stable, and lets price and activity controls enter transparently.
  (C) CONFOUND: the same gap would be produced by genuine demand destruction
      (tariff rises, industrial slowdown). Tested directly with a REAL tariff control.
  (D) CONSISTENCY: does (B) lie inside (A)?
"""
import numpy as np, pandas as pd, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')

X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
ALPHA_MIN = 0.30

# ---- exogenous macro series (published annual values) ----------------------
TARIFF_NOM = {2016: 6.10, 2017: 6.35, 2018: 6.60, 2019: 6.85, 2020: 6.94,
              2021: 7.04, 2022: 7.13, 2023: 7.72, 2024: 8.25, 2025: 8.60, 2026: 8.95}
CPI_INFL   = {2016: 0.058, 2017: 0.056, 2018: 0.057, 2019: 0.055, 2020: 0.056,
              2021: 0.055, 2022: 0.077, 2023: 0.099, 2024: 0.102, 2025: 0.085,
              2026: 0.070}
GDP_GROWTH = {2016: 7.1, 2017: 7.3, 2018: 7.9, 2019: 8.2, 2020: 3.4, 2021: 6.9,
              2022: 7.1, 2023: 5.8, 2024: 4.2, 2025: 4.0, 2026: 4.5}

cpi, lvl = {}, 1.0
for y in sorted(CPI_INFL):
    lvl *= (1 + CPI_INFL[y]); cpi[y] = lvl
base = cpi[2021]
TARIFF_REAL = {y: TARIFF_NOM[y] / (cpi[y]/base) for y in TARIFF_NOM}

print("="*92); print("REAL vs NOMINAL AVERAGE RETAIL TARIFF (2021 = base)"); print("="*92)
for y in sorted(TARIFF_NOM):
    print(f"  {y}: nominal {TARIFF_NOM[y]:5.2f} | CPI idx {cpi[y]/base:5.3f} | "
          f"real {TARIFF_REAL[y]:5.2f} BDT/kWh")
chg_nom  = TARIFF_NOM[2025]/TARIFF_NOM[2021] - 1
chg_real = TARIFF_REAL[2025]/TARIFF_REAL[2021] - 1
print(f"\n2021 -> 2025: nominal {chg_nom*100:+.1f}%   REAL {chg_real*100:+.1f}%")

X['yr'] = X.index.year
X['tariff_real'] = X.yr.map(TARIFF_REAL)
X['log_tariff_real'] = np.log(X.tariff_real)
X['gdp_growth'] = X.yr.map(GDP_GROWTH)
X['CDD_at2'] = X.CDD_at**2

# =====================================================================
print(); print("="*92); print("(A) MODEL-FREE BOUNDS ON SUPPRESSION (no estimation)"); print("="*92)
d = pd.read_csv('data/daily.csv', parse_dates=['date'], index_col='date')
yg = d.groupby(d.index.year)
G, S = yg.gen_mwh.sum(), yg.ls_mwh.sum()
bounds = pd.DataFrame({'served_TWh': G/1e6, 'lo_pct': S/G*100,
                       'hi_pct': S/(ALPHA_MIN*G)*100})
bounds = bounds.drop(index=[v for v in (2015, 2026) if v in bounds.index])
print(bounds.round(2).to_string())

# =====================================================================
FEAT = ['t', 'CDD_at', 'CDD_at2', 'HDD', 'RH2M', 'rain7', 'ALLSKY_SFC_SW_DWN',
        'sin1', 'cos1', 'sin2', 'cos2', 'sin3', 'cos3',
        'is_friday', 'is_saturday', 'is_holiday', 'is_ramadan', 'eid_prox',
        'covid_lockdown']
y  = np.log(X.gen_mwh)
tr = X.index.year <= 2021

def counterfactual(features):
    m = sm.OLS(y[tr], sm.add_constant(X.loc[tr, features])).fit()
    pred = m.predict(sm.add_constant(X[features], has_constant='add'))
    cf = pd.DataFrame({'served': X.gen_mwh, 'lat': np.exp(pred)}, index=X.index)
    a = cf.groupby(cf.index.year).sum()
    gap = (a.lat/a.served - 1)*100
    ins = gap.loc[[v for v in gap.index if 2016 <= v <= 2021]]
    return m, gap, float(ins.std()), float(ins.mean())

m0, gap0, floor0, mu0 = counterfactual(FEAT)
print(); print("="*92)
print("(B) STRUCTURAL COUNTERFACTUAL (OLS fitted 2016-2021, extrapolated)"); print("="*92)
print(f"in-sample R^2 = {m0.rsquared:.4f} | trend = {m0.params['t']*365.25*100:+.2f} %/yr")
print(f"placebo (2016-2021): mean {mu0:+.2f} pp, sd {floor0:.2f} pp  <- noise floor")
print(gap0.round(2).to_string())

# =====================================================================
print(); print("="*92)
print("(C) COULD THE GAP BE PRICE / ACTIVITY RESPONSE RATHER THAN SUPPRESSION?")
print("="*92)
m1, gap1, floor1, mu1 = counterfactual(FEAT + ['log_tariff_real'])
el = float(m1.params['log_tariff_real'])
print(f"own-price elasticity on the real tariff = {el:+.3f} "
      f"(t = {m1.tvalues['log_tariff_real']:+.2f})")
print(f"placebo sd with tariff control = {floor1:.2f} pp")
print(gap1.round(2).to_string())

m2, gap2, floor2, mu2 = counterfactual(FEAT + ['gdp_growth'])
m3, gap3, floor3, mu3 = counterfactual(FEAT + ['log_tariff_real', 'gdp_growth'])
print(f"\nActivity control: GDP-growth coef = {m2.params['gdp_growth']:+.4f} "
      f"(t = {m2.tvalues['gdp_growth']:+.2f}); placebo sd {floor2:.2f} pp")
print("gap with GDP control:");  print(gap2.round(2).to_string())
print("gap with BOTH controls:"); print(gap3.round(2).to_string())

print("\nBound on the price contribution, using literature short-run elasticities:")
eff = {}
for Y in (2022, 2023, 2024, 2025):
    dlog = np.log(TARIFF_REAL[Y]/TARIFF_REAL[2021])
    eff[int(Y)] = {str(e): float(e*dlog*100) for e in (-0.10, -0.20, -0.30)}
    s = '  '.join(f'e={e:+.2f}: {e*dlog*100:+.2f} pp' for e in (-0.10, -0.20, -0.30))
    print(f"  {Y}: real tariff {dlog*100:+5.1f}% -> demand effect  {s}")

# =====================================================================
print(); print("="*92); print("(D) CONSISTENCY OF THE TWO INDEPENDENT ROUTES"); print("="*92)
rows = []
for Y in (2022, 2023, 2024, 2025):
    lo, hi = bounds.loc[Y, 'lo_pct'], bounds.loc[Y, 'hi_pct']
    g, g1 = gap0.loc[Y], gap1.loc[Y]
    inside = (lo - floor0) <= g <= (hi + floor0)
    rows.append(dict(year=int(Y), bound_lo=float(lo), bound_hi=float(hi),
                     structural=float(g), structural_tariff_ctrl=float(g1),
                     noise_floor=floor0, inside=bool(inside)))
    print(f"  {Y}: bounds [{lo:5.2f}, {hi:5.2f}] | structural {g:+5.2f} "
          f"(tariff-ctrl {g1:+5.2f}) +-{floor0:.2f} | inside = {inside}")

json.dump({'alpha_min': ALPHA_MIN, 'tariff_real': TARIFF_REAL,
           'real_tariff_change_2021_2025_pct': chg_real*100,
           'elasticity_estimate': el, 'placebo_sd_pp': floor0,
           'bounds': bounds.round(3).to_dict(),
           'gap_no_control': gap0.round(3).to_dict(),
           'gap_tariff_control': gap1.round(3).to_dict(),
           'gap_gdp_control': gap2.round(3).to_dict(),
           'gap_both_controls': gap3.round(3).to_dict(),
           'placebo_sd_gdp': floor2, 'placebo_sd_both': floor3,
           'price_effect_pp': eff, 'consistency': rows},
          open('results/identification.json', 'w'), indent=2, default=str)
bounds.round(3).to_csv('results/bounds.csv')
pd.DataFrame({'gap': gap0, 'gap_tariff_ctrl': gap1, 'gap_gdp_ctrl': gap2,
              'gap_both_ctrl': gap3}).round(3).to_csv('results/structural_gap.csv')
print("\nsaved -> results/identification.json, bounds.csv, structural_gap.csv")
