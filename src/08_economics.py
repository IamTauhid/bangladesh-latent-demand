"""
Stage 8 - Loss-adjusted economic layer.

Chain:  latent demand  ->  dispatch composition  ->  fuel-wise generation cost
        ->  transmission & distribution loss  ->  delivered energy
        ->  revenue at retail tariff  ->  subsidy gap.

Plus: cost of unserved energy at a range of VoLL, and the counterfactual cost of
having served the suppressed demand recovered by the L1 censored model.

All monetary parameters are exogenous published values (BPDB/BERC/press), NOT
estimated from the PGCB telemetry; every one is swept in the sensitivity analysis.
"""
import numpy as np, pandas as pd, json

# ---------------------------------------------------------------- parameters
# Per-unit generation / purchase cost, BDT per kWh (FY2024-25 basis).
COST = {                       # source: BPDB FY25 disclosures & press reporting
    'gas'                 :  7.09,
    'liquid_fuel'         : 27.39,
    'coal'                : 13.20,
    'hydro'               :  1.20,   # Kaptai, fully depreciated public asset
    'solar'               :  9.00,   # utility-scale IPP band
    'wind'                :  9.50,
    'india_bheramara_hvdc':  9.34,   # non-Adani Indian suppliers
    'india_tripura'       :  9.34,
    'india_adani'         : 14.86,   # FY25 realised average
    'nepal'               :  8.35,   # FY25 realised average
}
COST_UNC = {k: 0.15 for k in COST}          # +/-15% sweep unless overridden
COST_UNC.update({'gas': 0.25, 'liquid_fuel': 0.20})   # LNG/HFO price volatility

# Losses, fraction of energy. Verified FY values; linearly interpolated between.
TX_LOSS = {2016:0.0285, 2019:0.0287, 2022:0.0289, 2023:0.0307, 2024:0.0313, 2026:0.0320}
DX_LOSS = {2016:0.0950, 2019:0.0880, 2022:0.0800, 2023:0.0765, 2024:0.0725, 2026:0.0700}
TARIFF  = {2016: 6.10, 2019: 6.85, 2022: 7.13, 2023: 7.72, 2024: 8.25, 2026: 8.95}  # avg retail, BDT/kWh
VOLL    = [40.0, 90.0, 180.0]               # BDT/kWh (~US$0.33 / 0.74 / 1.48)
USD     = 122.0                             # BDT per USD, 2025 average
CRORE   = 1e7

FUELS = list(COST)

def interp(tbl, years):
    s = pd.Series(tbl).sort_index()
    return pd.Series(np.interp(years, s.index, s.values), index=years)

# ---------------------------------------------------------------- historical
d = pd.read_csv('data/daily.csv', parse_dates=['date'], index_col='date')
yr = d.groupby(d.index.year)
gen = yr[[f+'_mwh' for f in FUELS]].sum()
gen.columns = FUELS
gen['total'] = gen.sum(axis=1)
gen['ls_mwh'] = yr.ls_mwh.sum()
gen['days']   = yr.size()

# annualise partial years (2015, 2026) so trajectories are comparable
scale = 365.0/gen.days
ann = gen[FUELS+['total','ls_mwh']].mul(scale, axis=0)

years = ann.index.values
txl, dxl, tar = interp(TX_LOSS, years), interp(DX_LOSS, years), interp(TARIFF, years)

cost_by_fuel = ann[FUELS].mul(pd.Series(COST), axis=1) * 1000        # MWh * BDT/kWh * 1000
tot_cost = cost_by_fuel.sum(axis=1)
wacog    = tot_cost/(ann.total*1000)                                  # BDT/kWh generated

delivered = ann.total*(1-txl)*(1-dxl)
revenue   = delivered*1000*tar
subsidy   = tot_cost - revenue

E = pd.DataFrame({
    'gen_TWh'        : ann.total/1e6,
    'delivered_TWh'  : delivered/1e6,
    'loss_TWh'       : (ann.total-delivered)/1e6,
    'tx_loss_%'      : txl*100,
    'dx_loss_%'      : dxl*100,
    'WACOG_BDT_kWh'  : wacog,
    'tariff_BDT_kWh' : tar,
    'gen_cost_crore' : tot_cost/CRORE,
    'revenue_crore'  : revenue/CRORE,
    'subsidy_crore'  : subsidy/CRORE,
    'subsidy_musd'   : subsidy/USD/1e6,
    'ENS_GWh'        : ann.ls_mwh/1e3,
})
E['cost_of_losses_crore'] = (ann.total-delivered)*1000*wacog/CRORE
for v in VOLL:
    E[f'ENS_cost_crore@{int(v)}'] = ann.ls_mwh*1000*v/CRORE

print("="*100); print("HISTORICAL LOSS-ADJUSTED ECONOMICS (annualised)"); print("="*100)
pd.set_option('display.width', 240)
print(E.round(2).to_string())

# calibration check against BPDB's published FY25 average purchase cost
w25 = float(wacog.loc[2025])
print(f"\n[calibration] model WACOG 2025 = {w25:.2f} BDT/kWh vs BPDB published FY25 "
      f"average purchase cost 12.10 BDT/kWh -> deviation {(w25/12.10-1)*100:+.1f}%")

# ---------------------------------------------------------------- fuel-wise cost shares
share_cost = cost_by_fuel.div(tot_cost, axis=0)*100
share_en   = ann[FUELS].div(ann.total, axis=0)*100
print("\n" + "="*100); print("ENERGY SHARE (%) vs COST SHARE (%) - the affordability wedge"); print("="*100)
cmp = pd.concat({'energy_%': share_en.round(1), 'cost_%': share_cost.round(1)}, axis=1)
print(cmp[[('energy_%',f) for f in ['gas','coal','liquid_fuel','india_adani']] +
          [('cost_%',f)  for f in ['gas','coal','liquid_fuel','india_adani']]].to_string())

# ---------------------------------------------------------------- counterfactual mixes
print("\n" + "="*100); print("COUNTERFACTUAL: cost of the 2021 mix vs the realised mix"); print("="*100)
mix21 = share_en.loc[2021]/100
rows = []
for y in [2022, 2023, 2024, 2025]:
    tot = ann.total.loc[y]
    actual = float(tot_cost.loc[y])
    cf = float((tot*mix21*pd.Series(COST)).sum()*1000)
    rows.append(dict(year=y, actual_crore=actual/CRORE, cf2021mix_crore=cf/CRORE,
                     delta_crore=(actual-cf)/CRORE, delta_musd=(actual-cf)/USD/1e6,
                     delta_pct=(actual/cf-1)*100))
CF = pd.DataFrame(rows).set_index('year')
print(CF.round(1).to_string())

# ---------------------------------------------------------------- sensitivity
print("\n" + "="*100); print("SENSITIVITY OF 2025 SUBSIDY GAP TO COST PARAMETERS (one-at-a-time)"); print("="*100)
base = float(subsidy.loc[2025]/CRORE)
sens = []
for f in FUELS:
    for sgn in (+1, -1):
        c2 = dict(COST); c2[f] = COST[f]*(1+sgn*COST_UNC[f])
        tc = float((ann.loc[2025, FUELS]*pd.Series(c2)).sum()*1000)
        sens.append(dict(param=f, direction=f"{sgn*COST_UNC[f]*100:+.0f}%",
                         subsidy_crore=(tc-float(revenue.loc[2025]))/CRORE))
S = pd.DataFrame(sens)
S['delta_vs_base'] = S.subsidy_crore - base
piv = S.pivot(index='param', columns='direction', values='delta_vs_base')
piv['abs_swing'] = piv.abs().max(axis=1)
print(f"base 2025 subsidy gap = {base:,.0f} crore BDT ({base*CRORE/USD/1e6:,.0f} M USD)")
print(piv.sort_values('abs_swing', ascending=False).round(1).to_string())

E.round(3).to_csv('results/l3_economics.csv')
CF.round(2).to_csv('results/l3_counterfactual.csv')
piv.round(2).to_csv('results/l3_sensitivity.csv')
json.dump({'cost_params': COST, 'tx_loss': TX_LOSS, 'dx_loss': DX_LOSS,
           'tariff': TARIFF, 'voll': VOLL, 'usd_bdt': USD,
           'wacog_2025_model': w25, 'wacog_2025_published': 12.10},
          open('results/l3_params.json', 'w'), indent=2)
print("\nsaved -> results/l3_economics.csv, l3_counterfactual.csv, l3_sensitivity.csv, l3_params.json")
