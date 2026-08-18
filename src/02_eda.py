"""Stage 2: seasonality, structural breaks, censoring regimes."""
import pandas as pd, numpy as np, json
pd.set_option('display.width',250)
d = pd.read_csv('data/daily.csv', parse_dates=['date'], index_col='date')
FU = ['gas','liquid_fuel','coal','hydro','solar','wind','india_bheramara_hvdc','india_tripura','india_adani','nepal']
MWH = [f+'_mwh' for f in FU]

print("="*78); print("A. ANNUAL GROWTH OF SERVED ENERGY"); print("="*78)
y = d.groupby(d.index.year).agg(gwh=('gen_mwh',lambda s: s.sum()/1000), days=('gen_mwh','size'),
                                peak=('peak_mw','max'), lf=('load_factor','mean'))
y['gwh_annualised'] = (y.gwh/y.days*365).round(0)
y['yoy_%'] = y.gwh_annualised.pct_change().mul(100).round(1)
print(y.round(2))

print(); print("="*78); print("B. SEASONALITY (monthly index, load normalised by year mean)"); print("="*78)
d['yr']=d.index.year; d['mo']=d.index.month
d['norm'] = d.gen_mwh / d.groupby('yr').gen_mwh.transform('mean')
piv = d.pivot_table(index='mo', columns='yr', values='norm', aggfunc='mean')
print(piv.round(3))
print("\nmonthly seasonal index (mean over 2016-2025):")
si = piv.loc[:, 2016:2025].mean(axis=1)
print(si.round(3).to_string())
print(f"summer/winter amplitude = {si.max()/si.min():.3f} (peak M{si.idxmax()} / trough M{si.idxmin()})")

print(); print("="*78); print("C. CENSORING REGIME (load shedding)"); print("="*78)
c = d.groupby(d.index.year).agg(days=('censored','size'), cens_days=('censored','sum'),
        ls_gwh=('ls_mwh',lambda s:s.sum()/1000), ls_max=('ls_max_mw','max'),
        ls_hrs=('ls_hours','sum'))
c['cens_day_%']=(c.cens_days/c.days*100).round(1)
c['ENS_%_of_served']=(c.ls_gwh/(y.gwh)*100).round(2)
print(c.round(2))

print(); print("="*78); print("D. FUEL MIX SHARES (% of annual generation)"); print("="*78)
fy = d.groupby(d.index.year)[MWH].sum()
sh = fy.div(fy.sum(axis=1),axis=0).mul(100)
print(sh.round(2))

print(); print("="*78); print("E. STRUCTURAL BREAK SCAN — coal share, CUSUM-style"); print("="*78)
m = d.groupby(pd.Grouper(freq='MS'))[MWH].sum()
msh = m.div(m.sum(axis=1),axis=0)
for f in ['coal_mwh','gas_mwh','india_adani_mwh','liquid_fuel_mwh']:
    s = msh[f].dropna()
    # largest single-month jump in 12-month rolling mean
    r = s.rolling(12).mean()
    jump = r.diff(12).abs()
    print(f"{f:22s} first month >1% share: {s[s>0.01].index.min().date() if (s>0.01).any() else '-'} "
          f"| max 12-mo share change: {jump.max()*100:5.1f} pp ending {jump.idxmax().date() if jump.notna().any() else '-'}")

print(); print("="*78); print("F. WEEKLY / RAMADAN-EID EFFECT"); print("="*78)
d['dow']=d.index.dayofweek
print("mean normalised load by weekday (0=Mon..6=Sun):")
print(d.groupby('dow').norm.mean().round(4).to_string())
