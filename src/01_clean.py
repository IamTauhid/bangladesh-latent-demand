"""Stage 1: physics-informed cleaning + hourly->daily aggregation of PGCB records."""
import pandas as pd, numpy as np, json, pathlib

RAW = "pgcb_V1_raw.csv"
OUT = pathlib.Path("data")
FUELS = ['gas','liquid_fuel','coal','hydro','solar','wind']
IMPORTS = ['india_bheramara_hvdc','india_tripura','india_adani','nepal']
ALL_SRC = FUELS + IMPORTS

# Physical ceilings: BD installed capacity reached ~28 GW (2025); grid peak ever ~17.2 GW.
CEIL = {'generation_mw':20000,'demand_mw':25000,'load_shedding':8000,
        'gas':12000,'liquid_fuel':8000,'coal':6000,'hydro':300,'solar':1500,'wind':200,
        'india_bheramara_hvdc':1200,'india_tripura':250,'india_adani':1600,'nepal':100}

df = pd.read_csv(RAW, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
n0 = len(df)
audit = {'raw_rows': n0}

# ---- 1. flag implausible values (data-entry / scraping corruption), set to NaN ----
viol = pd.Series(False, index=df.index)
for c, hi in CEIL.items():
    bad = df[c] > hi
    audit[f'ceiling_violations_{c}'] = int(bad.sum())
    viol |= bad.fillna(False)
    df.loc[bad, c] = np.nan
audit['rows_with_any_ceiling_violation'] = int(viol.sum())

# ---- 2. conservation residual: sum(sources) should equal generation ----
df['src_sum'] = df[ALL_SRC].sum(axis=1, min_count=1)
resid = df['src_sum'] - df['generation_mw']
df['cons_resid'] = resid
audit['conservation_median_abs_resid_MW'] = float(resid.abs().median())
audit['conservation_within_50MW_pct'] = float((resid.abs() < 50).mean()*100)
# reject rows where the energy balance is grossly violated (>10% of generation)
bad_bal = (resid.abs() > 0.10*df['generation_mw'].clip(lower=1))
audit['rows_failing_energy_balance'] = int(bad_bal.sum())
df.loc[bad_bal, ['generation_mw']+ALL_SRC] = np.nan

# ---- 3. reindex to a strict hourly grid; interpolate short gaps only (<=3 h) ----
df = df.drop_duplicates('datetime').set_index('datetime')
full = pd.date_range(df.index.min().floor('h'), df.index.max().ceil('h'), freq='h')
peak_flag = df['remarks'].reindex(full)
num = df[['generation_mw','demand_mw','load_shedding']+ALL_SRC].reindex(full)
audit['hourly_grid_slots'] = len(full)
audit['missing_before_interp_pct'] = float(num['generation_mw'].isna().mean()*100)
num = num.interpolate(method='time', limit=3, limit_area='inside')
audit['missing_after_interp_pct'] = float(num['generation_mw'].isna().mean()*100)

# renewables/imports absent early in the record are structural zeros, not gaps
for c in ['solar','wind','india_adani','nepal']:
    num[c] = num[c].fillna(0.0)
num[ALL_SRC] = num[ALL_SRC].fillna(0.0)

hourly = num.copy()
hourly['is_peak'] = peak_flag.values
hourly.index.name = 'datetime'
hourly.to_csv(OUT/'hourly_clean.csv')

# ---- 4. daily aggregation: MW hourly -> MWh/day (1 h steps) ----
d = hourly.dropna(subset=['generation_mw'])
day = pd.DataFrame({
    'gen_mwh'      : d['generation_mw'].resample('D').sum(),
    'gen_hours'    : d['generation_mw'].resample('D').count(),
    'peak_mw'      : d['generation_mw'].resample('D').max(),
    'min_mw'       : d['generation_mw'].resample('D').min(),
    'mean_mw'      : d['generation_mw'].resample('D').mean(),
    'demand_peak_mw': d['demand_mw'].resample('D').max(),
    'demand_mwh'   : d['demand_mw'].resample('D').sum(),
    'ls_mwh'       : d['load_shedding'].resample('D').sum(),
    'ls_max_mw'    : d['load_shedding'].resample('D').max(),
    'ls_hours'     : d['load_shedding'].gt(0).resample('D').sum(),
})
for c in ALL_SRC:
    day[c+'_mwh'] = d[c].resample('D').sum()

day = day[day.gen_hours >= 20]           # keep days with >=20 valid hours
day['load_factor'] = day.mean_mw/day.peak_mw
day['censored'] = (day.ls_hours > 0).astype(int)
audit['daily_rows_kept'] = int(len(day))
audit['daily_censored_pct'] = float(day.censored.mean()*100)
day.index.name='date'
day.to_csv(OUT/'daily.csv')

json.dump(audit, open('results/clean_audit.json','w'), indent=2)
print(json.dumps(audit, indent=2))
print("\ndaily range:", day.index.min().date(), "->", day.index.max().date(), "| n =", len(day))
