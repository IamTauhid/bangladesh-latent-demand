"""Stage 4: master daily design matrix — calendar, Islamic calendar, weather, regime covariates."""
import pandas as pd, numpy as np

d = pd.read_csv('data/daily.csv', parse_dates=['date'], index_col='date')
w = pd.read_csv('data/weather_national.csv', parse_dates=['date'], index_col='date')
X = d.join(w, how='left')

# ---------- calendar ----------
X['dow']   = X.index.dayofweek                 # 0=Mon
X['doy']   = X.index.dayofyear
X['month'] = X.index.month
X['year']  = X.index.year
X['t']     = (X.index - X.index.min()).days
X['is_friday']   = (X.dow == 4).astype(int)    # BD weekend = Fri+Sat
X['is_saturday'] = (X.dow == 5).astype(int)
X['is_weekend']  = X.is_friday | X.is_saturday
for k in (1,2,3):
    X[f'sin{k}'] = np.sin(2*np.pi*k*X.doy/365.25)
    X[f'cos{k}'] = np.cos(2*np.pi*k*X.doy/365.25)

# ---------- Islamic calendar (Bangladesh observed dates) ----------
RAMADAN = {2015:'2015-06-19',2016:'2016-06-07',2017:'2017-05-28',2018:'2018-05-17',
           2019:'2019-05-07',2020:'2020-04-25',2021:'2021-04-14',2022:'2022-04-03',
           2023:'2023-03-24',2024:'2024-03-12',2025:'2025-03-02',2026:'2026-02-19'}
EID_FITR= {2015:'2015-07-18',2016:'2016-07-07',2017:'2017-06-26',2018:'2018-06-16',
           2019:'2019-06-05',2020:'2020-05-25',2021:'2021-05-14',2022:'2022-05-03',
           2023:'2023-04-22',2024:'2024-04-11',2025:'2025-03-31',2026:'2026-03-20'}
EID_ADHA= {2015:'2015-09-25',2016:'2016-09-13',2017:'2017-09-02',2018:'2018-08-22',
           2019:'2019-08-12',2020:'2020-08-01',2021:'2021-07-21',2022:'2022-07-10',
           2023:'2023-06-29',2024:'2024-06-17',2025:'2025-06-07',2026:'2026-05-27'}
X['is_ramadan']=0; X['eid_prox']=0.0
for yr,s in RAMADAN.items():
    s=pd.Timestamp(s); X.loc[(X.index>=s)&(X.index<s+pd.Timedelta(days=30)),'is_ramadan']=1
for tbl in (EID_FITR, EID_ADHA):
    for yr,s in tbl.items():
        s=pd.Timestamp(s)
        win=X.index[(X.index>=s-pd.Timedelta(days=3))&(X.index<=s+pd.Timedelta(days=4))]
        X.loc[win,'eid_prox'] = np.maximum(X.loc[win,'eid_prox'],
                                           1-np.abs((win-s).days)/5.0)

# ---------- fixed national holidays ----------
FIXED=[(2,21),(3,26),(4,14),(5,1),(8,15),(12,16),(12,25)]
X['is_holiday']=0
for m,dd in FIXED:
    X.loc[(X.month==m)&(X.index.day==dd),'is_holiday']=1

# ---------- exogenous shocks (documented) ----------
def span(a,b): return (X.index>=a)&(X.index<=b)
X['covid_lockdown'] = (span('2020-03-26','2020-05-30')|span('2021-04-05','2021-08-10')).astype(int)
X['july_unrest']    =  span('2024-07-18','2024-08-11').astype(int)   # curfew + internet shutdown

# ---------- regime / capacity covariates (data-detected, see §III-C) ----------
X['reg_fuelcrisis'] = span('2022-07-19','2025-06-30').astype(int)    # LNG spot halt -> rationing era
X['cap_coal_p1']    = (X.index>=pd.Timestamp('2023-01-01')).astype(int)   # Payra full + Rampal U1
X['cap_coal_p2']    = (X.index>=pd.Timestamp('2023-08-01')).astype(int)   # Rampal U2
X['rep_adani']      = (X.index>=pd.Timestamp('2024-08-28')).astype(int)   # reporting-schema break
X['cap_wind']       = (X.index>=pd.Timestamp('2023-06-01')).astype(int)

# ---------- weather-derived ----------
X['CDD_ma7']  = X.CDD_at.rolling(7,min_periods=1).mean()
X['CDD_lag1'] = X.CDD_at.shift(1)
X['T_ma3']    = X.T2M.rolling(3,min_periods=1).mean()
X['dT']       = X.T2M.diff()
X['rain7']    = X.PRECTOTCORR.rolling(7,min_periods=1).sum()

# ---------- autoregressive ----------
for L in (1,2,3,7,14,364):
    X[f'gen_lag{L}'] = X.gen_mwh.shift(L)
X['gen_ma7']  = X.gen_mwh.shift(1).rolling(7).mean()
X['gen_ma28'] = X.gen_mwh.shift(1).rolling(28).mean()
X['gen_std7'] = X.gen_mwh.shift(1).rolling(7).std()

X = X.dropna(subset=['gen_lag364','CDD_at'])
X.to_csv('data/features.csv')
print('design matrix:', X.shape, X.index.min().date(), '->', X.index.max().date())

# quick signal check
import scipy.stats as st
y = np.log(X.gen_mwh)
print('\n--- univariate correlation with log(daily energy), detrended ---')
res = y - pd.Series(np.polyval(np.polyfit(X.t,y,2), X.t), index=X.index)
for c in ['CDD_at','CDD','T2M','RH2M','THI','apparent_temp','is_friday','is_ramadan','eid_prox','covid_lockdown','july_unrest']:
    print(f'  {c:16s} r = {st.pearsonr(X[c],res)[0]:+.3f}')
