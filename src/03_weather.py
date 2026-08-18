"""Fetch NASA POWER daily weather for all 8 BD divisions; build population-weighted indices."""
import pandas as pd, numpy as np, urllib.request, json, time, pathlib

# division: (lat, lon, population millions, BBS 2022 census)
DIV = {
 'Dhaka':      (23.8103, 90.4125, 44.22),
 'Chattogram': (22.3569, 91.7832, 33.20),
 'Rajshahi':   (24.3745, 88.6042, 20.35),
 'Khulna':     (22.8456, 89.5403, 17.42),
 'Rangpur':    (25.7439, 89.2752, 17.61),
 'Sylhet':     (24.8949, 91.8687, 11.40),
 'Mymensingh': (24.7471, 90.4203, 12.37),
 'Barishal':   (22.7010, 90.3535,  9.10),
}
PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR"
START, END = "20150401", "20260308"
BASE = ("https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={PARAMS}&community=RE&start={START}&end={END}&format=JSON")

frames = {}
for name,(lat,lon,pop) in DIV.items():
    url = f"{BASE}&latitude={lat}&longitude={lon}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                js = json.load(r)
            break
        except Exception as e:
            print(f"  {name} attempt {attempt+1} failed: {e}"); time.sleep(5)
    else:
        raise SystemExit(f"failed: {name}")
    p = js['properties']['parameter']
    f = pd.DataFrame(p)
    f.index = pd.to_datetime(f.index, format='%Y%m%d')
    f = f.replace(-999.0, np.nan)
    frames[name] = f
    print(f"  {name:12s} n={len(f)}  T2M mean={f.T2M.mean():.2f}C  missing={f.isna().mean().mean()*100:.2f}%")

# population weights
w = pd.Series({k: v[2] for k,v in DIV.items()}); w /= w.sum()

panel = pd.concat(frames, axis=1)          # (division, variable)
panel.to_csv('data/weather_divisions.csv')

nat = pd.DataFrame(index=next(iter(frames.values())).index)
for var in ['T2M','T2M_MAX','T2M_MIN','RH2M','WS2M','ALLSKY_SFC_SW_DWN','PRECTOTCORR']:
    nat[var] = sum(frames[d][var]*w[d] for d in DIV)

# --- comfort / cooling-load indices ---
T, RH = nat.T2M, nat.RH2M
# Magnus dew point -> heat index proxy (Steadman apparent temperature, humid form)
a,b = 17.62, 243.12
gam = np.log(RH/100.0) + a*T/(b+T)
nat['dewpoint'] = b*gam/(a-gam)
e = (RH/100.0)*6.112*np.exp(17.67*T/(T+243.5))            # vapour pressure hPa
nat['apparent_temp'] = T + 0.33*e - 0.70*nat.WS2M - 4.00   # Australian AT
BASE_C = 22.0                                              # cooling base for BD (literature 21-24C)
nat['CDD'] = (nat.T2M - BASE_C).clip(lower=0)
nat['HDD'] = (BASE_C - nat.T2M).clip(lower=0)
nat['CDD_at'] = (nat.apparent_temp - BASE_C).clip(lower=0)  # humidity-adjusted CDD
nat['THI'] = 0.8*T + RH*(T-14.4)/100 + 46.4                 # temperature-humidity index
nat.index.name='date'
nat.to_csv('data/weather_national.csv')
print("\nnational pop-weighted series:", nat.shape, nat.index.min().date(), '->', nat.index.max().date())
print(nat[['T2M','RH2M','apparent_temp','CDD','CDD_at','THI']].describe().T.round(2))
