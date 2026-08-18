"""Stage 10 - publication figures (IEEE two-column, vector PDF + PNG)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates, json, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('paper/figs', exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'lines.linewidth': 1.0,
    'axes.grid': True, 'grid.alpha': 0.25,  'savefig.dpi': 400,
    'axes.spines.top': False, 'axes.spines.right': False,
})
W1, W2 = 3.5, 7.16                     # IEEE single / double column width (inches)
C = dict(gas='#3B6EA5', liquid='#C4562F', coal='#4A4A4A', renew='#3E8E5A',
         imp='#8B5E9E', obs='#222222', lat='#C4562F', acc='#3B6EA5')

def save(fig, name):
    fig.savefig(f'paper/figs/{name}.pdf'); fig.savefig(f'paper/figs/{name}.png')
    plt.close(fig); print('  wrote', name)

d = pd.read_csv('data/daily.csv', parse_dates=['date'], index_col='date')
h = pd.read_csv('data/hourly_clean.csv', parse_dates=['datetime'], index_col='datetime').dropna(subset=['generation_mw'])
X = pd.read_csv('data/features.csv', parse_dates=['date'], index_col='date')
FU = ['gas','liquid_fuel','coal','hydro','solar','wind',
      'india_bheramara_hvdc','india_tripura','india_adani','nepal']

# ---------------------------------------------------------------- Fig 2 : series + regime
print('figures:')
fig, ax = plt.subplots(2, 1, figsize=(W2, 3.1), sharex=True,
                       gridspec_kw=dict(height_ratios=[2, 1], hspace=0.30))
s = d.gen_mwh/1e3
ax[0].plot(s.index, s, lw=0.35, color=C['obs'], alpha=0.55)
ax[0].plot(s.index, s.rolling(30, center=True).mean(), lw=1.3, color=C['acc'], label='30-day mean')
ax[0].axvspan(pd.Timestamp('2022-07-19'), pd.Timestamp('2025-06-30'),
              color='#C4562F', alpha=0.10, lw=0, label='rationing regime')
ax[0].set_ylabel('Daily energy (GWh)'); ax[0].legend(loc='upper left', frameon=False, ncol=2)
ax[0].set_title('(a) Daily served energy, Bangladesh national grid', loc='left')

m = d.groupby(pd.Grouper(freq='MS'))[[f+'_mwh' for f in FU]].sum()
sh = m.div(m.sum(axis=1), axis=0)*100
grp = {'Gas': ['gas_mwh'], 'Liquid fuel': ['liquid_fuel_mwh'], 'Coal': ['coal_mwh'],
       'Renewables': ['hydro_mwh','solar_mwh','wind_mwh'],
       'Imports': ['india_bheramara_hvdc_mwh','india_tripura_mwh','india_adani_mwh','nepal_mwh']}
G = pd.DataFrame({k: sh[v].sum(axis=1) for k, v in grp.items()})
ax[1].stackplot(G.index, *[G[k] for k in G], colors=[C['gas'],C['liquid'],C['coal'],C['renew'],C['imp']],
                labels=list(G), lw=0)
ax[1].set_ylim(0, 100); ax[1].set_ylabel('Mix (%)')
ax[1].legend(loc='lower left', ncol=5, frameon=False, columnspacing=0.8, handlelength=1.2)
ax[1].set_title('(b) Generation mix', loc='left')
ax[1].xaxis.set_major_locator(mdates.YearLocator(2)); ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
save(fig, 'fig2_series_mix')

# ---------------------------------------------------------------- Fig 3 : censoring diagnostics
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05), constrained_layout=True)
pre = h[h.index.year < 2022]; post = h[h.index.year.isin([2022,2023,2024])]
ax[0].scatter(pre.generation_mw/1e3, pre.demand_mw/1e3, s=0.4, alpha=0.10, color=C['acc'], rasterized=True, label='2015-2021')
ax[0].scatter(post.generation_mw/1e3, post.demand_mw/1e3, s=0.4, alpha=0.10, color=C['liquid'], rasterized=True, label='2022-2024')
lim = [2, 18]; ax[0].plot(lim, lim, 'k--', lw=0.7)
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('Generation (GW)'); ax[0].set_ylabel('Reported "demand" (GW)')
ax[0].set_title('(a) Reported demand is an\naccounting identity', loc='left')
lg = ax[0].legend(loc='upper left', frameon=False, markerscale=8)
for lh in lg.legend_handles: lh.set_alpha(1)

p = post.groupby(post.index.hour).apply(lambda x: (x.load_shedding > 0).mean()*100)
ax[1].bar(p.index, p.values, color=C['liquid'], width=0.8, lw=0)
ax[1].axhline(p.mean(), ls='--', lw=0.8, color='k')
ax[1].text(0.5, p.mean()+1.5, f'mean {p.mean():.0f}%', fontsize=6.5)
ax[1].set_xlabel('Hour of day'); ax[1].set_ylabel('Hours with shedding (%)')
ax[1].set_title('(b) Rationing is flat across\nthe day, not peak-clipped', loc='left')

for yr, col, ls in [(2021, C['acc'], '-'), (2023, C['liquid'], '-')]:
    g = np.sort(h[h.index.year == yr].generation_mw.values)[::-1]/1e3
    ax[2].plot(np.arange(len(g))/len(g)*100, g, color=col, ls=ls, label=str(yr))
ax[2].set_xlabel('Duration (% of hours)'); ax[2].set_ylabel('Load (GW)')
ax[2].set_title('(c) Load-duration curve shows\nno truncation', loc='left')
ax[2].legend(frameon=False)
save(fig, 'fig3_censoring_diagnostics')

# ---------------------------------------------------------------- Fig 4 : seasonality & weather
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0), constrained_layout=True)
X2 = X.copy(); X2['yr'] = X2.index.year; X2['mo'] = X2.index.month
X2['norm'] = X2.gen_mwh/X2.groupby('yr').gen_mwh.transform('mean')
piv = X2.pivot_table(index='mo', columns='yr', values='norm', aggfunc='mean').loc[:, 2016:2025]
im = ax[0].imshow(piv.values, aspect='auto', cmap='RdYlBu_r', vmin=0.72, vmax=1.22,
                  extent=[2015.5, 2025.5, 12.5, 0.5])
ax[0].set_yticks(range(1, 13)); ax[0].set_yticklabels(list('JFMAMJJASOND'), fontsize=6)
ax[0].set_xticks(range(2016, 2026, 3)); ax[0].set_ylabel('Month')
ax[0].set_title('(a) Seasonal index', loc='left'); ax[0].grid(False)
plt.colorbar(im, ax=ax[0], fraction=0.045, pad=0.03).ax.tick_params(labelsize=6)

sc = ax[1].scatter(X.CDD_at, X.gen_mwh/1e3, c=X.index.year, s=1.2, cmap='viridis', rasterized=True)
ax[1].set_xlabel('Humidity-adjusted CDD (°C)'); ax[1].set_ylabel('Daily energy (GWh)')
ax[1].set_title('(b) Cooling response', loc='left')
plt.colorbar(sc, ax=ax[1], fraction=0.045, pad=0.03).ax.tick_params(labelsize=6)

dw = X2.groupby(X2.index.dayofweek).norm.mean()
ax[2].bar(range(7), dw.values, color=[C['liquid'] if i in (4,5) else C['acc'] for i in range(7)], lw=0)
ax[2].set_xticks(range(7)); ax[2].set_xticklabels(['M','T','W','T','F','S','S'])
ax[2].set_ylim(0.90, 1.03); ax[2].set_ylabel('Normalised load')
ax[2].set_title('(c) Weekly cycle\n(Fri–Sat weekend)', loc='left')
save(fig, 'fig4_seasonality')

# ---------------------------------------------------------------- Fig 5 : censoring recovery
try:
    r = pd.read_csv('results/exp1_synthetic_censoring.csv')
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    if 'dcg_bias' in r.columns:
        g = r.groupby('admit')[['dcg_bias','naive_bias']].mean()
        x = np.arange(len(g)); w = 0.36
        ax.bar(x-w/2, g.naive_bias, w, label='Censoring-blind', color=C['liquid'], lw=0)
        ax.bar(x+w/2, g.dcg_bias,  w, label='DCG (proposed)',  color=C['acc'],    lw=0)
        ax.set_xticks(x); ax.set_xticklabels([f'{v:.0%}' for v in g.index])
    else:
        p = r.pivot(index='admit', columns='model', values='bias_pct')
        x = np.arange(len(p)); w = 0.36
        ax.bar(x-w/2, p.iloc[:,1], w, label='Censoring-blind', color=C['liquid'], lw=0)
        ax.bar(x+w/2, p.iloc[:,0], w, label='DCG (proposed)',  color=C['acc'],    lw=0)
        ax.set_xticks(x); ax.set_xticklabels([f'{v:.0%}' for v in p.index])
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xlabel('Fraction of true shortfall admitted by the operator')
    ax.set_ylabel('Bias vs. true demand (%)')
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'fig5_censoring_recovery')
except Exception as e: print('  skip fig5:', e)

# ---------------------------------------------------------------- Fig 7 : economics
E = pd.read_csv('results/l3_economics.csv', index_col=0)
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05), constrained_layout=True)
ax[0].plot(E.index, E.WACOG_BDT_kWh, 'o-', ms=2.5, color=C['obs'], label='Cost of supply')
ax[0].plot(E.index, E.tariff_BDT_kWh, 's--', ms=2.5, color=C['renew'], label='Average tariff')
ax[0].fill_between(E.index, E.tariff_BDT_kWh, E.WACOG_BDT_kWh, color=C['liquid'], alpha=0.18)
ax[0].set_ylabel('BDT / kWh'); ax[0].legend(frameon=False)
ax[0].set_title('(a) Cost–tariff gap', loc='left')

ann = d.groupby(d.index.year)[[f+'_mwh' for f in FU]].sum()
share_en = ann.div(ann.sum(axis=1), axis=0)*100
COST = json.load(open('results/l3_params.json'))['cost_params']
cost = ann.mul(pd.Series({k+'_mwh': v for k, v in COST.items()}), axis=1)
share_c = cost.div(cost.sum(axis=1), axis=0)*100
for f, lab, col in [('liquid_fuel_mwh','Liquid fuel',C['liquid']), ('coal_mwh','Coal',C['coal']),
                    ('gas_mwh','Gas',C['gas'])]:
    ax[1].plot(share_en.index, share_en[f], '-', color=col, label=lab+' (energy)')
    ax[1].plot(share_c.index, share_c[f], '--', color=col, alpha=0.75, label=lab+' (cost)')
ax[1].set_ylabel('Share (%)'); ax[1].legend(frameon=False, ncol=1, fontsize=5.6)
ax[1].set_title('(b) Energy vs. cost share', loc='left')

ax[2].bar(E.index, E.subsidy_musd/1e3, color=C['gas'], lw=0, label='Subsidy gap')
ax[2].plot(E.index, E['ENS_cost_crore@90']*1e7/122/1e9, 'o-', ms=2.5, color=C['liquid'],
           label='Cost of unserved energy\n(VoLL 90 BDT/kWh)')
ax[2].set_ylabel('Billion USD'); ax[2].legend(frameon=False, fontsize=6)
ax[2].set_title('(c) Fiscal burden', loc='left')
for a in ax:
    a.set_xlabel('Year')
    a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
save(fig, 'fig7_economics')

print('\nfigures written to paper/figs/')


# ---------------------------------------------------------------- Fig 6 : identification
try:
    import json as _json
    R = _json.load(open('results/identification.json'))
    bnd = R['bounds']; g3 = R['gap_both_controls']; floor = R['placebo_sd_gdp']
    getf = lambda dd, y: float(dd.get(str(y), dd.get(y)))
    yrs = [2022, 2023, 2024, 2025]
    lo = [getf(bnd['lo_pct'], y) for y in yrs]
    hi = [getf(bnd['hi_pct'], y) for y in yrs]
    st = [getf(g3, y) for y in yrs]
    fig, ax = plt.subplots(figsize=(W1, 2.2), constrained_layout=True)
    x = np.arange(len(yrs))
    ax.bar(x, np.array(hi)-np.array(lo), bottom=lo, width=0.45,
           color=C['acc'], alpha=0.30, lw=0, label='Model-free bound')
    ax.errorbar(x, st, yerr=floor, fmt='o', ms=4, color=C['liquid'], lw=1.1,
                capsize=2.5, label='Structural (controlled)')
    ax.plot(x, lo, '_', ms=13, color=C['obs'], mew=1.2, label='Officially admitted')
    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set_ylabel('Suppressed demand (pp of served)')
    ax.set_xlabel('Year')
    ax.legend(frameon=False, fontsize=6, loc='upper right')
    ax.set_ylim(-1, 12)
    save(fig, 'fig6_identification')
except Exception as e: print('  skip fig6:', e)

# ---------------------------------------------------------------- Fig 8 : mix forecast
try:
    z = np.load('results/l2_preds.npz', allow_pickle=True)
    comp = list(z['comp']); act = z['actual']
    dts = pd.to_datetime(z['dates'], unit='us')
    key = [k for k in z.files if k.startswith('A_') and 'ILR' in k]
    key = key[0] if key else [k for k in z.files if k.startswith('A_C')][0]
    pred = z[key]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True, constrained_layout=True)
    cols = [C['gas'], C['liquid'], C['coal'], C['renew'], C['imp']]
    for A, t, axx in [(act, 'Actual', ax[0]), (pred, key[2:], ax[1])]:
        axx.stackplot(dts, *[A[:, i]*100 for i in range(len(comp))], colors=cols,
                      labels=[c.capitalize() for c in comp], lw=0)
        axx.set_ylim(0, 100); axx.set_title(f'({"ab"[axx is ax[1]]}) {t}', loc='left')
        axx.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        axx.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax[0].set_ylabel('Share of generation (%)')
    ax[1].legend(loc='lower left', ncol=5, frameon=False, columnspacing=0.7, handlelength=1.1)
    save(fig, 'fig8_mix_forecast')
except Exception as e: print('  skip fig8:', e)
