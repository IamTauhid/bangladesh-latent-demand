"""Stage 11 - emit LaTeX tables (booktabs) from results/ into paper/tables.tex."""
import pandas as pd, numpy as np, json, os

os.makedirs('paper', exist_ok=True)
out = []
def add(s): out.append(s)
BS = '\\\\'          # LaTeX row terminator

# ---------------------------------------------------------------- T1 dataset
d = pd.read_csv('data/daily.csv', parse_dates=['date'], index_col='date')
FU = ['gas','liquid_fuel','coal','hydro','solar','wind',
      'india_bheramara_hvdc','india_tripura','india_adani','nepal']
y = d.groupby(d.index.year)
ann = (y.gen_mwh.sum()/1e6)/y.size()*365
cens = y.censored.mean()*100
ens = y.ls_mwh.sum()/1e3
gen = y[[f+'_mwh' for f in FU]].sum(); sh = gen.div(gen.sum(axis=1), axis=0)*100

add(r"""\begin{table*}[!t]
\caption{Annual profile of the Bangladesh national grid. The partial years 2015
(from 17 April) and 2026 (to 8 March) are omitted: annualising 68 winter days of a
September-peaking grid produces a meaningless year-on-year figure. ENS = energy not
served, as admitted by the operator.}
\label{tab:annual}
\centering\small
\begin{tabular}{lrrrrrrr}
\toprule
& \multicolumn{2}{c}{Energy} & \multicolumn{2}{c}{Rationing} & \multicolumn{3}{c}{Mix (\%)}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}
Year & TWh & YoY\% & Shed days\% & ENS (GWh) & Gas & Coal & Liquid \\
\midrule""")
prev = None
FULL = [v for v in ann.index if v not in (2015, 2026)]   # 2015/2026 are partial years
for yr in FULL:
    yoy = '--' if prev is None else f'{(ann[yr]/prev-1)*100:+.1f}'
    prev = ann[yr]
    add(f'{yr} & {ann[yr]:.1f} & {yoy} & {cens[yr]:.1f} & {ens[yr]:,.0f} & '
        f'{sh.loc[yr,"gas_mwh"]:.1f} & {sh.loc[yr,"coal_mwh"]:.1f} & '
        f'{sh.loc[yr,"liquid_fuel_mwh"]:.1f} ' + BS)
add(r'\bottomrule\end{tabular}\end{table*}')

# ---------------------------------------------------------------- T2 recovery
try:
    r = pd.read_csv('results/exp1c_interval.csv')
    r = r[np.isfinite(r.bias)]
    add(r"""\begin{table*}[!t]
\caption{Controlled synthetic-censoring recovery against \emph{known} truth. The
uncensored 2016--2021 era is artificially censored with the empirical 2022--2024
rationing process; ``admitted'' is the fraction of the true shortfall the operator
reports. ``Correction'' is a model's bias less the censoring-blind bias, and should
equal the true suppression. Neither specification achieves it. The Tobit correction is
near-constant regardless of the truth. The interval-censored correction is far smaller
in magnitude (mean absolute calibration error 2.25 to 0.65~pp) but, as the extended
sweep of Sec.~\ref{sec:tobitfail} shows, it responds to curtailment depth and not to
concealment: this sub-table varies only the admission rate, which is the dimension it
cannot see. That is why the point estimator is withdrawn in favour of the bounds in
Table~\ref{tab:ident}. Mean over 3 seeds; the 2020 holdout diverged numerically for
all specifications and is omitted.}
\label{tab:recovery}
\centering\small
\begin{tabular}{lrrrr}
\toprule
Holdout & Admitted & True supp. & \multicolumn{2}{c}{Correction applied (pp)}\\
\cmidrule(lr){4-5}
(train $n$) & (\%) & (pp) & Tobit & ICG \\
\midrule""")
    errs = {'tobit': [], 'interval': []}
    for (hl, tn, ad), g in r.groupby(['holdout', 'train_n', 'admit'], sort=True):
        gg = g.set_index('model').bias
        if not {'blind', 'tobit', 'interval'} <= set(gg.index): continue
        b = gg['blind']; true = g.true_suppression_pct.iloc[0]
        ct, ci = gg['tobit']-b, gg['interval']-b
        errs['tobit'].append(abs(ct-true)); errs['interval'].append(abs(ci-true))
        add(f'{hl} ({tn}) & {ad*100:.0f} & {true:+.2f} & {ct:+.2f} & '
            r'\textbf{' + f'{ci:+.2f}' + '} ' + BS)
    add(r'\midrule')
    add('Mean abs.\\ calibration error & & & '
        f'{np.mean(errs["tobit"]):.2f} & ' + r'\textbf{' +
        f'{np.mean(errs["interval"]):.2f}' + '} ' + BS)
    add(r'\bottomrule\end{tabular}\end{table*}')
except Exception as e: add(f'% recovery table unavailable: {e}')

# ---------------------------------------------------------------- T2b latent demand
# WITHDRAWN. This table reported a point estimate from the interval-censored
# likelihood. Sec. VI-A shows that point is not identified (the applied correction
# is anti-correlated with the truth), so it is replaced by tabs/ident.tex, built by
# src/18_identification_table.py from the partial-identification analysis.

# ---------------------------------------------------------------- T3 demand forecasting
try:
    F = json.load(open('results/l1_forecast.json'))
    A, B = F['task_A_day_ahead'], F['task_B_ex_ante']
    SHORT = {'Seasonal naive (t-364)': 'Seasonal naive',
             'Random Forest': 'Random forest',
             'DCG-blind (ablation)': 'PL-Net (ours)',
             'DCG censoring-aware (proposed)': 'ICG (latent target)'}
    def cells(v):
        if v is None: return ' & '.join(['--']*3)
        dm = '--'
        if 'DM_stat' in v:
            st = '**' if v['DM_p'] < 0.01 else ('*' if v['DM_p'] < 0.05 else '')
            dm = f"{v['DM_stat']:+.2f}$^{{{st}}}$" if st else f"{v['DM_stat']:+.2f}"
        return f"{v['MAPE']:.2f} & {v['RMSE']/1e3:.1f} & {dm}"
    add(r'\begin{table}[!t]' + '\n' +
        r'\caption{Demand forecasting accuracy against the \emph{observed} served '
        r'series. Task~A is day-ahead with autoregressive features; Task~B is an '
        r'ex-ante 433-day horizon on exogenous drivers only. DM = Diebold--Mariano '
        r'statistic against the best conventional baseline in that task '
        r'($^{*}p<0.05$, $^{**}p<0.01$; ``--'' marks the reference itself). The final '
        r'row targets \emph{latent} demand, so its error against censored observations '
        r'is not a defect; it is evaluated in Table~\ref{tab:recovery}.}' + '\n' +
        r'\label{tab:forecast}' + '\n' + r'\centering' + '\n' +
        r'\resizebox{\columnwidth}{!}{%' + '\n' +
        r'\begin{tabular}{lrrrrrr}' + '\n' + r'\toprule' + '\n' +
        r'& \multicolumn{3}{c}{Task A: day-ahead} & '
        r'\multicolumn{3}{c}{Task B: ex-ante 433 d} \\' + '\n' +
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\n' +
        r'Model & MAPE & RMSE & DM & MAPE & RMSE & DM \\' + '\n' +
        r'& (\%) & (GWh) & & (\%) & (GWh) & \\' + '\n' + r'\midrule')
    for k in A:
        nm = SHORT.get(k, k)
        nm = r'\textbf{' + nm + '}' if 'DCG-blind' in k else nm
        add(f'{nm} & {cells(A.get(k))} & {cells(B.get(k))} ' + BS)
    add(r'\bottomrule\end{tabular}}\end{table}')
except Exception as e: add(f'% forecasting tables unavailable: {e}')

# ---------------------------------------------------------------- T4 composition
try:
    Lc = json.load(open('results/l2_composition.json'))
    A, B = Lc['day_ahead'], Lc['ex_ante']
    SHORT = {'B2 seasonal naive (t-364)': 'B2 seasonal naive',
             'B3 per-fuel GBM (levels)+renorm': 'B3 per-fuel GBM',
             'B4 per-share GBM+renorm': 'B4 per-share GBM',
             'C1 ALR-GBM (gas reference)': 'C1 ALR-GBM',
             'C2 ILR-GBM (merit-order basis)': 'C2 ILR-GBM (ours)',
             'C3 Dirichlet-Net (proposed)': 'C3 Dirichlet-Net (ours)'}
    gkA = [c for c in list(A.values())[0] if c.startswith('aitch_gain')][0]
    gkB = [c for c in list(B.values())[0] if c.startswith('aitch_gain')][0]
    add(r'\begin{table}[!t]' + '\n' +
        r'\caption{Fuel-mix forecasting. Aitchison distance is the proper metric on '
        r"the simplex; ``viol.'' is the share of forecasts that leave the simplex "
        r"before renormalisation; ``gain'' is the Aitchison improvement over the best "
        r'baseline in that task. The compositional advantage is decisive day-ahead and '
        r'largely absent ex ante---a null we report rather than omit.}' + '\n' +
        r'\label{tab:mix}' + '\n' + r'\centering' + '\n' +
        r'\resizebox{\columnwidth}{!}{%' + '\n' +
        r'\begin{tabular}{lrrrrrr}' + '\n' + r'\toprule' + '\n' +
        r'& \multicolumn{3}{c}{Day-ahead} & \multicolumn{3}{c}{Ex-ante 432 d} \\' + '\n' +
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\n' +
        r'Model & Aitch. & MAE & Viol. & Aitch. & MAE & Viol. \\' + '\n' +
        r'& & (pp) & (\%) & & (pp) & (\%) \\' + '\n' + r'\midrule')
    for k in A:
        nm = SHORT.get(k, k)
        nm = r'\textbf{' + nm + '}' if k.startswith(('C2', 'C3')) else nm
        a, b = A[k], B.get(k)
        add(f"{nm} & {a['aitchison']:.3f} & {a['mae_total']:.2f} & "
            f"{a['simplex_violation_pct']:.0f} & "
            f"{b['aitchison']:.3f} & {b['mae_total']:.2f} & "
            f"{b['simplex_violation_pct']:.0f} " + BS)
    add(r'\bottomrule\end{tabular}}\end{table}')
except Exception as e: add(f'% composition tables unavailable: {e}')

# ---------------------------------------------------------------- T5 economics
try:
    E = pd.read_csv('results/l3_economics.csv', index_col=0)
    add(r"""\begin{table*}[!t]
\caption{Loss-adjusted economics. WACOG = weighted average cost of generation.
Subsidy gap = generation cost less revenue on delivered energy at the average retail
tariff. ENS cost at a value of lost load of 90~BDT/kWh. USD at 122~BDT.}
\label{tab:econ}
\centering\small
\begin{tabular}{lrrrrrr}
\toprule
Year & Gen. & Deliv. & WACOG & Tariff & Subsidy & ENS cost \\
& (TWh) & (TWh) & \multicolumn{2}{c}{(BDT/kWh)} & \multicolumn{2}{c}{(bn USD)} \\
\midrule""")
    for yr in [v for v in E.index if v not in (2015, 2026)]:
        q = E.loc[yr]
        add(f'{yr} & {q.gen_TWh:.1f} & {q.delivered_TWh:.1f} & {q.WACOG_BDT_kWh:.2f} & '
            f'{q.tariff_BDT_kWh:.2f} & {q.subsidy_musd/1e3:.2f} & '
            f'{q["ENS_cost_crore@90"]*1e7/122/1e9:.2f} ' + BS)
    add(r'\bottomrule\end{tabular}\end{table*}')

    CF = pd.read_csv('results/l3_counterfactual.csv', index_col=0)
    add(r"""\begin{table}[!t]
\caption{Counterfactual cost of retaining the 2021 generation mix. Negative
$\Delta$ means the realised mix was cheaper than the 2021 mix would have been.}
\label{tab:cf}
\centering\small
\begin{tabular}{lrrr}
\toprule
Year & Realised (bn USD) & 2021-mix (bn USD) & $\Delta$ (\%) \\
\midrule""")
    for yr in CF.index:
        q = CF.loc[yr]
        add(f'{yr} & {q.actual_crore*1e7/122/1e9:.2f} & '
            f'{q.cf2021mix_crore*1e7/122/1e9:.2f} & {q.delta_pct:+.1f} ' + BS)
    add(r'\bottomrule\end{tabular}\end{table}')
except Exception as e: add(f'% economics tables unavailable: {e}')

# ---------------------------------------------------------------- T6 projection
# The 2030 projection was cut from the manuscript (scenario analysis, not a
# validated result). results/projection_demand.csv is still produced by
# src/09_projection.py for the record; it is simply no longer tabulated.

open('paper/tables.tex', 'w', encoding='utf-8').write('\n'.join(out))
print(f'wrote paper/tables.tex ({len(out)} lines)')
