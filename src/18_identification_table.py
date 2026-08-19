"""Emit the partial-identification table (replaces the withdrawn point estimate)."""
import json, pathlib

R = json.load(open('results/identification.json'))
BS = chr(92)*2
bnd = R['bounds']; g0 = R['gap_no_control']; g1 = R['gap_tariff_control']
g2 = R['gap_gdp_control']; g3 = R['gap_both_controls']
floor0, floor2 = R['placebo_sd_pp'], R['placebo_sd_gdp']

def num(dic, y):
    v = dic.get(str(y), dic.get(y))
    return float(v) if v is not None else float('nan')

rows = []
for y in (2022, 2023, 2024, 2025):
    lo, hi = num(bnd['lo_pct'], y), num(bnd['hi_pct'], y)
    inside = (lo - floor2) <= num(g3, y) <= (hi + floor2)
    rows.append((y, lo, hi, num(g0, y), num(g1, y), num(g2, y), num(g3, y), inside))

out = [r"""\begin{table*}[!t]
\caption{Partial identification of suppressed demand, in percentage points of annual
served energy. \textbf{Bounds} are model-free: with an admission rate
$\alpha\in[\alpha_{\min},1]$, suppression lies in $[S/G,\;S/(\alpha_{\min}G)]$ with
$\alpha_{\min}=0.30$. \textbf{Structural} is an independent OLS counterfactual fitted
on the uncensored 2016--2021 era and extrapolated; it never uses $S$. Controls enter
as the log real (CPI-deflated) average tariff and annual real GDP growth. The placebo
column is the standard deviation of the same statistic over the uncensored years,
which is the relevant noise floor. Agreement of two independent routes---not either
one alone---is the evidence.}
\label{tab:ident}
\centering\small
\begin{tabular}{lrrrrrrc}
\toprule
& \multicolumn{2}{c}{Model-free bounds} & \multicolumn{4}{c}{Structural counterfactual (pp)} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-7}
Year & lower & upper & none & +tariff & +GDP & +both & consistent? \\
\midrule"""]
for y, lo, hi, a, b, c, dd, ins in rows:
    out.append(f'{y} & {lo:.2f} & {hi:.2f} & {a:+.2f} & {b:+.2f} & {c:+.2f} & '
               + (r'\textbf{' + f'{dd:+.2f}' + '}') + ' & '
               + ('yes' if ins else 'no') + ' ' + BS)
out.append(r'\midrule')
out.append(f'Placebo sd (2016--2021) & -- & -- & {floor0:.2f} & {R["placebo_sd_both"]:.2f} & '
           f'{floor2:.2f} & {R["placebo_sd_both"]:.2f} & ' + BS)
out.append(r'\bottomrule\end{tabular}\end{table*}')

pathlib.Path('paper/tabs').mkdir(parents=True, exist_ok=True)
pathlib.Path('paper/tabs/ident.tex').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('wrote paper/tabs/ident.tex')
print(f'  real tariff change 2021->2025: {R["real_tariff_change_2021_2025_pct"]:+.1f}%')
print(f'  placebo sd (no control) {floor0:.2f} pp; (GDP control) {floor2:.2f} pp')
for y, lo, hi, a, b, c, dd, ins in rows:
    print(f'  {y}: bounds [{lo:.2f}, {hi:.2f}]  both-controls {dd:+.2f}  consistent={ins}')
