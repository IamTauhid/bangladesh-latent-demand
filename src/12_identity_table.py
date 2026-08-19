"""Emit the accounting-identity table (Sec. III-B); appended to paper/tables.tex."""
import pandas as pd, os

BS = chr(92)*2          # LaTeX row terminator, built without escape sequences

h = pd.read_csv('data/hourly_clean.csv', parse_dates=['datetime'],
                index_col='datetime').dropna(subset=['generation_mw', 'demand_mw'])
h['gap'] = h.demand_mw - h.generation_mw

head = r"""\begin{table}[!t]
\caption{The published demand series is an accounting identity. Column~3 is the share
of hours in which reported demand equals generation plus reported load shedding
exactly; column~4 the share in which it equals generation alone. Before 2022 the
series carries essentially no information beyond served generation.}
\label{tab:identity}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Year & Hours & $\tilde D=G+S$ (\%) & $\tilde D=G$ (\%) & Mean $S$ (MW) """ + BS + r"""
\midrule"""

out = [head]
for yr, s in h.groupby(h.index.year):
    ident = (s.gap.round() == s.load_shedding.round()).mean()*100
    equal = (s.gap.abs() < 1e-6).mean()*100
    out.append(f'{yr} & {len(s):,} & {ident:.1f} & {equal:.1f} & '
               f'{s.load_shedding.mean():.1f} ' + BS)

pre  = h[h.index.year < 2022]
post = h[h.index.year >= 2022]
out.append(r'\midrule')
for lab, s in [('2015--2021', pre), ('2022--2026', post)]:
    ident = (s.gap.round() == s.load_shedding.round()).mean()*100
    equal = (s.gap.abs() < 1e-6).mean()*100
    cell = (r'\textbf{' + f'{equal:.1f}' + '}') if s is pre else f'{equal:.1f}'
    out.append(f'{lab} & {len(s):,} & {ident:.1f} & {cell} & '
               f'{s.load_shedding.mean():.1f} ' + BS)
out.append(r'\bottomrule\end{tabular}}\end{table}')

with open('paper/tables.tex', 'a', encoding='utf-8', newline='\n') as f:
    f.write('\n' + '\n'.join(out) + '\n')

print('appended tab:identity')
print(f'  pre-2022, demand == generation exactly: '
      f'{(pre.gap.abs() < 1e-6).mean()*100:.2f}% of hours')
