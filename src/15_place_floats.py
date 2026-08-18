"""
Move every float to the point in the source where it is first discussed.

LaTeX places a float at or after its position in the source, so a block of floats
declared just before the bibliography can only be typeset at the end of the paper.
This script (a) lifts the grouped figure environments out of main.tex into one file
each, (b) deletes the grouped \input{tables}, and (c) re-inserts each float
immediately before the (sub)section that first cites it.
"""
import re, pathlib

paper = pathlib.Path('paper')
figdir = paper / 'figtex'; figdir.mkdir(exist_ok=True)
main = paper / 'main.tex'
s = main.read_text(encoding='utf-8')

# ---- 1. extract grouped figure environments into paper/figtex/<name>.tex ----
fig_re = re.compile(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.S)
found = fig_re.findall(s)
names = []
for blk in found:
    m = re.search(r'figs/([A-Za-z0-9_]+)\.pdf', blk)
    if not m:
        continue
    n = m.group(1)
    (figdir / f'{n}.tex').write_text(blk.strip() + '\n', encoding='utf-8')
    names.append(n)
    s = s.replace(blk, '')
print(f'extracted {len(names)} figures:', ', '.join(names))

# ---- 2. drop the grouped table input ----
s = s.replace('\\input{tables}\n', '').replace('\\input{tables}', '')

# ---- 3. relax IEEEtran's float limits so several floats can share a page ----
if r'\setcounter{topnumber}' not in s:
    s = s.replace(r'\DeclareSIUnit{\pp}{pp}',
                  r'\DeclareSIUnit{\pp}{pp}' + '\n\n'
                  '% --- float placement: allow more floats per page than the default ---\n'
                  r'\setcounter{topnumber}{3}' + '\n'
                  r'\setcounter{bottomnumber}{2}' + '\n'
                  r'\setcounter{totalnumber}{5}' + '\n'
                  r'\setcounter{dbltopnumber}{3}' + '\n'
                  r'\renewcommand{\topfraction}{0.92}' + '\n'
                  r'\renewcommand{\dbltopfraction}{0.92}' + '\n'
                  r'\renewcommand{\bottomfraction}{0.5}' + '\n'
                  r'\renewcommand{\textfraction}{0.06}' + '\n'
                  r'\renewcommand{\floatpagefraction}{0.72}' + '\n'
                  r'\renewcommand{\dblfloatpagefraction}{0.72}' + '\n')

# ---- 4. insert floats immediately before the text that first cites them ----
def insert_before(text, anchor, payload):
    i = text.find(anchor)
    if i == -1:
        print('  ANCHOR NOT FOUND:', anchor[:60]); return text
    return text[:i] + payload + '\n\n' + text[i:]

MAIN_PLACEMENT = [
    (r'\subsection{Dataset and Physics-Informed Cleaning}',
     '\\input{figtex/fig2_series_mix}\n\\input{tabs/annual}'),
    (r'\subsection{The Published Demand Series Is Determined by Supply}',
     '\\input{tabs/identity}'),
    (r'\subsection{The Rationing Regime Is Flat, Not Peak-Clipped}',
     '\\input{figtex/fig3_censoring_diagnostics}'),
]
for anchor, payload in MAIN_PLACEMENT:
    s = insert_before(s, anchor, payload)
main.write_text(s, encoding='utf-8')

# ---- 5. same for the results section ----
res = paper / 'results_section.tex'
r = res.read_text(encoding='utf-8')
RES_PLACEMENT = [
    (r'\subsection{Right-Censored versus Interval-Censored Recovery}',
     '\\input{tabs/recovery}'),
    (r'\subsection{Recovered Latent Demand}',
     '\\input{tabs/latent}\n\\input{figtex/fig6_latent_projection}'),
    (r'\subsection{Demand Forecasting}',
     '\\input{tabs/forecast}'),
    (r'\subsection{Fuel-Mix Forecasting}',
     '\\input{tabs/mix}\n\\input{figtex/fig8_mix_forecast}'),
    (r'\subsection{Seasonality and Its Physical Drivers}',
     '\\input{figtex/fig4_seasonality}'),
    (r'\subsection{Loss-Adjusted Economics}',
     '\\input{tabs/econ}\n\\input{figtex/fig7_economics}'),
    (r'\subsection{Projection to 2030}',
     '\\input{tabs/cf}\n\\input{tabs/proj}'),
]
for anchor, payload in RES_PLACEMENT:
    r = insert_before(r, anchor, payload)
res.write_text(r, encoding='utf-8')
print('floats repositioned next to their first citation')
