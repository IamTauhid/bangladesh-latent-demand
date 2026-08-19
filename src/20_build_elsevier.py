"""
Build the Energy Policy (Elsevier) manuscript from the IEEE source.

Reuses every section, table and figure file unchanged; only the document class and
front matter differ. Running this keeps the two versions in sync -- edit the section
files, re-run, and both rebuild.
"""
import pathlib, re

paper = pathlib.Path('paper')
if not (paper / 'main.tex').exists():
    raise SystemExit(
        'paper/main.tex not found. The manuscript sources are not distributed '
        'in this repository while the paper is under review (see README). This '
        'builder converts the IEEE source to Elsevier format and is included '
        'for completeness; every figure and table it references is reproducible '
        'by running stages 01-19.')
src = (paper / 'main.tex').read_text(encoding='utf-8')

# ---- body: everything from the Nomenclature to just before the bibliography ----
a = src.index(r'\section*{Nomenclature}')
b = src.index(r'\bibliographystyle{IEEEtran}')
body = src[a:b]
body = body.replace(r'\IEEEPARstart{P}{lanning}', 'Planning')
body = body.replace(r'\addcontentsline{toc}{section}{Nomenclature}', '')
# elsarticle is single-column: starred floats are unnecessary and upset placement
body = body.replace(r'\begin{table*}', r'\begin{table}').replace(r'\end{table*}', r'\end{table}')
body = body.replace(r'\begin{figure*}', r'\begin{figure}').replace(r'\end{figure*}', r'\end{figure}')
body = body.replace(r'\includegraphics[width=\columnwidth]',
                    r'\includegraphics[width=0.72\textwidth]')
body = body.replace(r'\includegraphics[width=\textwidth]',
                    r'\includegraphics[width=0.95\textwidth]')
body = body.replace(r'\resizebox{\columnwidth}', r'\resizebox{0.98\textwidth}')

# ---- abstract, lifted verbatim from the IEEE source ----
abs_a = src.index(r'\begin{abstract}') + len(r'\begin{abstract}')
abs_b = src.index(r'\end{abstract}')
abstract = src[abs_a:abs_b].strip()

REPO = 'https://github.com/IamTauhid/bangladesh-latent-demand'

HIGHLIGHTS = [
 "Published Bangladeshi grid demand equals served supply in 90\\% of pre-crisis hours.",
 "Rationing was flat across the day: a fuel-cost, not a capacity-adequacy, constraint.",
 "Suppressed demand is partially identified; we report bounds, not a point estimate.",
 "2023 suppression was 3.0--9.9\\% of served energy against 2.97\\% officially admitted.",
 "Real electricity tariffs fell 13.7\\%, so price response cannot explain the gap.",
]

doc = r"""%% ============================================================================
%%  Energy Policy submission -- Islam & Mohiuddin
%%  Built by src/20_build_elsevier.py from the shared section files.
%%  Do not edit directly; edit paper/*.tex and re-run the builder.
%% ============================================================================
\documentclass[preprint,review,12pt]{elsarticle}

\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{graphicx,booktabs,multirow,array}
\usepackage[table]{xcolor}
\usepackage{url}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\usepackage{siunitx}
\usepackage{lineno}
\modulolinenumbers[1]
\sisetup{detect-all,group-separator={,}}
\DeclareSIUnit{\billionUSD}{bn~USD}
\DeclareSIUnit{\megaUSD}{M~USD}
\DeclareSIUnit{\BDTkWh}{BDT/kWh}
\DeclareSIUnit{\pp}{pp}
\newcommand{\BDT}{\text{BDT}}
\DeclareMathOperator{\ilr}{ilr}
\DeclareMathOperator{\alr}{alr}
\DeclareMathOperator{\clr}{clr}
\DeclareMathOperator*{\argmin}{arg\,min}

\journal{Energy Policy}

\begin{document}
\linenumbers

\begin{frontmatter}

\title{Interval-Censored Learning for Latent Demand and Dispatch
Composition in Rationed Power Systems: Evidence from Bangladesh}

\author[unn]{Md Tauhidul Islam\corref{cor}}
\ead{tauhidsuman10@gmail.com}
\author[mephi]{Saifullah Mohiuddin}
\ead{saif010256@gmail.com}
\cortext[cor]{Corresponding author.}

\address[unn]{Lobachevsky State University of Nizhny Novgorod,
Nizhny Novgorod 603022, Russia}
\address[mephi]{National Research Nuclear University MEPhI
(Moscow Engineering Physics Institute), Moscow 115409, Russia}

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\begin{highlights}
""" + '\n'.join(r'\item ' + h for h in HIGHLIGHTS) + r"""
\end{highlights}

\begin{keyword}
Electricity demand \sep censored regression \sep load shedding \sep
unserved energy \sep compositional data analysis \sep Bangladesh
\end{keyword}

\end{frontmatter}

%% ---------------------------------------------------------------------------
""" + body + r"""

\section*{Data and code availability}
All source code, the processed design matrix, and every result file required to
reproduce the tables and figures are openly available at
\url{""" + REPO + r"""}. The primary data are the Power Grid Company of Bangladesh
hourly records published as Mendeley Data, doi:10.17632/vpk8spw2mm.1 under CC~BY~4.0;
weather variables are derived from the NASA POWER daily API.

\section*{Declaration of competing interest}
The authors declare no competing financial interests or personal relationships that
could have appeared to influence the work reported in this paper.

\section*{Declaration of generative AI and AI-assisted technologies in the writing
process}
During the preparation of this work the authors used Claude (Anthropic) in order to
assist with software development for the analysis pipeline, and to draft and edit
portions of the manuscript text. After using this tool the authors reviewed and
edited the content as needed and take full responsibility for the content of the
publication. The research question, the identification strategy, the interpretation
of results, and all scientific claims are the authors' own.

\bibliographystyle{elsarticle-num}
\input{refs}

\end{document}
"""

(paper / 'energypolicy.tex').write_text(doc, encoding='utf-8')
print('wrote paper/energypolicy.tex')
print(f'  body carried over: {len(body):,} chars')
print(f'  abstract: {len(abstract.split())} words')
print(f'  highlights: {len(HIGHLIGHTS)}')
for h in HIGHLIGHTS:
    n = len(h.replace(chr(92)+chr(92), ''))
    flag = '' if n <= 85 else '  <-- OVER 85 CHARS'
    print(f'    {n:3d}  {h[:70]}{flag}')
