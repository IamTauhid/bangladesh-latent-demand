r"""One-off: replace the invalid siunitx currency units with declared ones."""
import pathlib

BAD_BILLION = "{" + chr(92) + "billion" + chr(92) + "$}"
BAD_MEGA    = "{" + chr(92) + "mega" + chr(92) + "$}"
GOOD_BILLION = "{" + chr(92) + "billionUSD}"
GOOD_MEGA    = "{" + chr(92) + "megaUSD}"
BS = chr(92)

base = pathlib.Path('paper')

# 1. repair the corrupted declaration line (a literal backspace byte got in)
m = base / 'main.tex'
lines = m.read_text(encoding='utf-8').split('\n')
for i, l in enumerate(lines):
    if 'illionUSD' in l and 'DeclareSIUnit' in l:
        lines[i] = BS + "DeclareSIUnit{" + BS + "billionUSD}{bn~USD}"
m.write_text('\n'.join(lines), encoding='utf-8')

# 2. swap the invalid units everywhere
for f in ['main.tex', 'results_section.tex', 'discussion_section.tex']:
    p = base / f
    s = p.read_text(encoding='utf-8')
    n0 = s.count(BAD_BILLION) + s.count(BAD_MEGA)
    s = s.replace(BAD_BILLION, GOOD_BILLION).replace(BAD_MEGA, GOOD_MEGA)
    p.write_text(s, encoding='utf-8')
    print(f'{f}: replaced {n0}; remaining bad = '
          f'{s.count(BAD_BILLION) + s.count(BAD_MEGA)}')

print('\ndeclarations now:')
for l in (base / 'main.tex').read_text(encoding='utf-8').split('\n'):
    if 'DeclareSIUnit' in l:
        print('  ', repr(l))
