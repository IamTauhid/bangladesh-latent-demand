"""
Split the generated tables.tex into one file per table so each float can be placed
next to the text that first cites it. LaTeX can only place a float at or after the
point it appears in the source; grouping every table at the end of the document
forces them all to the back of the paper.
"""
import re, pathlib

src = pathlib.Path('paper/tables.tex').read_text(encoding='utf-8')
out_dir = pathlib.Path('paper/tabs'); out_dir.mkdir(parents=True, exist_ok=True)

# split on table / table* openers, keeping the delimiter
chunks = re.split(r'(?=\\begin\{table\*?\})', src)
written = []
for c in chunks:
    if not c.strip().startswith(r'\begin{table'):
        continue
    m = re.search(r'\\label\{(tab:[A-Za-z0-9_]+)\}', c)
    if not m:
        continue
    name = m.group(1).split(':', 1)[1]
    # trim anything after the closing environment
    end = c.rfind(r'\end{table*}')
    end = end + len(r'\end{table*}') if end != -1 else \
          c.rfind(r'\end{table}') + len(r'\end{table}')
    (out_dir / f'{name}.tex').write_text(c[:end].rstrip() + '\n', encoding='utf-8')
    written.append(name)

print(f'wrote {len(written)} table files to paper/tabs/:')
for w in written:
    print('   ', w)
