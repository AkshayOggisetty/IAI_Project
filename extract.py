import json

with open('IAIProject.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('out.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb.get('cells', [])):
        f.write(f"Cell {i} ({cell.get('cell_type')}):\n")
        source = cell.get('source', [])
        f.write(''.join(source) + '\n')
        f.write('='*40 + '\n')
