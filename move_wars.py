"""
Move all Portfolio Wars files into portfolio_wars/ folder.
Run from the stockapp directory.
"""
import os, shutil

os.makedirs('portfolio_wars/static', exist_ok=True)

moves = [
    ('static/wars.css',  'portfolio_wars/static/wars.css'),
    ('static/wars.js',   'portfolio_wars/static/wars.js'),
    ('test_wars.py',     'portfolio_wars/test_wars.py'),
    ('smoke_test.py',    'portfolio_wars/smoke_test.py'),
]

for src, dst in moves:
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f'  moved: {src} → {dst}')
    else:
        print(f'  skip (not found): {src}')

print('Done moving files.')
