import urllib.request, json

base = 'http://localhost:8000'

# 1. Check homepage HTML for required elements
try:
    r = urllib.request.urlopen(base + '/', timeout=5)
    html = r.read().decode('utf-8')
    checks = ['wars-screen-hub', 'wars-screen-create', 'wars-screen-league', 'wars-screen-match',
              'wars.css', 'wars.js', 'Orbitron', 'code-badge', 'w-btn-gold', 'w-btn-purple',
              'stock-tile-grid', 'chat-feed', 'lineup-slots', 'match-content']
    print('=== HTML Checks ===')
    for c in checks:
        present = c in html
        status = 'OK' if present else 'MISSING'
        print(f'  [{status}] {c}')
except Exception as e:
    print('Server error:', e)

# 2. Test backend API routes
endpoints = [
    ('POST', '/api/wars/league', {'name': 'Smoke Test League'}),
]
print()
print('=== API Routes ===')
for method, path, body in endpoints:
    try:
        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(base + path, data=data,
            headers={'Content-Type': 'application/json'}, method=method)
        r = urllib.request.urlopen(req, timeout=10)
        resp = json.loads(r.read())
        print(f'  [OK] {method} {path} -> code={resp.get("code","?")}, id={resp.get("id","?")}')
    except Exception as e:
        print(f'  [ERR] {method} {path}: {e}')

print('DONE')
