import urllib.request, json, sys, time

base = 'http://localhost:8000'

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{base}{path}', data=data,
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(f'{base}{path}') as r:
        return json.loads(r.read())

try:
    unique_name = f"Test League {int(time.time())}"
    print(f"=== 1. Create League ({unique_name}) ===")
    league = post('/api/wars/league', {'name': unique_name})
    print(json.dumps(league, indent=2))
    assert 'id' in league

    print("\n=== 2. Join as Player 1 ===")
    t1 = post('/api/wars/team', {
        'league_id': league['id'],
        'player_name': 'BullishNina',
        'stocks': ['AAPL', 'TSLA', 'NVDA']
    })
    print(json.dumps(t1, indent=2))
    assert 'id' in t1

    print("\n=== 3. Join as Player 2 ===")
    t2 = post('/api/wars/team', {
        'league_id': league['id'],
        'player_name': 'BearBoss',
        'stocks': ['MSFT', 'AMZN', 'META']
    })
    print(json.dumps(t2, indent=2))
    assert 'id' in t2

    print("\n=== 4. Create Matchup ===")
    mu = post('/api/wars/matchup', {
        'league_id': league['id'],
        'team1_id': t1['id'],
        'team2_id': t2['id']
    })
    print(json.dumps(mu, indent=2))
    assert 'id' in mu

    print("\n=== 5. Score Matchup (fetches real yfinance data) ===")
    score = post(f'/api/wars/score/{mu["id"]}', {})
    print(json.dumps(score, indent=2))
    assert 'winner' in score

    print("\n=== 6. Leaderboard ===")
    lb = get(f'/api/wars/league/{league["id"]}')
    print(json.dumps(lb, indent=2))
    assert len(lb['teams']) == 2

    print("\n=== 7. DB Tables ===")
    import sqlite3
    conn = sqlite3.connect('data/portfolio.db')
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("Tables:", tables)
    for t in ['pw_leagues', 'pw_teams', 'pw_matchups', 'pw_results']:
        assert t in tables, f"Missing table: {t}"
    conn.close()

    print("\n\n ALL PORTFOLIO WARS TESTS PASSED ")

except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
