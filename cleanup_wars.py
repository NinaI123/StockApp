"""Remove all remaining wars content from index.html and main.py."""
import re

# ── 1. index.html ─────────────────────────────────────────────────────────────
with open('templates/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Remove from the stray wars-bg div to end of /wars-section comment
# Use regex to find and remove the entire block
html = re.sub(
    r'\n?\s*<div class="wars-bg">.*?</div><!--\s*/wars-section\s*-->',
    '',
    html,
    flags=re.DOTALL
)

# Also remove Orbitron font link (no longer needed)
html = re.sub(
    r'\s*<link rel="preconnect" href="https://fonts.googleapis.com">\s*\n',
    '\n',
    html
)
html = re.sub(
    r'\s*<link\s+href="https://fonts.googleapis.com/css2\?family=Orbitron[^"]*"[^>]*>\s*\n',
    '\n',
    html
)

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"index.html: {html.count(chr(10))} lines, wars-bg present: {'wars-bg' in html}")


# ── 2. main.py ────────────────────────────────────────────────────────────────
with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the wars section start and the predictions section start
wars_start = None
wars_end   = None
pred_start = None
for i, ln in enumerate(lines):
    if '# ═' in ln and 'PORTFOLIO WARS' in ln:
        wars_start = i
    if '# ═' in ln and 'PREDICTIONS' in ln and wars_start:
        pred_start = i
        wars_end   = i
        break

print(f"main.py: wars block lines {wars_start+1}–{wars_end} (0-idx), keeping from {pred_start+1}")

# Also remove wars Pydantic models (LeagueCreate, TeamCreate, MatchupCreate, MessageCreate, DraftPick)
# and wars helpers (_current_week, _gen_code, _compute_streak)
# They are between lines 99-158 (0-idx 98-157)
# Let's find them precisely
model_start = model_end = helper_start = helper_end = None
for i, ln in enumerate(lines):
    if 'class LeagueCreate' in ln and model_start is None:
        model_start = i
    if 'class PredictionCreate' in ln and model_start is not None and model_end is None:
        model_end = i  # keep PredictionCreate
    if 'def _current_week' in ln:
        helper_start = i
    if '# ── Routes' in ln and helper_start is not None:
        helper_end = i  # stop before routes

print(f"Models: {model_start+1}–{model_end} | Helpers: {helper_start+1}–{helper_end}")

# Also find pw_ references in get_user_profile (lines ~741-786)
# Remove the pw_ query blocks from that function — we'll handle by keeping the function
# but removing the wars stats queries
# For now: remove the two cur.execute blocks that reference pw_results/pw_teams
# We'll do that separately after the main cuts

# Build new lines
new_lines = (
    lines[:model_start] +          # everything before LeagueCreate
    lines[model_end:helper_start] + # PredictionCreate + gap
    lines[helper_end:wars_start] +  # routes up to wars section
    lines[wars_end:]                 # predictions onwards
)

with open('main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print(f"main.py written: {len(new_lines)} lines")
