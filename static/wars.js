/* ═══════════════════════════════════════════════════════════
   ⚔️  PORTFOLIO WARS  –  wars.js
   Multi-screen SPA logic: Hub → Create/Join → League → Match
═══════════════════════════════════════════════════════════ */

'use strict';

// ── State ──────────────────────────────────────────────────
const PW = {
    leagueCode: null,
    leagueId: null,
    leagueName: null,
    teamId: null,
    teamCode: null,
    playerName: null,
    matchupId: null,
    lineup: [],          // selected tickers
    refreshTimer: null,
    msgRefreshTimer: null,
    stockData: {},          // cache of tile data
};

const MAX_STOCKS = 5;
const MIN_STOCKS = 3;

// ── Screen Navigation ──────────────────────────────────────
function showWarsScreen(name) {
    document.querySelectorAll('.wars-screen').forEach(s => s.classList.remove('active'));
    const el = document.getElementById('wars-screen-' + name);
    if (el) el.classList.add('active');
    window.scrollTo(0, 0);
}

// ── Utility ────────────────────────────────────────────────
async function pwFetch(url, opts = {}) {
    const res = await fetch(url, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    return data;
}

function pwStatus(id, msg, ok = true) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.color = ok ? '#a78bfa' : '#f87171';
    el.innerHTML = msg;
}

function fmtRet(v) {
    if (v === null || v === undefined || v === 'N/A') return `<span class="ret-neu">N/A</span>`;
    const n = parseFloat(v);
    const cls = n > 0 ? 'ret-pos' : n < 0 ? 'ret-neg' : 'ret-neu';
    const sign = n > 0 ? '+' : '';
    return `<span class="${cls}">${sign}${n.toFixed(2)}%</span>`;
}

function timeAgo(ts) {
    if (!ts) return '';
    const diff = (Date.now() - new Date(ts + 'Z').getTime()) / 1000;
    if (diff < 60) return `${Math.round(diff)}s ago`;
    if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
    return `${Math.round(diff / 3600)}h ago`;
}

function copyToClipboard(text, btnId) {
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById(btnId);
        if (!btn) return;
        const orig = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check"></i>';
        btn.style.color = '#4ade80';
        setTimeout(() => { btn.innerHTML = orig; btn.style.color = ''; }, 1500);
    });
}

// ── Nav Section Toggle (main ↔ wars) ──────────────────────
function showSection(section) {
    const mainEl = document.querySelector('.container');
    const warsEl = document.getElementById('wars-section');
    if (section === 'wars') {
        mainEl.style.display = 'none';
        warsEl.style.display = 'block';
        document.getElementById('tab-main').classList.remove('active-tab');
        document.getElementById('tab-wars').classList.add('active-tab');
        showWarsScreen('hub');
    } else {
        mainEl.style.display = '';
        warsEl.style.display = 'none';
        document.getElementById('tab-main').classList.add('active-tab');
        document.getElementById('tab-wars').classList.remove('active-tab');
    }
}

// ══════════════════════════════════════════════════════════
//  HUB SCREEN
// ══════════════════════════════════════════════════════════
function goToCreate(mode) {
    // mode: 'create' (create new league) or 'join' (join existing)
    PW.lineup = [];
    renderLineup();
    if (mode === 'join') {
        document.getElementById('create-mode-tabs').querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
        document.getElementById('tab-mode-join').classList.add('active');
        showCreateMode('join');
    } else {
        document.getElementById('create-mode-tabs').querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
        document.getElementById('tab-mode-create').classList.add('active');
        showCreateMode('create');
    }
    showWarsScreen('create');
    loadStockTiles();
}

function showCreateMode(mode) {
    document.getElementById('create-panel').style.display = mode === 'create' ? 'block' : 'none';
    document.getElementById('join-panel').style.display = mode === 'join' ? 'block' : 'none';
    document.getElementById('draft-panel').style.display = mode === 'join' ? 'block' : 'none';
}

// ══════════════════════════════════════════════════════════
//  CREATE / JOIN SCREEN
// ══════════════════════════════════════════════════════════

// ── Create League ─────────────────────────────────────────
async function pwCreateLeague() {
    const name = document.getElementById('pw-league-name').value.trim();
    if (!name) return pwStatus('pw-create-status', '⚠️ Enter a league name.', false);
    pwStatus('pw-create-status', '<i class="fas fa-spinner fa-spin me-1"></i>Creating league…');
    try {
        const data = await pwFetch('/api/wars/league', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        PW.leagueCode = data.code;
        PW.leagueId = data.id;
        PW.leagueName = data.name;
        document.getElementById('pw-created-code').textContent = data.code;
        document.getElementById('pw-created-name').textContent = data.name;
        document.getElementById('pw-create-success').style.display = 'block';
        document.getElementById('pw-create-form').style.display = 'none';
        // pre-fill join panel
        document.getElementById('pw-join-league-code').value = data.code;
        pwStatus('pw-create-status', '');
    } catch (e) {
        pwStatus('pw-create-status', `❌ ${e.message}`, false);
    }
}

function copyLeagueCode() {
    copyToClipboard(PW.leagueCode || document.getElementById('pw-created-code').textContent, 'copy-code-btn');
}

function proceedToDraft() {
    showCreateMode('join');
    document.getElementById('tab-mode-join').classList.add('active');
    document.getElementById('tab-mode-create').classList.remove('active');
}

// ── Join & Draft Team ─────────────────────────────────────
async function pwJoinLeague() {
    const league_code = document.getElementById('pw-join-league-code').value.trim();
    const player_name = document.getElementById('pw-player-name').value.trim();
    const team_name = document.getElementById('pw-team-name').value.trim();
    if (!league_code) return pwStatus('pw-join-status', '⚠️ League code required.', false);
    if (!player_name) return pwStatus('pw-join-status', '⚠️ Your name required.', false);
    if (!team_name) return pwStatus('pw-join-status', '⚠️ Team name required.', false);
    if (PW.lineup.length < MIN_STOCKS)
        return pwStatus('pw-join-status', `⚠️ Pick at least ${MIN_STOCKS} stocks.`, false);

    pwStatus('pw-join-status', '<i class="fas fa-spinner fa-spin me-1"></i>Drafting team…');
    try {
        const data = await pwFetch('/api/wars/team', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ league_code, player_name, team_name, stocks: PW.lineup })
        });
        PW.teamId = data.id;
        PW.teamCode = data.code;
        PW.playerName = player_name;
        PW.leagueCode = league_code;
        PW.leagueId = data.league_id;
        pwStatus('pw-join-status', `✅ Team <strong>${team_name}</strong> drafted! Code: <span class="mono">${data.code}</span>`);
        // load the league after short delay
        await new Promise(r => setTimeout(r, 900));
        pwLoadLeagueByCode(league_code);
    } catch (e) {
        pwStatus('pw-join-status', `❌ ${e.message}`, false);
    }
}

// ── Stock Tile Picker ──────────────────────────────────────
const DEFAULT_TICKERS = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'NFLX'];

async function loadStockTiles() {
    const grid = document.getElementById('stock-tile-grid');
    grid.innerHTML = DEFAULT_TICKERS.map(sym => `
    <div class="col-6 col-sm-3">
      <div class="stock-tile" id="tile-${sym}" onclick="toggleStock('${sym}')">
        <span class="check-mark"><i class="fas fa-check-circle"></i></span>
        <div class="sym">${sym}</div>
        <div class="price" id="tile-price-${sym}">—</div>
        <div id="tile-chg-${sym}">—</div>
      </div>
    </div>`).join('');

    // Fetch prices in background
    for (const sym of DEFAULT_TICKERS) {
        fetchTilePrice(sym);
    }
}

async function fetchTilePrice(sym) {
    try {
        const data = await pwFetch(`/api/stock/summary?symbol=${sym}`);
        PW.stockData[sym] = data;
        const price = data.current_price ?? data.price;
        const chg = data.change_percent ?? data.chg;
        const priceEl = document.getElementById(`tile-price-${sym}`);
        const chgEl = document.getElementById(`tile-chg-${sym}`);
        if (priceEl && price) priceEl.textContent = `$${parseFloat(price).toFixed(2)}`;
        if (chgEl && chg !== undefined) {
            const n = parseFloat(chg);
            chgEl.className = n >= 0 ? 'chg-pos' : 'chg-neg';
            chgEl.textContent = (n >= 0 ? '+' : '') + n.toFixed(2) + '%';
        }
    } catch (e) { /* prices non-critical */ }
}

function toggleStock(sym) {
    const idx = PW.lineup.indexOf(sym);
    if (idx >= 0) {
        PW.lineup.splice(idx, 1);
        document.getElementById('tile-' + sym)?.classList.remove('selected');
    } else {
        if (PW.lineup.length >= MAX_STOCKS) {
            pwStatus('pw-join-status', `⚠️ Max ${MAX_STOCKS} stocks allowed.`, false);
            return;
        }
        PW.lineup.push(sym);
        document.getElementById('tile-' + sym)?.classList.add('selected');
    }
    renderLineup();
}

function addCustomStock() {
    const input = document.getElementById('custom-stock-input');
    const sym = input.value.trim().toUpperCase();
    if (!sym) return;
    if (PW.lineup.includes(sym)) { input.value = ''; return; }
    if (PW.lineup.length >= MAX_STOCKS) {
        pwStatus('pw-join-status', `⚠️ Max ${MAX_STOCKS} stocks.`, false);
        return;
    }
    PW.lineup.push(sym);
    input.value = '';
    // Add tile if not already present
    if (!document.getElementById('tile-' + sym)) {
        const grid = document.getElementById('stock-tile-grid');
        const div = document.createElement('div');
        div.className = 'col-6 col-sm-3';
        div.innerHTML = `<div class="stock-tile selected" id="tile-${sym}" onclick="toggleStock('${sym}')">
      <span class="check-mark"><i class="fas fa-check-circle"></i></span>
      <div class="sym">${sym}</div>
      <div class="price" id="tile-price-${sym}">—</div>
      <div id="tile-chg-${sym}">—</div>
    </div>`;
        grid.appendChild(div);
        fetchTilePrice(sym);
    } else {
        document.getElementById('tile-' + sym)?.classList.add('selected');
    }
    renderLineup();
}

function renderLineup() {
    const container = document.getElementById('lineup-slots');
    const html = Array.from({ length: MAX_STOCKS }, (_, i) => {
        const sym = PW.lineup[i];
        if (sym) return `<div class="lineup-slot filled">
      <div class="d-flex align-items-center gap-2">
        <span class="lineup-sym">${sym}</span>
      </div>
      <button class="w-btn w-btn-sm w-btn-ghost" onclick="toggleStock('${sym}')">✕</button>
    </div>`;
        return `<div class="lineup-slot"><span class="lineup-empty">Empty slot ${i + 1}</span></div>`;
    }).join('');
    container.innerHTML = html;

    const count = PW.lineup.length;
    document.getElementById('lineup-count').textContent = `${count}/${MAX_STOCKS}`;
    document.getElementById('draft-btn').disabled = count < MIN_STOCKS;
}

// ══════════════════════════════════════════════════════════
//  LEAGUE SCREEN
// ══════════════════════════════════════════════════════════

async function pwLoadLeagueByCode(code) {
    if (!code) {
        code = document.getElementById('pw-load-league-code')?.value.trim();
        if (!code) return;
    }
    pwStatus('pw-load-status', '<i class="fas fa-spinner fa-spin me-1"></i>Loading league…');
    try {
        const data = await pwFetch(`/api/wars/league/${code}`);
        PW.leagueId = data.id;
        PW.leagueCode = data.code || code;
        PW.leagueName = data.name;
        renderLeagueScreen(data);
        showWarsScreen('league');
        pwStatus('pw-load-status', '');
        // auto-refresh standings every 60s
        clearInterval(PW.refreshTimer);
        PW.refreshTimer = setInterval(() => pwRefreshLeague(), 60000);
        loadLeagueMessages(data.id);
    } catch (e) {
        pwStatus('pw-load-status', `❌ ${e.message}`, false);
    }
}

async function pwRefreshLeague() {
    if (!PW.leagueCode) return;
    try {
        const data = await pwFetch(`/api/wars/league/${PW.leagueCode}`);
        renderLeagueScreen(data);
    } catch (e) { /* silent */ }
}

function renderLeagueScreen(data) {
    // Header
    document.getElementById('lg-name').textContent = data.name;
    document.getElementById('lg-code').textContent = data.code || '—';
    document.getElementById('lg-week').textContent = data.week_start ? `Week of ${data.week_start}` : '';
    document.getElementById('lg-team-count').textContent = `${data.teams.length} team${data.teams.length !== 1 ? 's' : ''}`;

    // Standings table
    const tbody = document.getElementById('lg-standings-body');
    if (!data.teams.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="color:#4b5563;text-align:center;padding:1.5rem;">No teams yet — be the first to join!</td></tr>';
    } else {
        tbody.innerHTML = data.teams.map((t, i) => {
            const rankClass = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : '';
            const record = `<span style="color:#9ca3af">${t.wins}W – ${t.losses}L</span>`;
            const stocks = t.stocks.map(s => `<span class="stag">${s}</span>`).join('');
            const streak = t.streak && t.streak !== '-' ? t.streak : '–';
            return `<tr>
        <td><span class="rank-badge ${rankClass}">${i + 1}</span></td>
        <td>
          <div style="font-weight:700;color:#e2e2f0">${t.team_name}</div>
          <div style="font-size:0.75rem;color:#6b7280">${t.player_name}</div>
        </td>
        <td>${stocks}</td>
        <td>${record}</td>
        <td style="color:#FFD700;font-weight:600;">${streak}</td>
      </tr>`;
        }).join('');
    }

    // Matchups
    const mu = document.getElementById('lg-matchups');
    if (!data.matchups || !data.matchups.length) {
        mu.innerHTML = '<p style="color:#4b5563;font-size:0.85rem;">No matchups scheduled yet.</p>';
    } else {
        const teamsById = {};
        data.teams.forEach(t => { teamsById[t.id] = t; });
        mu.innerHTML = data.matchups.map(m => {
            const t1 = teamsById[m.team1_id] || { team_name: 'Team ?', player_name: '' };
            const t2 = teamsById[m.team2_id] || { team_name: 'Team ?', player_name: '' };
            let scoreHtml = '';
            if (m.team1_return !== null && m.team1_return !== undefined) {
                scoreHtml = `${fmtRet(m.team1_return)} vs ${fmtRet(m.team2_return)}`;
            }
            return `<div class="matchup-card" onclick="pwViewMatch(${m.id})">
        <div class="d-flex justify-content-between align-items-center">
          <div>
            <span style="color:#c084fc;font-weight:600">${t1.team_name}</span>
            <span style="color:#4b5563;margin:0 0.4rem">⚔️</span>
            <span style="color:#c084fc;font-weight:600">${t2.team_name}</span>
          </div>
          <div class="d-flex align-items-center gap-2">
            ${scoreHtml ? `<span style="font-size:0.8rem;">${scoreHtml}</span>` : ''}
            <span style="color:#6b7280;font-size:0.75rem">→</span>
          </div>
        </div>
      </div>`;
        }).join('');
    }

    // Create matchup dropdowns
    const opts = data.teams.map(t => `<option value="${t.id}">${t.team_name} (${t.player_name})</option>`).join('');
    document.getElementById('mu-team1-sel').innerHTML = '<option value="">Select team 1…</option>' + opts;
    document.getElementById('mu-team2-sel').innerHTML = '<option value="">Select team 2…</option>' + opts;
    document.getElementById('mu-league-id').value = data.id;
}

// ── Create Matchup ─────────────────────────────────────────
async function pwCreateMatchup() {
    const league_id = parseInt(document.getElementById('mu-league-id').value);
    const team1_id = parseInt(document.getElementById('mu-team1-sel').value);
    const team2_id = parseInt(document.getElementById('mu-team2-sel').value);
    if (!team1_id || !team2_id || team1_id === team2_id)
        return pwStatus('pw-mu-status', '⚠️ Select two different teams.', false);
    pwStatus('pw-mu-status', '<i class="fas fa-spinner fa-spin me-1"></i>Setting up battle…');
    try {
        const data = await pwFetch('/api/wars/matchup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ league_id, team1_id, team2_id })
        });
        pwStatus('pw-mu-status', `⚔️ Matchup created! ID: <span class="mono">${data.id}</span>`);
        await pwRefreshLeague();
    } catch (e) {
        pwStatus('pw-mu-status', `❌ ${e.message}`, false);
    }
}

// ── Trash Talk Messages ────────────────────────────────────
async function loadLeagueMessages(league_id) {
    clearInterval(PW.msgRefreshTimer);
    await renderMessages(league_id);
    PW.msgRefreshTimer = setInterval(() => renderMessages(league_id), 15000);
}

async function renderMessages(league_id) {
    try {
        const msgs = await pwFetch(`/api/wars/messages/${league_id}`);
        const feed = document.getElementById('chat-feed');
        if (!feed) return;
        if (!msgs.length) {
            feed.innerHTML = '<p style="color:#4b5563;font-size:0.85rem;text-align:center;padding:1rem;">No trash talk yet. Start talking! 🗣️</p>';
            return;
        }
        feed.innerHTML = msgs.map(m => `
      <div class="chat-bubble">
        <div class="chat-name">💬 ${m.player_name}</div>
        <div class="chat-text">${m.message}</div>
        <div class="d-flex gap-2 align-items-center chat-meta">
          <span>${timeAgo(m.created_at)}</span>
          <button class="like-btn" onclick="likeMsg(${m.id},this)">❤️ ${m.likes}</button>
        </div>
      </div>`).join('');
    } catch (e) { /* silent */ }
}

async function pwPostMessage() {
    const msgEl = document.getElementById('chat-input');
    const nameEl = document.getElementById('chat-name-input');
    const msg = msgEl.value.trim();
    const name = nameEl.value.trim() || PW.playerName || 'Anonymous';
    if (!msg) return;
    try {
        await pwFetch('/api/wars/message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ league_id: PW.leagueId, player_name: name, message: msg })
        });
        msgEl.value = '';
        await renderMessages(PW.leagueId);
    } catch (e) { /* silent */ }
}

async function likeMsg(id, btn) {
    try {
        const data = await pwFetch(`/api/wars/message/${id}/like`, { method: 'POST' });
        btn.textContent = `❤️ ${data.likes}`;
    } catch (e) { /* silent */ }
}

// ══════════════════════════════════════════════════════════
//  MATCH (BATTLE) SCREEN
// ══════════════════════════════════════════════════════════
async function pwViewMatch(matchup_id) {
    PW.matchupId = matchup_id;
    showWarsScreen('match');
    document.getElementById('match-content').innerHTML = '<p style="color:#9ca3af;text-align:center;padding:3rem"><i class="fas fa-spinner fa-spin fa-2x"></i></p>';
    document.getElementById('match-score-id').value = matchup_id;
    await pwScoreMatchup(matchup_id);
}

async function pwScoreMatchup(matchup_id) {
    matchup_id = matchup_id || parseInt(document.getElementById('match-score-id').value);
    if (!matchup_id) return;
    const content = document.getElementById('match-content');
    content.innerHTML = '<p style="color:#9ca3af;text-align:center;padding:2rem"><i class="fas fa-spinner fa-spin fa-2x me-2"></i>Fetching real market data…</p>';
    try {
        const d = await pwFetch(`/api/wars/score/${matchup_id}`, { method: 'POST' });

        const t1win = d.winner_id === d.team1.id;
        const t2win = d.winner_id === d.team2.id;

        function barHtml(ret) {
            const abs = Math.abs(parseFloat(ret) || 0);
            const pct = Math.min(abs * 8, 100);
            const cls = ret >= 0 ? 'bar-pos' : 'bar-neg';
            return `<div class="stock-bar-bg"><div class="${cls}" style="width:${pct}%"></div></div>`;
        }

        function teamPanel(team, score, isWinner) {
            const rows = Object.entries(team.breakdown || {}).map(([sym, ret]) => `
        <div class="stock-bar-row">
          <div class="stock-bar-label">
            <span class="stag">${sym}</span>${fmtRet(ret)}
          </div>
          ${barHtml(ret)}
        </div>`).join('');
            const winClass = isWinner ? 'winning' : 'losing';
            const badge = isWinner ? '<div class="winner-badge mt-2">🏆 WINNER</div>' : '';
            const scoreClass = isWinner ? 'ret-pos' : 'ret-neg';
            return `
        <div class="scoreboard-side ${winClass}">
          <div class="team-name-big mb-1">${team.player || 'Team'}</div>
          <div class="score-big ${scoreClass}">${score >= 0 ? '+' : ''}${parseFloat(score).toFixed(2)}%</div>
          <div style="font-size:0.75rem;color:#6b7280">avg weekly return</div>
          ${badge}
          <hr style="border-color:rgba(255,255,255,0.07);margin:1rem 0">
          ${rows}
        </div>`;
        }

        content.innerHTML = `
      <div class="row g-3 mb-3">
        <div class="col-md-5">${teamPanel(d.team1, d.team1.avg_return_pct, t1win)}</div>
        <div class="col-md-2 d-flex align-items-center justify-content-center">
          <div class="vs-divider text-center">
            <div style="font-size:2rem">⚔️</div>
            <div style="font-size:0.75rem;color:#4b5563">VS</div>
          </div>
        </div>
        <div class="col-md-5">${teamPanel(d.team2, d.team2.avg_return_pct, t2win)}</div>
      </div>
      <p style="text-align:center;font-size:0.78rem;color:#4b5563">
        Week: ${d.week_start} &nbsp;·&nbsp; Matchup #${d.matchup_id} &nbsp;·&nbsp;
        <a href="#" style="color:#a855f7" onclick="pwScoreMatchup(${d.matchup_id});return false">Recalculate</a>
      </p>`;
    } catch (e) {
        content.innerHTML = `<p style="color:#f87171;text-align:center;padding:2rem">❌ ${e.message}</p>`;
    }
}

// ── Original analyzer functions ────────────────────────────
const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
const formatPercent = (val) => (val * 100).toFixed(1) + '%';

async function analyzeStock() {
    const symbol = document.getElementById('symbolInput').value.toUpperCase();
    if (!symbol) return;
    document.getElementById('loading').style.display = 'block';
    document.getElementById('dashboard').classList.add('d-none');
    document.getElementById('errorAlert').classList.add('d-none');
    try {
        const res = await fetch(`/api/stock/analysis?symbol=${symbol}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('stockTitle').textContent = data.symbol;
        document.getElementById('stockPrice').textContent = formatCurrency(data.current_price);
        updateSignal('combinedSignal', data.trading_signals?.combined_signal || '--');
        document.getElementById('confidence').textContent = `Confidence: ${data.trading_signals?.confidence ? formatPercent(data.trading_signals.confidence) : '--'}`;
        updateSignal('sentimentSignal', data.trading_signals?.sentiment_signal || '--');
        document.getElementById('sentimentScore').textContent = `Score: ${(data.sentiment_score ?? '--') !== '--' ? data.sentiment_score.toFixed(2) : '--'}`;
        updateSignal('trendSignal', data.trading_signals?.trend_signal || '--');
        document.getElementById('rsiValue').textContent = data.technical_analysis?.rsi ? data.technical_analysis.rsi.toFixed(2) : '--';
        document.getElementById('macdStatus').textContent = (data.technical_analysis?.macd || '--').toUpperCase();
        document.getElementById('supportLevel').textContent = data.technical_analysis?.support_level ? formatCurrency(data.technical_analysis.support_level) : '--';
        document.getElementById('resistanceLevel').textContent = data.technical_analysis?.resistance_level ? formatCurrency(data.technical_analysis.resistance_level) : '--';
        document.getElementById('marketRegime').textContent = data.risk_assessment?.market_regime || '--';
        document.getElementById('volatilityAtr').textContent = data.risk_metrics?.volatility_atr ? data.risk_metrics.volatility_atr.toFixed(2) : (data.risk_assessment?.volatility || '--');
        document.getElementById('positionSize').textContent = data.risk_assessment.recommended_position_size.toUpperCase();
        document.getElementById('loading').style.display = 'none';
        document.getElementById('dashboard').classList.remove('d-none');
    } catch (err) {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('errorAlert').textContent = `Error: ${err.message}`;
        document.getElementById('errorAlert').classList.remove('d-none');
    }
}

function updateSignal(elementId, signal) {
    const el = document.getElementById(elementId);
    el.textContent = signal;
    el.className = 'metric-value mt-2';
    if (signal.includes('BUY') || signal === 'UP') el.classList.add('signal-buy');
    else if (signal.includes('SELL') || signal === 'DOWN') el.classList.add('signal-sell');
    else el.classList.add('signal-hold');
}

async function loadExtras() {
    try {
        const res = await fetch('/api/insights/daily');
        const data = await res.json();
        if (data.status !== 'maintenance' && data.status !== 'unavailable') {
            document.getElementById('dailyInsights').innerHTML = `<p class="lead">${data.summary}</p><small class="text-muted">Market Trend: ${data.trend}</small>`;
        } else {
            document.getElementById('dailyInsights').innerHTML = '<p class="text-warning">Insights temporarily unavailable.</p>';
        }
    } catch (e) { console.error(e); }
    try {
        const res = await fetch('/api/model/performance');
        const data = await res.json();
        if (!data.error) {
            const trend = data.models.trend_classifier;
            const sent = data.models.sentiment_classifier;
            document.getElementById('modelMetrics').innerHTML = `
        <div class="mb-2"><strong>Trend Accuracy:</strong> ${formatPercent(trend.accuracy)}</div>
        <div><strong>Sentiment Accuracy:</strong> ${formatPercent(sent.accuracy)}</div>
        <small class="text-muted mt-2 d-block">Updated: ${data.last_updated.split('T')[0]}</small>`;
        }
    } catch (e) { console.error(e); }
}

async function callApi(url) {
    const output = document.getElementById('apiOutput');
    output.textContent = 'Fetching data...';
    output.parentElement.scrollIntoView({ behavior: 'smooth' });
    try {
        const res = await fetch(url);
        const data = await res.json();
        output.textContent = JSON.stringify(data, null, 2);
    } catch (err) { output.textContent = 'Error: ' + err.message; }
}

async function callBacktest() {
    const output = document.getElementById('apiOutput');
    output.textContent = 'Running backtest simulation...';
    output.parentElement.scrollIntoView({ behavior: 'smooth' });
    const symbol = (document.getElementById('symbolInput').value || 'AAPL').toUpperCase();
    try {
        const res = await fetch('/api/backtest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol, strategy: 'momentum' }) });
        const data = await res.json();
        output.textContent = JSON.stringify(data, null, 2);
    } catch (err) { output.textContent = 'Error: ' + err.message; }
}

// Init
loadExtras();
