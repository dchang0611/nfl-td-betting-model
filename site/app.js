const tbody = document.querySelector('#rows');
const search = document.querySelector('#search');
const gamesView = document.querySelector('#games-view');
const performanceView = document.querySelector('#performance-view');
const resultWeek = document.querySelector('#result-week');
const resultLimit = document.querySelector('#result-limit');
let rows = [], backtest = {metrics: []}, currentView = 'players', positionFilter = 'ALL';

const text = v => v ?? '--';
const pct = v => v == null ? '--' : `${(Number(v) * 100).toFixed(1)}%`;
const num = (v, d = 1) => v == null ? '--' : Number(v).toFixed(d);
const esc = v => String(text(v)).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const metric = (label, value) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
const positionGroup = position => position === 'FB' ? 'RB' : position;
const positionNames = {ALL:'Player breakdowns', RB:'Running backs', WR:'Wide receivers', TE:'Tight ends', QB:'Quarterbacks'};

function details(r) {
  return `<tr class="detail-row" hidden><td colspan="8"><div class="detail-panel">
    ${metric('Recent workload share', pct(r.recent_opportunities))}
    ${metric('Red-zone role', pct(r.red_zone_opp_share_ewm))}
    ${metric('Inside-10 role', pct(r.inside10_opp_share_ewm))}
    ${metric('Inside-5 role', pct(r.inside5_opp_share_ewm))}
    ${metric('Rush share', pct(r.rush_share_ewm))}
    ${metric('Target share', pct(r.target_share_ewm))}
    ${metric('Route participation', pct(r.route_participation_ewm))}
    ${metric('Role stability', pct(r.role_stability))}
    ${metric('Expert role percentile', pct(r.expert_rank_percentile))}
    ${metric('Rookie specialist estimate', pct(r.rookie_specialist_probability))}
    ${metric('Board status', esc(r.snapshot_status))}
    ${metric('Planned 24h freeze', r.planned_snapshot_utc ? esc(new Date(r.planned_snapshot_utc).toLocaleString()) : '--')}
    ${metric('Availability estimate', pct(r.availability_probability))}
    ${metric('Official game status', esc(r.game_status))}
    ${metric('Practice status', esc(r.practice_status))}
    ${metric('Injury', esc(r.injury))}
    ${metric('Team implied total', r.team_implied_total == null ? '--' : num(r.team_implied_total))}
    ${metric('Point spread', r.spread == null ? '--' : num(r.spread))}
    ${metric('Prior games', text(r.games_prior))}
    ${metric('New team', r.new_team_flag ? 'Yes' : 'No')}
    <p class="detail-note">The displayed TD probability and production ranking come from the validated full-history model. Experimental injury, expert, rule-era, and rookie estimates are shown for monitoring only.</p>
  </div></td></tr>`;
}

function renderPlayers(q = '') {
  const needle = q.toLowerCase();
  const positionRanks = new Map();
  const counters = {};
  rows.forEach(r => {
    const group = positionGroup(r.position);
    counters[group] = (counters[group] || 0) + 1;
    positionRanks.set(r.player_id, counters[group]);
  });
  const filtered = rows.filter(r =>
    (positionFilter === 'ALL' || positionGroup(r.position) === positionFilter)
    && JSON.stringify(r).toLowerCase().includes(needle)
  );
  tbody.innerHTML = filtered.length ? filtered.map((r, i) => `
    <tr><td class="rank">#${esc(r.ranking)}</td><td class="rank">#${esc(positionRanks.get(r.player_id))} ${esc(positionGroup(r.position))}</td><td><strong>${esc(r.player_name)}</strong><br><small>${esc(r.position)} &middot; ${esc(r.team)}</small></td>
    <td><strong>${esc(r.team)} vs ${esc(r.opponent_team)}</strong><br><small>${esc(r.home_away)}</small></td>
    <td class="prob">${pct(r.model_probability)}</td><td><div class="confidence"><div class="bar"><i style="width:${Math.max(1, Number(r.confidence) || 1)}%"></i></div><span>${num(r.confidence, 0)}</span></div></td>
    <td>${esc(r.model_note)}</td><td><button class="details-btn" data-i="${i}" aria-expanded="false">Breakdown</button></td></tr>${details(r)}`).join('')
    : `<tr><td colspan="8" class="muted">No ${esc(positionNames[positionFilter].toLowerCase())} appear in the overall Top 60.</td></tr>`;
  document.querySelectorAll('.details-btn').forEach(btn => btn.addEventListener('click', () => {
    const detail = btn.closest('tr').nextElementSibling;
    detail.hidden = !detail.hidden;
    btn.textContent = detail.hidden ? 'Breakdown' : 'Close';
    btn.setAttribute('aria-expanded', String(!detail.hidden));
  }));
}

document.querySelectorAll('.position-tab').forEach(tab => tab.addEventListener('click', () => {
  positionFilter = tab.dataset.position;
  document.querySelectorAll('.position-tab').forEach(t => t.classList.toggle('active', t === tab));
  document.querySelector('#title').textContent = positionNames[positionFilter];
  document.querySelector('#kicker').textContent = positionFilter === 'ALL' ? 'TOP 60' : `POSITION: ${positionFilter}`;
  document.querySelector('#download').href = positionFilter === 'ALL' ? 'data/latest-board.csv' : `data/latest-board-${positionFilter.toLowerCase()}.csv`;
  renderPlayers(search.value);
}));

function renderGames(q = '') {
  const needle = q.toLowerCase(), filtered = rows.filter(r => JSON.stringify(r).toLowerCase().includes(needle)), games = new Map();
  filtered.forEach(r => { const key = r.nfl_game_id || [r.team, r.opponent_team].sort().join('-'); if (!games.has(key)) games.set(key, []); games.get(key).push(r); });
  gamesView.innerHTML = [...games.values()].map(game => {
    const first = game[0], teams = [...new Set(game.flatMap(r => [r.team, r.opponent_team]))];
    const sides = teams.map(team => `<section class="team-side"><h4>${esc(team)}</h4>${game.filter(r => r.team === team).slice(0, 8).map(r => `<div class="player-line"><span class="rank">#${esc(r.ranking)}</span><span><strong>${esc(r.player_name)}</strong><br><small>${esc(r.position)}</small></span><strong class="prob">${pct(r.model_probability)}</strong></div>`).join('') || '<small>No ranked players</small>'}</section>`).join('');
    const kick = first.kickoff ? new Date(first.kickoff).toLocaleString([], {weekday:'short', hour:'numeric', minute:'2-digit'}) : 'Time TBD';
    return `<article class="game-card"><div class="game-head"><div><small>${esc(kick)}</small><h3>${esc(teams.join(' vs '))}</h3></div><div class="muted">Game total ${first.game_total == null ? '--' : num(first.game_total)}<br>Top-ranked player: #${esc(Math.min(...game.map(r => r.ranking)))}</div></div><div class="game-list">${sides}</div></article>`;
  }).join('') || '<p class="error">No games match that search.</p>';
}

function renderWeekResults() {
  const weeks = backtest.weeks || [];
  const selected = weeks.find(item => String(item.week) === resultWeek.value) || weeks[0];
  if (!selected) {
    document.querySelector('#week-summary').innerHTML = '';
    document.querySelector('#result-rows').innerHTML = '<tr><td colspan="5" class="muted">No weekly replay data is available.</td></tr>';
    return;
  }
  const limit = Number(resultLimit.value || 10);
  const sample = selected.rows.filter(r => Number(r.board_rank) <= limit);
  const graded = sample.filter(r => Number(r.void) !== 1);
  const hits = graded.filter(r => Number(r.scored_td) === 1).length;
  const voids = sample.length - graded.length;
  document.querySelector('#week-summary').innerHTML = `
    <span class="result-chip"><strong>${hits} / ${graded.length}</strong> hit</span>
    <span class="result-chip"><strong>${pct(graded.length ? hits / graded.length : null)}</strong> hit rate</span>
    <span class="result-chip"><strong>${voids}</strong> void${voids === 1 ? '' : 's'}</span>`;
  document.querySelector('#result-rows').innerHTML = sample.map(r => {
    const isVoid = Number(r.void) === 1;
    const isHit = Number(r.scored_td) === 1 && !isVoid;
    const touchdowns = Number(r.td_count) || 0;
    const outcomeClass = isVoid ? 'void' : isHit ? 'hit' : 'miss';
    const outcomeText = isVoid ? 'Void / did not play' : isHit ? `Hit${touchdowns > 1 ? ` (${touchdowns} TDs)` : ''}` : 'Miss';
    return `<tr><td class="rank">#${esc(r.board_rank)}</td><td><strong>${esc(r.player_name)}</strong><br><small>${esc(r.position)} &middot; ${esc(r.team)}</small></td><td>${esc(r.team)} vs ${esc(r.opponent_team)}</td><td class="prob">${pct(r.model_probability)}</td><td><span class="outcome ${outcomeClass}">${outcomeText}</span></td></tr>`;
  }).join('');
}

function renderPerformance() {
  const names = {legacy_features:'Production ranking (selected)', legacy_plus_rookie_specialist:'Rookie specialist (probability monitor)', availability:'Workload redistribution (rejected)', rules_environment_only:'Rule/environment features (rejected)', rookie_pathway_only:'Blended rookie pathway (rejected)', history_2024_plus:'2024+ history only (rejected)', history_recency_weighted:'Recency weighting (rejected)', expert_role:'Expert rerank (rejected)'};
  const metrics = backtest.metrics || [];
  document.querySelector('#performance-cards').innerHTML = metrics.map(r => `<article class="performance-card"><h3>${esc(names[r.variant] || r.variant)}</h3>
    <div class="performance-stat"><span>Top-5 hits</span><strong>${esc(r.top5_hits)} / ${esc(r.top5_graded)}</strong></div>
    <div class="performance-stat"><span>Top-5 hit rate</span><strong>${pct(r.top5_precision)}</strong></div>
    <div class="performance-stat"><span>Top-10 hits</span><strong>${esc(r.top10_hits)} / ${esc(r.top10_graded)}</strong></div>
    <div class="performance-stat"><span>Top-10 hit rate</span><strong>${pct(r.top10_precision)}</strong></div>
    <div class="performance-stat"><span>Top-10 voids</span><strong>${esc(r.top10_voids)}</strong></div>
    <div class="performance-stat"><span>Candidate AUC</span><strong>${num(r.candidate_auc, 3)}</strong></div></article>`).join('') || '<p class="error">Run the optional 2025 fixed-board backtest once to populate this view.</p>';
  renderWeekResults();
}

function render() {
  if (currentView === 'players') renderPlayers(search.value);
  else if (currentView === 'games') renderGames(search.value);
  else renderPerformance();
}

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  currentView = tab.dataset.view;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
  document.querySelector('#players-view').hidden = currentView !== 'players';
  gamesView.hidden = currentView !== 'games';
  performanceView.hidden = currentView !== 'performance';
  search.hidden = currentView === 'performance';
  document.querySelector('#position-tabs').hidden = currentView !== 'players';
  document.querySelector('#kicker').textContent = currentView === 'performance' ? 'FIXED PREGAME TEST' : 'TOP 60';
  document.querySelector('#title').textContent = currentView === 'players' ? positionNames[positionFilter] : currentView === 'games' ? 'Games and matchups' : 'Model performance';
  if (currentView === 'players' && positionFilter !== 'ALL') document.querySelector('#kicker').textContent = `POSITION: ${positionFilter}`;
  render();
}));

fetch('data/board.json', {cache:'no-store'}).then(r => { if (!r.ok) throw Error(); return r.json(); }).then(data => {
  rows = data.rows || []; backtest = data.backtest || {metrics: []};
  resultWeek.innerHTML = (backtest.weeks || []).map(item => `<option value="${item.week}">Week ${item.week}</option>`).join('');
  document.querySelectorAll('.position-tab').forEach(tab => {
    const position = tab.dataset.position;
    const count = position === 'ALL' ? rows.length : rows.filter(r => positionGroup(r.position) === position).length;
    tab.textContent = `${tab.textContent} (${count})`;
  });
  document.querySelector('#slate').textContent = data.label;
  document.querySelector('#count').textContent = rows.length;
  document.querySelector('#updated').textContent = new Date(data.updatedAt).toLocaleString();
  render();
}).catch(() => document.querySelector('#error').hidden = false);
search.addEventListener('input', render);
resultWeek.addEventListener('change', renderWeekResults);
resultLimit.addEventListener('change', renderWeekResults);
