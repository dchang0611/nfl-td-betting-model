const tbody = document.querySelector('#rows');
const search = document.querySelector('#search');
const gamesView = document.querySelector('#games-view');
const confluenceView = document.querySelector('#confluence-view');
const performanceView = document.querySelector('#performance-view');
const resultWeek = document.querySelector('#result-week');
const resultLimit = document.querySelector('#result-limit');
const boardWeek = document.querySelector('#board-week');
let availableWeeks = [], selectedWeek = null, downloadUrl = null;
const weekKey = w => `${w.kind}-${w.season}-${w.week}`;
const weekLabel = w => `${w.season} Week ${w.week} · ${w.kind === 'replay' ? 'Replay' : 'Saved board'}`;
const outcome = r => r.result_status || (Number(r.void) === 1 ? 'void' : r.scored_td == null ? 'pending' : Number(r.scored_td) === 1 ? 'hit' : 'miss');
let rows = [], backtest = {metrics: []}, currentView = 'players', positionFilter = 'ALL';
let factorDefinitions = [];
const confluenceState = {selected: ['Red-zone role', 'Inside-10 role', 'Goal-line / end-zone role'], minMatches: 2};

const text = v => v ?? '--';
const pct = v => v == null ? '--' : `${(Number(v) * 100).toFixed(1)}%`;
const num = (v, d = 1) => v == null ? '--' : Number(v).toFixed(d);
const esc = v => String(text(v)).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const metric = (label, value) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
const positionGroup = position => position === 'FB' ? 'RB' : position;
const positionNames = {ALL:'Player breakdowns', RB:'Running backs', WR:'Wide receivers', TE:'Tight ends', QB:'Quarterbacks'};
const positionLabels = {ALL:'All', RB:'Running backs', WR:'Wide receivers', TE:'Tight ends', QB:'Quarterbacks'};
const factorValues = (row, factor) => factor.fields.map(field => row[field]).filter(value => value !== null && value !== undefined && value !== '').map(Number).filter(Number.isFinite);
const factorValue = (row, factor) => Math.max(...factorValues(row, factor));
const factorHasData = (row, factor) => factorValues(row, factor).length > 0;
const factorMatches = (row, factor) => factorValue(row, factor) >= Number(factor.threshold);
const matchedFactors = row => factorDefinitions.filter(factor => factorMatches(row, factor)).map(factor => factor.label);
const selectedMatches = row => matchedFactors(row).filter(label => confluenceState.selected.includes(label));
const factorChips = row => confluenceState.selected.map(label => {
  const matched = selectedMatches(row).includes(label);
  const factor = factorDefinitions.find(item => item.label === label);
  return `<span class="factor-chip ${matched ? 'matched' : ''}" title="${esc(factor?.description)}">${matched ? '&#10003;' : '&#8212;'} ${esc(label)}</span>`;
}).join('');
const allFactorChips = row => factorDefinitions.map(factor => {
  const matched = factorMatches(row, factor);
  return `<span class="factor-chip ${matched ? 'matched' : ''}" title="${esc(factor.description)}">${matched ? '&#10003;' : '&#8212;'} ${esc(factor.label)}</span>`;
}).join('');
const renderFactorRead = row => `<div class="factor-read"><strong>${matchedFactors(row).length}/${factorDefinitions.length}</strong><div class="factor-chips">${allFactorChips(row)}</div></div>`;

function renderConfluenceControls() {
  document.querySelector('#factor-selectors').innerHTML = factorDefinitions.map(({label, description}) => `<button type="button" class="factor-selector ${confluenceState.selected.includes(label) ? 'active' : ''}" data-factor="${esc(label)}" aria-pressed="${confluenceState.selected.includes(label)}" title="${esc(description)}">${esc(label)}</button>`).join('');
  if (confluenceState.minMatches > confluenceState.selected.length) confluenceState.minMatches = confluenceState.selected.length;
  document.querySelector('#factor-min').innerHTML = Array.from({length:confluenceState.selected.length}, (_, i) => i + 1).map(count => `<option value="${count}" ${count === confluenceState.minMatches ? 'selected' : ''}>At least ${count} of ${confluenceState.selected.length}</option>`).join('');
}

function renderConfluence() {
  renderConfluenceControls();
  const allReplay = (backtest.weeks || []).flatMap(week => week.rows || []).filter(r => positionFilter === 'ALL' || positionGroup(r.position) === positionFilter);
  const gradedAll = allReplay.filter(r => Number(r.void) !== 1);
  const baselineHits = gradedAll.filter(r => Number(r.scored_td) === 1).length;
  const baseline = gradedAll.length ? baselineHits / gradedAll.length : null;
  const replay = allReplay.filter(r => selectedMatches(r).length >= confluenceState.minMatches);
  const graded = replay.filter(r => Number(r.void) !== 1);
  const hits = graded.filter(r => Number(r.scored_td) === 1).length;
  const misses = graded.length - hits;
  const hitRate = graded.length ? hits / graded.length : null;
  const lift = hitRate == null || baseline == null ? null : hitRate - baseline;
  const label = confluenceState.selected.join(' + ');
  document.querySelector('#confluence-performance').innerHTML = `<tr><td><strong>${esc(label)}</strong><br><small>AT LEAST ${confluenceState.minMatches} OF ${confluenceState.selected.length}</small></td><td>${hits}-${misses}</td><td class="prob">${pct(hitRate)}</td><td>${pct(baseline)}</td><td class="${lift > 0 ? 'lift-positive' : lift < 0 ? 'lift-negative' : ''}">${lift == null ? '--' : `${lift >= 0 ? '+' : ''}${(lift * 100).toFixed(1)} pts`}</td><td>${graded.length}</td><td>${replay.length - graded.length}</td></tr>`;
  const current = rows.filter(r => (positionFilter === 'ALL' || positionGroup(r.position) === positionFilter) && selectedMatches(r).length >= confluenceState.minMatches);
  document.querySelector('#confluence-summary').innerHTML = `<span class="result-chip"><strong>${current.length}</strong> selected-week qualifiers</span><span class="result-chip"><strong>${graded.length}</strong> replay sample</span><span class="result-chip"><strong>${pct(hitRate)}</strong> historical hit rate</span>`;
  document.querySelector('#confluence-rows').innerHTML = current.length ? current.map(r => `<tr><td class="rank">#${esc(r.ranking)}</td><td><strong>${esc(r.player_name)}</strong><br><small>${esc(r.position)} &middot; ${esc(r.team)}</small></td><td>${esc(r.team)} vs ${esc(r.opponent_team)}</td><td class="prob">${pct(r.model_probability)}</td><td class="confluence-score">${selectedMatches(r).length}/${confluenceState.selected.length}</td><td><div class="factor-chips">${factorChips(r)}</div></td></tr>`).join('') : '<tr><td colspan="6" class="muted">No selected-week players meet this factor combination.</td></tr>';
  renderFactorPerformance(allReplay, baseline);
}

function renderFactorPerformance(allReplay, baseline) {
  const current = rows.filter(r => positionFilter === 'ALL' || positionGroup(r.position) === positionFilter);
  const results = factorDefinitions.map(factor => {
    const available = allReplay.filter(row => factorHasData(row, factor));
    const matched = available.filter(row => factorMatches(row, factor));
    const graded = matched.filter(row => Number(row.void) !== 1);
    const hits = graded.filter(row => Number(row.scored_td) === 1).length;
    const rate = graded.length ? hits / graded.length : null;
    return {factor, available, matched, graded, hits, rate, lift:rate == null || baseline == null ? null : rate - baseline, pending:current.filter(row => factorMatches(row, factor)).length};
  });
  const fullyArchived = results.filter(item => item.available.length === allReplay.length).length;
  document.querySelector('#factor-performance-notice').textContent = `${allReplay.length} frozen replay rows are tracked for ${positionFilter === 'ALL' ? 'all positions' : positionFilter}. ${fullyArchived} of ${factorDefinitions.length} factors have complete archived inputs; incomplete factors show their available coverage.`;
  document.querySelector('#factor-performance-rows').innerHTML = results.map(({factor, available, matched, graded, hits, rate, lift, pending}) => {
    const misses = graded.length - hits;
    const coverage = allReplay.length ? available.length / allReplay.length : null;
    return `<tr><td><strong>${esc(factor.label)}</strong><br><small>${esc(factor.description)} · ${pct(coverage)} input coverage</small></td><td>${graded.length ? `${hits}-${misses}` : '--'}</td><td>${pct(rate)}</td><td class="${lift > 0 ? 'lift-positive' : lift < 0 ? 'lift-negative' : ''}">${lift == null ? '--' : `${lift >= 0 ? '+' : ''}${(lift * 100).toFixed(1)} pts`}</td><td>${graded.length}</td><td>${matched.length - graded.length}</td><td>${pending}</td></tr>`;
  }).join('');
}

function updatePositionTabCounts(sourceRows) {
  document.querySelectorAll('.position-tab').forEach(tab => {
    const position = tab.dataset.position;
    const count = position === 'ALL' ? sourceRows.length : sourceRows.filter(r => positionGroup(r.position) === position).length;
    tab.textContent = `${positionLabels[position]} (${count})`;
  });
}

function details(r) {
  return `<tr class="detail-row" hidden><td colspan="8"><div class="detail-panel">
    <div class="factor-detail"><span>Confluence read</span><div class="factor-chips">${allFactorChips(r)}</div></div>
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
    <p class="detail-note">The Player Board read and Factor Confluence are generated from the same seven definitions and thresholds. The TD probability remains the production model output; confluence is its descriptive factor summary.</p>
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
    <td>${renderFactorRead(r)}</td><td><button class="details-btn" data-i="${i}" aria-expanded="false">Breakdown</button></td></tr>${details(r)}`).join('')
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
  document.querySelector('#title').textContent = currentView === 'players' ? positionNames[positionFilter] : currentView === 'confluence' ? 'Factor confluence' : 'Historical performance';
  document.querySelector('#kicker').textContent = currentView === 'performance' ? (positionFilter === 'ALL' ? 'WEEKLY RESULTS: ALL POSITIONS' : `WEEKLY RESULTS: ${positionFilter}`) : currentView === 'confluence' ? (positionFilter === 'ALL' ? 'SELECTED BOARD + 2025 REPLAY' : `CONFLUENCE: ${positionFilter}`) : (positionFilter === 'ALL' ? 'TOP 60' : `POSITION: ${positionFilter}`);
  render();
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
  const selected = selectedWeek;
  if (!selected) {
    document.querySelector('#week-summary').innerHTML = '';
    document.querySelector('#result-rows').innerHTML = '<tr><td colspan="5" class="muted">No weekly replay data is available.</td></tr>';
    return;
  }
  const limit = Number(resultLimit.value || 10);
  const positionRows = selected.rows.filter(r => positionFilter === 'ALL' || positionGroup(r.position) === positionFilter);
  const sample = positionFilter === 'ALL' ? positionRows.filter(r => Number(r.board_rank) <= limit) : positionRows.slice(0, limit);
  updatePositionTabCounts(selected.rows);
  const graded = sample.filter(r => ['hit', 'miss'].includes(outcome(r)));
  const hits = graded.filter(r => outcome(r) === 'hit').length;
  const voids = sample.filter(r => outcome(r) === 'void').length;
  const pending = sample.filter(r => outcome(r) === 'pending').length;
  document.querySelector('#results-source').textContent = weekLabel(selected);
  document.querySelector('#week-summary').innerHTML = `
    <span class="result-chip"><strong>${positionFilter === 'ALL' ? 'Overall' : positionFilter}</strong> Top ${Math.min(limit, sample.length)}</span>
    <span class="result-chip"><strong>${hits} / ${graded.length}</strong> hit</span>
    <span class="result-chip"><strong>${pct(graded.length ? hits / graded.length : null)}</strong> hit rate</span>
    <span class="result-chip"><strong>${voids}</strong> void${voids === 1 ? '' : 's'}</span>
    <span class="result-chip"><strong>${pending}</strong> pending</span>`;
  document.querySelector('#result-rows').innerHTML = sample.map((r, index) => {
    const status = outcome(r);
    const isVoid = status === 'void';
    const isHit = status === 'hit';
    const touchdowns = Number(r.td_count) || 0;
    const outcomeClass = status === 'pending' ? 'void' : status;
    const outcomeText = status === 'pending' ? 'Pending' : isVoid ? 'Void / no offensive snaps' : isHit ? `Hit${touchdowns > 1 ? ` (${touchdowns} TDs)` : ''}` : 'Miss';
    const rank = positionFilter === 'ALL' ? `#${esc(r.board_rank)}` : `#${index + 1} ${esc(positionFilter)}<br><small>#${esc(r.board_rank)} overall</small>`;
    return `<tr><td class="rank">${rank}</td><td><strong>${esc(r.player_name)}</strong><br><small>${esc(r.position)} &middot; ${esc(r.team)}</small></td><td>${esc(r.team)} vs ${esc(r.opponent_team)}</td><td class="prob">${pct(r.model_probability)}</td><td><span class="outcome ${outcomeClass}">${outcomeText}</span></td></tr>`;
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
  updateDownload();
  if (currentView === 'players') renderPlayers(search.value);
  else if (currentView === 'games') renderGames(search.value);
  else if (currentView === 'confluence') renderConfluence();
  else renderPerformance();
}

function updateDownload() {
  if (!selectedWeek) return;
  const sample = rows.filter(r => positionFilter === 'ALL' || positionGroup(r.position) === positionFilter);
  const columns = [...new Set(sample.flatMap(r => Object.keys(r)))];
  const csvCell = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const csv = [columns.map(csvCell).join(','), ...sample.map(r => columns.map(c => csvCell(r[c])).join(','))].join('\r\n');
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
  const link = document.querySelector('#download');
  link.href = downloadUrl;
  link.download = `nfl-td-${weekKey(selectedWeek)}-${positionFilter.toLowerCase()}.csv`;
}

function selectWeek(key) {
  selectedWeek = availableWeeks.find(w => weekKey(w) === key) || availableWeeks[0];
  if (!selectedWeek) return;
  boardWeek.value = resultWeek.value = weekKey(selectedWeek);
  rows = selectedWeek.rows.map(r => ({...r, ranking:r.ranking ?? r.board_rank}));
  updatePositionTabCounts(rows);
  document.querySelector('#slate').textContent = weekLabel(selectedWeek);
  document.querySelector('#count').textContent = rows.length;
  document.querySelector('#updated').textContent = selectedWeek.generatedAt ? new Date(selectedWeek.generatedAt).toLocaleString() : 'Archived replay';
  render();
}

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  currentView = tab.dataset.view;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
  document.querySelector('#players-view').hidden = currentView !== 'players';
  gamesView.hidden = currentView !== 'games';
  confluenceView.hidden = currentView !== 'confluence';
  performanceView.hidden = currentView !== 'performance';
  search.hidden = currentView === 'performance' || currentView === 'confluence';
  document.querySelector('#position-tabs').hidden = currentView === 'games';
  document.querySelector('#kicker').textContent = currentView === 'performance' ? 'FIXED PREGAME TEST' : 'TOP 60';
  document.querySelector('#title').textContent = currentView === 'players' ? positionNames[positionFilter] : currentView === 'games' ? 'Games and matchups' : currentView === 'confluence' ? 'Factor confluence' : 'Historical performance';
  if (currentView === 'players') {
    updatePositionTabCounts(rows);
    if (positionFilter !== 'ALL') document.querySelector('#kicker').textContent = `POSITION: ${positionFilter}`;
  }
  if (currentView === 'performance') document.querySelector('#kicker').textContent = positionFilter === 'ALL' ? 'WEEKLY RESULTS: ALL POSITIONS' : `WEEKLY RESULTS: ${positionFilter}`;
  if (currentView === 'confluence') document.querySelector('#kicker').textContent = positionFilter === 'ALL' ? 'SELECTED BOARD + 2025 REPLAY' : `CONFLUENCE: ${positionFilter}`;
  render();
}));

fetch('data/board.json', {cache:'no-store'}).then(r => { if (!r.ok) throw Error(); return r.json(); }).then(data => {
  rows = data.rows || []; backtest = data.backtest || {metrics: []}; factorDefinitions = data.factorDefinitions || [];
  if (!factorDefinitions.length) throw Error('Factor definitions are missing from the board data.');
  const saved = data.savedWeeks?.length ? data.savedWeeks : [{season:data.season, week:data.week, rows:data.rows, generatedAt:data.updatedAt, kind:'saved'}];
  availableWeeks = [...saved, ...(backtest.weeks || []).map(w => ({...w, kind:'replay'}))].sort((a,b) => b.season-a.season || b.week-a.week);
  boardWeek.innerHTML = resultWeek.innerHTML = availableWeeks.map(w => `<option value="${weekKey(w)}">${esc(weekLabel(w))}</option>`).join('');
  selectWeek(weekKey(availableWeeks[0]));
}).catch(() => document.querySelector('#error').hidden = false);
search.addEventListener('input', render);
resultWeek.addEventListener('change', () => selectWeek(resultWeek.value));
boardWeek.addEventListener('change', () => selectWeek(boardWeek.value));
resultLimit.addEventListener('change', renderWeekResults);
document.querySelector('#factor-selectors').addEventListener('click', event => {
  const button = event.target.closest('.factor-selector');
  if (!button) return;
  const factor = button.dataset.factor;
  if (confluenceState.selected.includes(factor)) {
    if (confluenceState.selected.length === 1) return;
    confluenceState.selected = confluenceState.selected.filter(label => label !== factor);
  } else confluenceState.selected = [...confluenceState.selected, factor];
  renderConfluence();
});
document.querySelector('#factor-min').addEventListener('change', event => { confluenceState.minMatches = Number(event.target.value); renderConfluence(); });
