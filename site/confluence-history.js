// Build a descriptive history pool without changing archived predictions.
function confluenceHistory(data) {
  const groups = new Map();
  let excluded = 0;
  for (const [kind, weeks] of [['replay', data.backtest?.weeks || []], ['saved', data.savedWeeks || []]]) {
    for (const week of weeks) {
      if (Number(week.season) > Number(data.season) ||
          (Number(week.season) === Number(data.season) && Number(week.week) >= Number(data.week))) continue;
      for (const row of week.rows || []) {
        if (!row.player_id) { excluded++; continue; }
        // One observation per player/week, including overlapping archive sources.
        const key = `${week.season}-${week.week}-${row.player_id}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push({row, kind, season:week.season, week:week.week});
      }
    }
  }
  const rows = [];
  const includedWeeks = new Set();
  for (const group of groups.values()) {
    // Exclude every ambiguous duplicate; never choose a favorable prediction.
    if (group.length !== 1) { excluded += group.length; continue; }
    const {row, kind, season, week} = group[0];
    if (kind === 'saved' && (row.snapshot_status !== 'frozen' ||
        !(Date.parse(row.actual_snapshot_utc) < Date.parse(row.scheduled_kickoff || row.kickoff)))) {
      excluded++; continue;
    }
    const status = Number(row.void) === 1 ? 'void' : kind === 'saved'
      ? (['hit', 'miss', 'pending', 'void'].includes(row.result_status) ? row.result_status : 'pending')
      : row.scored_td === 1 ? 'hit' : row.scored_td === 0 ? 'miss' : 'pending';
    rows.push({...row, result_status:status});
    includedWeeks.add(`${season} Week ${week}`);
  }
  return {rows, excluded, weeks:includedWeeks.size};
}
if (typeof module !== 'undefined') module.exports = {confluenceHistory};
