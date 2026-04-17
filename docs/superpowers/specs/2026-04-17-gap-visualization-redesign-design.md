# Gap Visualization Redesign

**Date:** 2026-04-17
**Status:** Approved (verbal)
**Scope:** `dashboard.html` — `Performance Gap Analysis` section only.

## Goal

Bring the visual style and chart treatment of [ai-time-horizons.vercel.app](https://ai-time-horizons.vercel.app/) to the existing dashboard's Performance Gap Analysis section. Stat cards stay; the chart is rebuilt; the underlying gap calculation is corrected.

## Non-goals

- No changes to the headline trend chart, calculators, methodology section, or raw-data table.
- No new top-level navigation, dark-mode toggle, or page restructure.
- No data-pipeline changes — work is contained to `dashboard.html`.

## Reference

The reference site renders, per region, a step-curve of cumulative-max time horizon over time, with a shaded fill between the two curves representing the visible "gap". Toggles control success rate (p50/p80) and scale (linear/log). Frontier-advancing models only appear as dots.

## Calculation fixes

Two functions in `dashboard.html` (current line numbers, may shift):

### `getFrontierModels` (line ~1080)

**Bug:** When called with a pre-filtered group (e.g. all US models), it groups *by family* internally and emits each family's cumulative-max points. Concatenating across families produces a "frontier" that contains many models that are not actually leading the group at the time.

**Fix:** Replace with `frontierFor(groupMembers)` — a flat cumulative-max scan:

```
sort groupMembers by date ascending
maxH = 0
out = []
for m in sorted:
  if m.horizon > maxH:           # strict greater-than → ties don't duplicate
    maxH = m.horizon
    out.push(m)
return out
```

This returns the true leading models for the group, in date order.

### `calculateHorizontalGaps` (line ~1106)

**Bug 1:** Search loop excludes lagging models with `lagging.date <= leading.date`. So if the lagging side already exceeded the leading horizon before the leading model shipped, the leading model is wrongly classified "unmatched" (open gap).

**Bug 2:** Tie handling (`>=`) duplicates frontier points.

**Fix:** Rewrite around the corrected frontier:

```
leadFrontier = frontierFor(leadingData)        # cumulative max over leading group
lagFrontier  = frontierFor(laggingData)        # cumulative max over lagging group

for L in leadFrontier:
  # find the date when the lagging cumulative max first reached L.horizon
  crossing = first lag in lagFrontier where lag.horizon >= L.horizon
  if crossing:
    gapMonths = max(0, (crossing.date - L.date) / DAYS_PER_MONTH)
    matched = true
  else:
    gapMonths = (today - L.date) / DAYS_PER_MONTH
    matched = false
  push { L, crossing, gapMonths, matched }
```

`max(0, …)` collapses "lagging was already ahead" into a 0-month gap rather than a negative number — this matches user expectation that a gap of zero means "already matched" without polluting the average.

### Stat-card semantics after the fix

- **Average Gap**: mean of `gapMonths` over `matched=true` rows. Now honest: includes the zero-gap rows where the lagging side was already ahead, and excludes inflated unmatched rows.
- **Current Horizon Ratio**: leading-frontier latest horizon ÷ lagging-frontier latest horizon. Unchanged formula — driven by corrected frontier.
- **Matched / Unmatched**: counts based on the corrected pairing.
- **Growth Rate Comparison**: per-group OLS slope on log(horizon) vs date over each frontier. Unchanged formula.
- **Current Gap (Est.)** survival-analysis estimator: keep the existing log-normal censored estimator; it now operates on correct inputs.

## Visual redesign

### Chart engine

Add D3 v7 (`https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js`) via `<script>` in `<head>`. Plotly remains for the rest of the dashboard untouched.

### Layout

Replace `#gapPlot` with a new SVG-rendering container. Above it, an updated control row (toggle pills, replacing the existing `<button class="tab" data-gap-tab=…>` blocks) and a new custom legend row.

```
[ stat cards row — unchanged content ]
[ controls: Comparison | Success Rate | Scale | Gap Metric ] (toggle pills)
[ legend: ●─── leading | ○--- lagging | ▢ gap ]
[ SVG chart, ~400px height ]
[ existing <details> blocks: Gap Details, Model Classification ]
```

### Chart elements

- **X axis**: time, from earliest leading-frontier date to today, no chart border (`.domain { display:none }`).
- **Y axis**: time horizon. Linear and log scales both available. Tick labels formatted with the existing `formatHours` helper.
- **Two step lines** — solid for leading group, dashed for lagging.
  - Open vs Closed: Closed solid (`#457b9d`), Open dashed (`#e63946`).
  - China vs US: US solid (`#457b9d`), China dashed (`#e63946`).
- **Filled gap region** between the two step curves at color `rgba(230,57,70,0.07)` light / `rgba(242,85,97,0.10)` dark.
- **Dots** on each frontier-advancing model (the inflection points). Solid fill for leading, hollow ring for lagging.
- **Hover tooltip** showing model name, release date, horizon (formatted via `formatHours`), and group label.

### Controls (toggle pills)

Replace the existing tabs with the reference's toggle-group pattern. Four pill groups:

1. **Comparison**: `Open vs Closed` | `China vs US`
2. **Success Rate**: `80%` | `50%`
3. **Scale**: `Linear` | `Log`
4. **Gap Metric** (stat-card driver, unchanged purpose): `Average` | `Current (Est.)`

Wired to the same state variables that drive the section today; chart and stat cards both react.

### Color tokens

Add CSS custom properties scoped to the section so we can tune without hunting:

```
.gap-section {
  --gap-leading: #457b9d;
  --gap-lagging: #e63946;
  --gap-fill: rgba(230,57,70,0.07);
  --gap-grid: #f0f0f0;
  --gap-axis: #777;
}
```

No global dark-mode rework — the rest of the dashboard stays light. (Auto dark-mode for the section can be a follow-up; out of scope here.)

### Class scoping

All new CSS uses `.gap-*` class prefixes (`.gap-chart`, `.gap-toggle-group`, `.gap-toggle-pill`, `.gap-legend`, `.gap-tooltip`) inside the existing `<style>` block. Cannot leak into other sections.

## Behavior

- Switching any toggle re-renders chart + stat cards from the existing in-memory model list (no refetch).
- Resize: chart re-renders on window resize via a debounced listener (100ms).
- Empty state: if a toggle combo yields fewer than 2 frontier models per side, render a "Not enough data" message instead of an SVG.

## Implementation files

Single file: `dashboard.html`.

Changes:
1. Add D3 `<script>` in `<head>`.
2. Add `.gap-*` CSS in the existing `<style>` block.
3. Replace the section's controls markup (lines ~482-497) with toggle-pill structure.
4. Replace `#gapPlot` div with the chart container + legend.
5. Rewrite `getFrontierModels` and `calculateHorizontalGaps` (lines ~1080-1153).
6. Add a new `renderGapChart()` function called wherever `#gapPlot` is currently re-rendered.

## Testing

- Manual: toggle every combination (2 × 2 × 2 × 2 = 16). Verify the chart and all four stat cards update consistently.
- Sanity: confirm Closed-vs-Open frontier shows recognizable headline-leading models only (e.g. GPT-5, Claude Opus 4.5) on the leading side.
- Regression: confirm the rest of the dashboard (headline chart, calculators, gap-details table, methodology) is unchanged.
- Open `test_dashboard.js` and run if it exercises gap logic — update assertions to match corrected numbers.

## Risks

- **Stat values will move.** The bug fixes change Average Gap, Matched/Unmatched counts, and the survival estimate. This is intentional but worth noting in any release blurb.
- **D3 + Plotly on the same page** adds ~85kb gz; acceptable for a single dashboard page.
- **Color choice for Open vs Closed** uses the same blue/red palette as China/US. We rely on the toggle-pill state and legend labels to disambiguate; users switching toggles should see the legend update.
