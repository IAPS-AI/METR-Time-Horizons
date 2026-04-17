# Gap Visualization Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Plotly gap chart in `dashboard.html` with a D3 step-curve / shaded-gap visualization in the style of [ai-time-horizons.vercel.app](https://ai-time-horizons.vercel.app/), and fix two bugs in the underlying gap math (per-family frontier, missed-backward-match).

**Architecture:** Single-file edit (`dashboard.html`), with a parallel update to `test_dashboard.js` (which keeps copies of the pure functions for node-based testing). New chart is rendered by a `renderGapChart()` function using D3 v7 (added via CDN script tag). Existing stat cards stay in place; their numbers update because the calc is now correct. Toggle controls are restyled as pill groups matching the reference; two new pill groups are added (success-rate, scale).

**Tech Stack:** Vanilla JS, D3 v7 (new dep), existing Plotly + js-yaml. All in one HTML file.

**Spec:** `docs/superpowers/specs/2026-04-17-gap-visualization-redesign-design.md`

---

## File map

- **Modify:** `dashboard.html` — gap calc functions, controls markup, chart container, render function, CSS, `<head>` script.
- **Modify:** `test_dashboard.js` — corrected pure-function copies + updated assertions for new behavior.
- **Create:** none.

---

## Task 1: Fix gap calculation in `test_dashboard.js` (TDD)

**Files:**
- Modify: `test_dashboard.js:125-195` (`getFrontierModels`, `calculateHorizontalGaps`)
- Modify: `test_dashboard.js:469-526` (tests 6 & 7)

- [ ] **Step 1: Replace `getFrontierModels` with `frontierFor` in `test_dashboard.js`**

Locate the existing `getFrontierModels` (around lines 125-146). Replace with:

```javascript
function frontierFor(groupMembers) {
    const sorted = [...groupMembers].sort((a, b) => a.date - b.date);
    const out = [];
    let maxH = 0;
    for (const m of sorted) {
        if (m.horizon > maxH) {
            maxH = m.horizon;
            out.push(m);
        }
    }
    return out;
}
```

Note: `>` strict — ties do not duplicate frontier points. Function takes a flat list (no inner grouping).

- [ ] **Step 2: Replace `calculateHorizontalGaps` in `test_dashboard.js`**

Locate the existing function (around lines 148-195). Replace with:

```javascript
function calculateHorizontalGaps(data, groupKey, leadingValue) {
    const leadingData = data.filter(d => d[groupKey] === leadingValue);
    const laggingData = data.filter(d => d[groupKey] !== leadingValue);

    const leadingFrontier = frontierFor(leadingData);
    const laggingFrontier = frontierFor(laggingData);

    const gaps = [];

    for (const leading of leadingFrontier) {
        // First lagging-frontier model whose horizon reaches leading.horizon,
        // regardless of whether it released before or after leading.
        const crossing = laggingFrontier.find(l => l.horizon >= leading.horizon) || null;

        if (crossing) {
            const gapDays = (crossing.date - leading.date) / (1000 * 60 * 60 * 24);
            // Negative gaps (lagging was already ahead) collapse to 0.
            const gapMonths = Math.max(0, gapDays / DAYS_PER_MONTH);
            gaps.push({
                leadingModel: leading.name,
                leadingDate: leading.date,
                leadingHorizon: leading.horizon,
                laggingModel: crossing.name,
                laggingDate: crossing.date,
                laggingHorizon: crossing.horizon,
                gapMonths,
                matched: true,
            });
        } else {
            const gapDays = (new Date() - leading.date) / (1000 * 60 * 60 * 24);
            gaps.push({
                leadingModel: leading.name,
                leadingDate: leading.date,
                leadingHorizon: leading.horizon,
                laggingModel: null,
                laggingDate: null,
                laggingHorizon: null,
                gapMonths: gapDays / DAYS_PER_MONTH,
                matched: false,
            });
        }
    }

    return gaps;
}
```

- [ ] **Step 3: Update test #6 (frontier) for new signature and behavior**

Locate test #6 (around lines 469-490). Replace the test block with:

```javascript
console.log('\n6. frontierFor');
console.log('--------------');

const testModels = [
    { name: 'A', family: 'F1', date: new Date('2023-01-01'), horizon: 1 },
    { name: 'B', family: 'F1', date: new Date('2023-06-01'), horizon: 2 },
    { name: 'C', family: 'F1', date: new Date('2023-12-01'), horizon: 1.5 },  // not frontier (< B)
    { name: 'D', family: 'F1', date: new Date('2024-06-01'), horizon: 3 },
    { name: 'E', family: 'F2', date: new Date('2023-03-01'), horizon: 0.5 },   // below current global max → not frontier
    { name: 'F', family: 'F2', date: new Date('2023-09-01'), horizon: 1 },     // below current global max → not frontier
];

const frontier = frontierFor(testModels);
const frontierNames = frontier.map(m => m.name);

assert(frontierNames.includes('A'), 'A is frontier (first model)');
assert(frontierNames.includes('B'), 'B is frontier (improves on A)');
assert(!frontierNames.includes('C'), 'C is NOT frontier (1.5 < 2)');
assert(frontierNames.includes('D'), 'D is frontier (improves on B)');
assert(!frontierNames.includes('E'), 'E is NOT frontier (0.5 < 1)');
assert(!frontierNames.includes('F'), 'F is NOT frontier (1 < 2 at its time)');
assert(frontier.length === 3, 'Total 3 global frontier models');

// Tie handling: equal horizon does not duplicate
const tieModels = [
    { name: 'X', family: 'F1', date: new Date('2023-01-01'), horizon: 1 },
    { name: 'Y', family: 'F1', date: new Date('2023-02-01'), horizon: 1 },  // tie
];
const tieFrontier = frontierFor(tieModels);
assert(tieFrontier.length === 1, 'Tied horizon does not push duplicate frontier');
assert(tieFrontier[0].name === 'X', 'First-released wins on tie');
```

- [ ] **Step 4: Update test #7 (gaps) for new behavior**

Locate test #7 (around lines 492-526). Replace the test block with:

```javascript
console.log('\n7. calculateHorizontalGaps');
console.log('--------------------------');

const gapTestData = [
    // Closed (leading)
    { name: 'Closed1', family: 'Anthropic', date: new Date('2023-01-01'), horizon: 1, isOpen: false },
    { name: 'Closed2', family: 'Anthropic', date: new Date('2023-06-01'), horizon: 2, isOpen: false },
    { name: 'Closed3', family: 'OpenAI',    date: new Date('2024-01-01'), horizon: 4, isOpen: false },
    // Open (lagging)
    { name: 'Open1',   family: 'DeepSeek',  date: new Date('2023-03-01'), horizon: 0.5, isOpen: true },
    { name: 'Open2',   family: 'DeepSeek',  date: new Date('2023-09-01'), horizon: 1.5, isOpen: true },
    { name: 'Open3',   family: 'Alibaba',   date: new Date('2024-03-01'), horizon: 2.5, isOpen: true },
];

const gaps = calculateHorizontalGaps(gapTestData, 'isOpen', false);

// Leading frontier (global cumulative max in closed group): Closed1 (1), Closed2 (2), Closed3 (4).
// Lagging frontier (global cumulative max in open group):  Open1 (0.5), Open2 (1.5), Open3 (2.5).
assert(gaps.length === 3, 'One gap per leading frontier model');

const closed1 = gaps.find(g => g.leadingModel === 'Closed1');
assert(closed1.matched === true, 'Closed1 matched');
assert(closed1.laggingModel === 'Open2', 'Closed1 matched by Open2 (first open ≥ 1)');
assertApprox(closed1.gapMonths, 8, 1, 'Closed1 gap ≈ 8 months');

const closed2 = gaps.find(g => g.leadingModel === 'Closed2');
assert(closed2.matched === true, 'Closed2 matched');
assert(closed2.laggingModel === 'Open3', 'Closed2 matched by Open3 (first open ≥ 2)');
assertApprox(closed2.gapMonths, 9, 1, 'Closed2 gap ≈ 9 months');

const closed3 = gaps.find(g => g.leadingModel === 'Closed3');
assert(closed3.matched === false, 'Closed3 unmatched (no open ≥ 4)');
assert(closed3.laggingModel === null, 'Closed3 has no lagging model');

// Backward-match case: lagging side already ahead at leading-release time → gap = 0.
const backwardData = [
    { name: 'L1', family: 'F1', date: new Date('2024-06-01'), horizon: 1, isOpen: false },
    { name: 'O1', family: 'F2', date: new Date('2024-01-01'), horizon: 5, isOpen: true },
];
const backGaps = calculateHorizontalGaps(backwardData, 'isOpen', false);
assert(backGaps[0].matched === true, 'Backward case is matched, not unmatched');
assertApprox(backGaps[0].gapMonths, 0, 0.001, 'Backward case clamped to 0');
```

- [ ] **Step 5: Run tests, expect calc tests to pass**

Run: `node test_dashboard.js`
Expected: all assertions pass (or at least sections 6 & 7 fully pass; the rest unchanged).

If a previously passing test now fails because it depended on the old buggy behavior, the test was checking the bug, not the spec. Update it to match the spec.

- [ ] **Step 6: Commit**

```bash
git add test_dashboard.js
git commit -m "Fix gap-calc: global frontier and backward-match handling

frontierFor now operates on a flat group (no per-family interleaving),
and calculateHorizontalGaps treats lagging-already-ahead as a 0-month
matched gap rather than an open censored gap."
```

---

## Task 2: Apply the same fixes to `dashboard.html`

**Files:**
- Modify: `dashboard.html` (`getFrontierModels` and `calculateHorizontalGaps` definitions, plus their call sites)

- [ ] **Step 1: Replace `getFrontierModels` with `frontierFor` in `dashboard.html`**

Locate the function (around lines 1080-1101 — search for `function getFrontierModels`). Replace with the same body as Task 1 / Step 1:

```javascript
function frontierFor(groupMembers) {
    const sorted = [...groupMembers].sort((a, b) => a.date - b.date);
    const out = [];
    let maxH = 0;
    for (const m of sorted) {
        if (m.horizon > maxH) {
            maxH = m.horizon;
            out.push(m);
        }
    }
    return out;
}
```

- [ ] **Step 2: Replace `calculateHorizontalGaps` in `dashboard.html`**

Locate the function (around lines 1106-1153). Replace with the same body as Task 1 / Step 2.

- [ ] **Step 3: Update both call sites in `dashboard.html`**

There are two callers of the old `getFrontierModels(groupData, d => d.family)` pattern. Both should call `frontierFor(groupData)` instead — no inner grouping function.

Search for `getFrontierModels(` in `dashboard.html`. Replace each call:

Before:
```javascript
const leadingFrontier = getFrontierModels(leadingData, d => d.family);
const laggingFrontier = getFrontierModels(laggingData, d => d.family);
```

After:
```javascript
const leadingFrontier = frontierFor(leadingData);
const laggingFrontier = frontierFor(laggingData);
```

Apply this to every call site (currently in `createGapPlot` around line 1938, and inside the new `calculateHorizontalGaps` you wrote in Step 2).

- [ ] **Step 4: Quick smoke test in browser**

Open `dashboard.html` in a browser. Confirm:
- The dashboard loads without console errors.
- The Performance Gap Analysis section's stat cards show finite numbers.
- The Gap Details table renders rows.

Do not yet expect the chart to look new — that's later tasks. Just confirm no JS errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard.html
git commit -m "Apply gap-calc fixes to dashboard.html

Mirrors the corrected frontierFor and calculateHorizontalGaps from
test_dashboard.js into the inline scripts in dashboard.html."
```

---

## Task 3: Add D3 dependency and scoped CSS

**Files:**
- Modify: `dashboard.html` (`<head>` and `<style>`)

- [ ] **Step 1: Add D3 v7 script tag in `<head>`**

Find the existing Plotly/js-yaml script tags (around lines 7-8):

```html
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"></script>
```

Add immediately after:

```html
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
```

- [ ] **Step 2: Add scoped CSS for the gap section**

At the end of the existing `<style>` block (just before `</style>`, around line 299), append:

```css
/* ============ Gap section ============ */
.gap-section {
    --gap-leading: #457b9d;
    --gap-lagging: #e63946;
    --gap-fill: rgba(230, 57, 70, 0.07);
    --gap-stroke: rgba(230, 57, 70, 0.18);
    --gap-grid: #f0f0f0;
    --gap-axis: #777;
    --gap-panel: #f3f3f3;
    --gap-panel-hover: #eaeaea;
    --gap-text: #1a1a1a;
    --gap-muted: #666;
    --gap-tooltip-bg: #1a1a1a;
    --gap-tooltip-ink: #fff;
    --gap-border: #e8e8e8;
}

.gap-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-end;
    margin-bottom: 12px;
}

.gap-toggle-field {
    border: 0;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.gap-toggle-field legend {
    font-size: 11px;
    font-weight: 600;
    color: var(--gap-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 0;
}

.gap-toggle-group {
    display: inline-flex;
    background: var(--gap-panel);
    border-radius: 7px;
    padding: 3px;
}

.gap-toggle-group label {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    color: var(--gap-muted);
    padding: 5px 13px;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.15s, color 0.15s;
    user-select: none;
}

.gap-toggle-group input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}

.gap-toggle-group input:checked + label {
    background: var(--gap-text);
    color: #fff;
}

.gap-legend {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 13px;
    color: var(--gap-text);
    margin: 8px 0 6px;
}

.gap-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.gap-legend-line {
    width: 22px;
    height: 0;
    border-top-width: 2.5px;
    border-top-style: solid;
}

.gap-legend-line.lagging {
    border-top-style: dashed;
}

.gap-legend-swatch {
    width: 14px;
    height: 14px;
    background: var(--gap-fill);
    border: 1px solid var(--gap-stroke);
    border-radius: 2px;
}

.gap-chart-container {
    position: relative;
    width: 100%;
    height: 420px;
    margin-bottom: 8px;
    overflow: hidden;
}

.gap-chart-container svg {
    display: block;
    width: 100%;
    height: 100%;
}

.gap-chart-container .domain {
    display: none;
}

.gap-chart-container .grid line {
    stroke: var(--gap-grid);
    shape-rendering: crispEdges;
}

.gap-chart-container .tick text {
    fill: var(--gap-axis);
    font-size: 11px;
}

.gap-chart-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--gap-muted);
    font-size: 13px;
}

.gap-tooltip {
    position: fixed;
    background: var(--gap-tooltip-bg);
    color: var(--gap-tooltip-ink);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 9999;
    max-width: 240px;
    line-height: 1.45;
}

.gap-tooltip .tt-title {
    font-weight: 700;
    margin-bottom: 2px;
}
```

- [ ] **Step 3: Commit**

```bash
git add dashboard.html
git commit -m "Add D3 v7 and scoped CSS for new gap section"
```

---

## Task 4: Replace gap-section controls with toggle pills

**Files:**
- Modify: `dashboard.html` — the Performance Gap Analysis controls (lines ~482-497) and the section wrapper.

- [ ] **Step 1: Add `gap-section` class to the section wrapper**

Locate the section: search for `<h2>🔍 Performance Gap Analysis</h2>`. The enclosing `<div class="calculator-section">` (around line 476) — add the `gap-section` class so the CSS variables apply:

Before:
```html
<div class="calculator-section">
    <h2>🔍 Performance Gap Analysis</h2>
```

After:
```html
<div class="calculator-section gap-section">
    <h2>🔍 Performance Gap Analysis</h2>
```

- [ ] **Step 2: Replace the controls block**

Locate the existing controls (around lines 482-497, the block that starts `<div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">` and ends at the matching `</div>`). Replace the entire block with:

```html
<div class="gap-controls">
    <fieldset class="gap-toggle-field">
        <legend>Comparison</legend>
        <div class="gap-toggle-group">
            <input type="radio" name="gap-framing" id="gap-framing-oc" value="open_closed" checked>
            <label for="gap-framing-oc">Open vs Closed</label>
            <input type="radio" name="gap-framing" id="gap-framing-cu" value="china_us">
            <label for="gap-framing-cu">China vs US</label>
        </div>
    </fieldset>

    <fieldset class="gap-toggle-field">
        <legend>Success rate</legend>
        <div class="gap-toggle-group">
            <input type="radio" name="gap-success" id="gap-success-p80" value="p80" checked>
            <label for="gap-success-p80">80%</label>
            <input type="radio" name="gap-success" id="gap-success-p50" value="p50">
            <label for="gap-success-p50">50%</label>
        </div>
    </fieldset>

    <fieldset class="gap-toggle-field">
        <legend>Scale</legend>
        <div class="gap-toggle-group">
            <input type="radio" name="gap-scale" id="gap-scale-linear" value="linear">
            <label for="gap-scale-linear">Linear</label>
            <input type="radio" name="gap-scale" id="gap-scale-log" value="log" checked>
            <label for="gap-scale-log">Log</label>
        </div>
    </fieldset>

    <fieldset class="gap-toggle-field">
        <legend>Gap metric</legend>
        <div class="gap-toggle-group">
            <input type="radio" name="gap-metric" id="gap-metric-avg" value="average" checked>
            <label for="gap-metric-avg">Average</label>
            <input type="radio" name="gap-metric" id="gap-metric-cur" value="current">
            <label for="gap-metric-cur">Current (Est.)</label>
        </div>
    </fieldset>
</div>
```

- [ ] **Step 3: Wire the new pill controls**

Search for the existing event-binding code: `document.querySelectorAll('[data-gap-tab]')` and `document.querySelectorAll('[data-gap-metric]')` (around lines 2413-2435). Replace **both** blocks with:

```javascript
// Comparison framing
document.querySelectorAll('input[name="gap-framing"]').forEach(input => {
    input.addEventListener('change', (e) => {
        if (!e.target.checked) return;
        currentGapFraming = e.target.value;
        if (currentData) updateGapAnalysis(currentData);
    });
});

// Success rate (per-section override of the global horizonType for the gap chart only)
document.querySelectorAll('input[name="gap-success"]').forEach(input => {
    input.addEventListener('change', (e) => {
        if (!e.target.checked) return;
        currentGapSuccess = e.target.value;
        if (rawData) updateGapAnalysis(parseYAMLData(rawData, currentGapSuccess));
    });
});

// Scale
document.querySelectorAll('input[name="gap-scale"]').forEach(input => {
    input.addEventListener('change', (e) => {
        if (!e.target.checked) return;
        currentGapScale = e.target.value;
        if (currentData) updateGapAnalysis(getGapData());
    });
});

// Gap metric (average vs current)
document.querySelectorAll('input[name="gap-metric"]').forEach(input => {
    input.addEventListener('change', (e) => {
        if (!e.target.checked) return;
        currentGapMetric = e.target.value;
        if (currentData) updateGapAnalysis(getGapData());
    });
});
```

Also add the new state variables and helper near the existing `let currentGapFraming = 'open_closed';` declaration (around line 1922):

```javascript
let currentGapFraming = 'open_closed';
let currentGapMetric = 'average';
let currentGapSuccess = 'p80';   // NEW: overrides global horizonType for the gap chart
let currentGapScale = 'log';     // NEW: 'linear' | 'log'

function getGapData() {
    // Re-parse with the gap-section's success-rate override.
    return parseYAMLData(rawData, currentGapSuccess);
}
```

And update the `updateGapAnalysis` invocation pattern. Replace its body (around line 2171) with:

```javascript
function updateGapAnalysis(data) {
    renderGapChart(data, currentGapFraming, currentGapScale);
    updateGapStats(data, currentGapFraming, currentGapMetric);
}
```

(`renderGapChart` is implemented in Task 5; this wiring is the contract.)

Finally, find the call from `refresh()` that triggers the gap section (search for `updateGapAnalysis(allData)`). It currently passes `allData` which was parsed with the global `horizonType`. Change it to use the gap-section's success rate:

```javascript
updateGapAnalysis(getGapData());
```

- [ ] **Step 4: Verify in browser**

Reload `dashboard.html`. Toggle each pill — the section should not throw errors. The Plotly chart will still render (we replace it next task), but the toggles should re-trigger it. The `[data-gap-tab]` / `[data-gap-metric]` `<button>` elements no longer exist; all wiring goes through the new `<input>` elements.

- [ ] **Step 5: Commit**

```bash
git add dashboard.html
git commit -m "Restyle gap-section controls as toggle pills

Adds two new pill groups: Success rate (p50/p80) and Scale (linear/log)
that operate independently of the dashboard's global controls."
```

---

## Task 5: Build the D3 chart

**Files:**
- Modify: `dashboard.html` — replace `#gapPlot` div + add `renderGapChart` + remove old `createGapPlot`.

- [ ] **Step 1: Replace the chart container markup**

Locate `<div id="gapPlot" style="margin-top: 1rem; min-height: 400px;"></div>` (around line 538). Replace with:

```html
<div class="gap-legend" id="gapLegend"></div>
<div class="gap-chart-container" id="gapChart"></div>
<div class="gap-tooltip" id="gapTooltip" role="status" aria-live="polite"></div>
```

- [ ] **Step 2: Add `renderGapChart` function**

Locate the existing `createGapPlot` function (around lines 1925-2040). **Delete it entirely** and replace with the following implementation. Place this where `createGapPlot` used to be, just above `updateGapStats`:

```javascript
function renderGapChart(data, framing, scaleMode) {
    const isOpenClosed = framing === 'open_closed';
    const groupKey = isOpenClosed ? 'isOpen' : 'isChina';
    const leadingValue = false; // Closed / US lead

    const leadingLabel = isOpenClosed ? 'Closed' : 'US';
    const laggingLabel = isOpenClosed ? 'Open' : 'China';

    const leadingData = data.filter(d => d[groupKey] === leadingValue);
    const laggingData = data.filter(d => d[groupKey] !== leadingValue);

    const leadingFrontier = frontierFor(leadingData);
    const laggingFrontier = frontierFor(laggingData);

    const container = document.getElementById('gapChart');
    container.innerHTML = '';

    // Legend
    const legend = document.getElementById('gapLegend');
    legend.innerHTML = `
        <span class="gap-legend-item">
            <span class="gap-legend-line" style="border-color: var(--gap-leading);"></span>
            ${leadingLabel} frontier
        </span>
        <span class="gap-legend-item">
            <span class="gap-legend-line lagging" style="border-color: var(--gap-lagging);"></span>
            ${laggingLabel} frontier
        </span>
        <span class="gap-legend-item">
            <span class="gap-legend-swatch"></span>
            Capability gap
        </span>
    `;

    if (leadingFrontier.length < 2 || laggingFrontier.length < 1) {
        container.innerHTML = '<div class="gap-chart-empty">Not enough frontier data for this comparison.</div>';
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { top: 16, right: 24, bottom: 36, left: 60 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // X scale: time
    const allDates = [...leadingFrontier, ...laggingFrontier].map(d => d.date);
    const today = new Date();
    const xMin = d3.min(allDates);
    const xMax = today;
    const x = d3.scaleTime().domain([xMin, xMax]).range([0, innerW]).nice();

    // Y scale: horizon (hours)
    const allHorizons = [...leadingFrontier, ...laggingFrontier].map(d => d.horizon);
    const yMaxData = d3.max(allHorizons);
    const yMinData = d3.min(allHorizons);
    let y;
    if (scaleMode === 'log') {
        y = d3.scaleLog()
            .domain([Math.max(yMinData * 0.5, 1e-3), yMaxData * 1.2])
            .range([innerH, 0])
            .clamp(true);
    } else {
        y = d3.scaleLinear()
            .domain([0, yMaxData * 1.1])
            .range([innerH, 0])
            .nice();
    }

    // Grid
    g.append('g')
        .attr('class', 'grid')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(6).tickSize(-innerH).tickFormat(''));
    g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(y).ticks(6).tickSize(-innerW).tickFormat(''));

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat('%b %Y')));

    const yTickFormat = (v) => formatHours(v);
    g.append('g').call(d3.axisLeft(y).ticks(6).tickFormat(yTickFormat));

    // Build step series extended to today at last horizon
    function extendToToday(frontier) {
        if (frontier.length === 0) return [];
        const last = frontier[frontier.length - 1];
        return [...frontier, { ...last, date: today, _isExtension: true }];
    }
    const leadingSeries = extendToToday(leadingFrontier);
    const laggingSeries = extendToToday(laggingFrontier);

    // Step line generator
    const line = d3.line()
        .x(d => x(d.date))
        .y(d => y(d.horizon))
        .curve(d3.curveStepAfter);

    // Shaded gap area: build a combined polygon at every event date.
    // For each unique event date, determine the cumulative-max horizon for
    // each side (using step-after semantics: value at time t is the horizon
    // of the most recent frontier point with date <= t).
    function horizonAt(series, t) {
        let h = 0;
        for (const p of series) {
            if (p.date <= t) h = p.horizon; else break;
        }
        return h;
    }

    const eventDates = Array.from(new Set([
        ...leadingSeries.map(p => p.date.getTime()),
        ...laggingSeries.map(p => p.date.getTime()),
    ])).sort((a, b) => a - b).map(t => new Date(t));

    const gapPoints = eventDates.map(t => {
        const lead = horizonAt(leadingSeries, t);
        const lag = horizonAt(laggingSeries, t);
        return { date: t, lead, lag };
    }).filter(p => p.lead > 0 && p.lag > 0);

    if (gapPoints.length >= 2) {
        const area = d3.area()
            .x(p => x(p.date))
            .y0(p => y(p.lag))
            .y1(p => y(p.lead))
            .curve(d3.curveStepAfter);

        g.append('path')
            .attr('d', area(gapPoints))
            .attr('fill', 'var(--gap-fill)')
            .attr('stroke', 'var(--gap-stroke)')
            .attr('stroke-width', 0.5);
    }

    // Lines
    g.append('path')
        .attr('d', line(leadingSeries))
        .attr('fill', 'none')
        .attr('stroke', 'var(--gap-leading)')
        .attr('stroke-width', 2.5);

    g.append('path')
        .attr('d', line(laggingSeries))
        .attr('fill', 'none')
        .attr('stroke', 'var(--gap-lagging)')
        .attr('stroke-width', 2.5)
        .attr('stroke-dasharray', '6 4');

    // Dots (frontier-advancing models only, exclude the synthetic extension)
    function drawDots(frontier, color, hollow) {
        g.selectAll(null)
            .data(frontier)
            .enter()
            .append('circle')
            .attr('cx', d => x(d.date))
            .attr('cy', d => y(d.horizon))
            .attr('r', 5)
            .attr('fill', hollow ? '#ffffff' : color)
            .attr('stroke', color)
            .attr('stroke-width', 2)
            .attr('class', 'dot')
            .style('cursor', 'pointer')
            .on('mouseenter', function (event, d) {
                showGapTooltip(event, d, hollow ? laggingLabel : leadingLabel);
            })
            .on('mouseleave', hideGapTooltip);
    }
    drawDots(leadingFrontier, '#457b9d', false);
    drawDots(laggingFrontier, '#e63946', true);
}

function showGapTooltip(event, d, groupLabel) {
    const tt = document.getElementById('gapTooltip');
    tt.innerHTML = `
        <div class="tt-title">${d.name}</div>
        <div>${groupLabel} · ${formatDate(d.date)}</div>
        <div>Horizon: ${formatHours(d.horizon)}</div>
    `;
    tt.style.opacity = 1;
    tt.style.left = (event.clientX + 12) + 'px';
    tt.style.top = (event.clientY + 12) + 'px';
}

function hideGapTooltip() {
    document.getElementById('gapTooltip').style.opacity = 0;
}
```

- [ ] **Step 3: Add a debounced resize handler**

Find the bottom of the script (just before the closing `</script>`). Add:

```javascript
// Re-render gap chart on resize (debounced)
let _gapResizeTimer = null;
window.addEventListener('resize', () => {
    clearTimeout(_gapResizeTimer);
    _gapResizeTimer = setTimeout(() => {
        if (rawData) renderGapChart(getGapData(), currentGapFraming, currentGapScale);
    }, 100);
});
```

- [ ] **Step 4: Verify in browser**

Open `dashboard.html`. Confirm:
- The gap section now shows a D3 SVG chart with two step lines and a shaded fill between them.
- Toggling Comparison swaps Open/Closed ↔ China/US datasets.
- Toggling Success rate (p80/p50) swaps the data underneath.
- Toggling Scale (linear/log) changes the Y axis.
- Toggling Gap metric swaps the first stat card (avg vs current-est).
- Hovering a dot shows the tooltip.
- Resizing the window re-renders.
- The headline trend chart, calculators, milestone calc, and methodology section all still work normally.

- [ ] **Step 5: Commit**

```bash
git add dashboard.html
git commit -m "Replace Plotly gap chart with D3 step-curve + shaded gap

New renderGapChart draws cumulative-max frontier step lines for the
leading and lagging groups, with a shaded fill between them. Includes
hover tooltip on frontier-advancing dots and a debounced resize handler."
```

---

## Task 6: Manual verification matrix and final cleanup

**Files:**
- Modify: `dashboard.html` (only if regressions found)

- [ ] **Step 1: Run the test suite**

```bash
node test_dashboard.js
```

Expected: all tests pass. If any newly fail because of the calc changes from Task 1, fix the test (it was checking the bug) — not the implementation.

- [ ] **Step 2: Walk the full toggle matrix in the browser**

Open `dashboard.html`. For each combination, confirm the chart and stat cards both update consistently:

| Comparison | Success rate | Scale | Metric | Sanity check |
|---|---|---|---|---|
| Open vs Closed | 80% | Log | Average | Closed leading line includes GPT-5 / Claude Opus 4.5 era |
| Open vs Closed | 80% | Linear | Average | Same data, linear y |
| Open vs Closed | 50% | Log | Average | Higher absolute horizons |
| Open vs Closed | 80% | Log | Current (Est.) | First stat card shows estimated current gap, explainer visible |
| China vs US | 80% | Log | Average | US leading; DeepSeek/Qwen/Kimi on lagging line |
| China vs US | 50% | Log | Current (Est.) | Stat cards reflect China-vs-US framing |

Verify: gap-details table headers update with the comparison; classification table is unchanged content.

- [ ] **Step 3: Confirm unrelated sections are untouched**

Scroll through the dashboard and confirm:
- Headline trend chart renders and updates from its own controls.
- Custom Growth Scenarios calculator works.
- Implied Doubling Time Calculator (both tabs) works.
- Time Horizon Milestone Calculator works.
- Methodology & Limitations section is unchanged.
- Raw Data table at the bottom still renders.

- [ ] **Step 4: Commit any final fixes**

If the verification matrix surfaced bugs, fix them and commit:

```bash
git add dashboard.html
git commit -m "Fix <specific issue> found during gap-redesign verification"
```

If no fixes were needed, skip this step.

- [ ] **Step 5: Final summary**

Report to the user:
- Tasks completed.
- Manual verification matrix outcomes.
- Any deferred follow-ups (e.g. dark-mode rework was explicitly out of scope; could be added separately).

---

## Notes for the executor

- **Single file:** Almost all changes land in `dashboard.html`. Keep edits surgical — do not refactor unrelated sections.
- **Two copies of the calc:** `dashboard.html` has its own inline copy of the gap functions; `test_dashboard.js` keeps copies for node-based testing. Both must stay in sync. If you change one, change the other.
- **Plotly is still used** elsewhere on the page (headline chart). Don't remove the Plotly script tag.
- **No new files.** The spec, this plan, and the changes are it.
- **Format-helper reuse:** The new chart uses the existing `formatHours` and `formatDate` helpers. Don't duplicate them.
