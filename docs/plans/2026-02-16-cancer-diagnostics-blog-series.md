# Cancer Diagnostics Blog Series Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a 3-part data journalism blog series on cancer detection assays using OpenOnco data, with interactive D3.js v7 charts and mermaid diagrams.

**Architecture:** Each post is a Jekyll markdown file with inline `<script>` blocks for D3 v7 charts. Chart data is pre-extracted from OpenOnco_AllTests.json and embedded inline. All charts are theme-aware (light/dark) using the blog's existing `themechange` event system. Mermaid diagrams use the blog's existing hand-drawn xkcd style.

**Tech Stack:** Jekyll 4.3.3, D3.js v7 (CDN), Mermaid 11 (already in head.html), inline `<style>`/`<script>` per post.

---

## Task 0: Extract chart data from OpenOnco JSON

**Files:**
- Read: `_drafts/openconco/OpenOnco_AllTests.json`
- Read: `_drafts/openconco/investigation.md`

This is a preparatory data extraction step. All subsequent tasks reference the extracted data structures below. Extract these datasets by reading the JSON and investigation.md, then hardcode them as inline JS objects in each post's `<script>` block.

### Dataset A: Market Timeline (Part 1)
From investigation.md "Trends by Year" table — tests entering market by year, grouped by category:
```js
const timelineData = [
  { year: 2014, MRD: 0, ECD: 1, CGP: 1, HCT: 0 },
  { year: 2016, MRD: 0, ECD: 1, CGP: 0, HCT: 0 },
  // ... through 2026
];
```

### Dataset B: Category Overview (Part 1)
From meta.testCount per category, plus regulatory breakdown from investigation.md:
```js
const categoryData = [
  { category: "MRD", count: 44, fdaApproved: 1, cliaLdt: 15, other: 28 },
  { category: "ECD", count: 31, fdaApproved: 5, cliaLdt: 5, other: 21 },
  { category: "CGP", count: 46, fdaApproved: 18, cliaLdt: 5, other: 23 },
  { category: "HCT", count: 34, fdaApproved: 1, cliaLdt: 32, other: 1 },
];
```

### Dataset C: MRD ROC (Part 2)
From investigation.md MRD performance table + JSON fields `sensitivity`, `specificity`, `approach`:
```js
const mrdRocData = [
  { name: "NeXT Personal", vendor: "Personalis", sensitivity: 100, specificity: 99.9, approach: "tumor-informed", fdaCleared: false, cohortSize: 493 },
  { name: "Signatera", vendor: "Natera", sensitivity: 94, specificity: 98, approach: "tumor-informed", fdaCleared: false, cohortSize: 300000 },
  { name: "clonoSEQ", vendor: "Adaptive Biotech", sensitivity: 95, specificity: 99, approach: "tumor-informed", fdaCleared: true, cohortSize: null },
  { name: "Reveal MRD", vendor: "Guardant Health", sensitivity: 81, specificity: 98, approach: "tumor-naive", fdaCleared: false, cohortSize: null },
  // ... all tests with both sensitivity and specificity data
];
```

### Dataset D: MRD LoD (Part 2)
From investigation.md LoD table + JSON `lod` field. Parse ppm values:
```js
const mrdLodData = [
  { name: "Foresight CLARITY", vendor: "Natera", lod: 0.7, approach: "tumor-informed" },
  { name: "Signatera Genome", vendor: "Natera", lod: 1.0, approach: "tumor-informed" },
  // ... all tests with LoD data, sorted ascending
];
```

### Dataset E: MRD Regulatory (Part 2)
From investigation.md regulatory table:
```js
const mrdRegData = [
  { status: "CLIA LDT", count: 15 },
  { status: "FDA Cleared/Authorized", count: 12 },
  { status: "FDA Approved", count: 7 },
  { status: "Research Use Only", count: 3 },
  { status: "Breakthrough Designation", count: 2 },
  { status: "FDA De Novo", count: 1 },
  { status: "FDA 510(k)", count: 1 },
  { status: "CE-IVD/IVDR", count: 1 },
  { status: "Discontinued", count: 1 },
];
```

### Dataset F: ECD ROC (Part 3)
From investigation.md ECD performance table:
```js
const ecdRocData = [
  { name: "Cologuard Plus", vendor: "Exact Sciences", sensitivity: 93.9, specificity: 91, indication: "CRC-stool", fdaApproved: true },
  { name: "Shield", vendor: "Guardant Health", sensitivity: 83.1, specificity: 89.6, indication: "CRC-blood", fdaApproved: true },
  { name: "Galleri", vendor: "GRAIL", sensitivity: 51.5, specificity: 99.5, indication: "MCED", fdaApproved: false },
  // ... all ECD tests with both sensitivity and specificity data
];
```

### Dataset G: ECD Stage Sensitivity (Part 3)
From investigation.md stage-specific table:
```js
const ecdStageData = [
  { name: "ColoSense", stages: [{ stage: "I", sensitivity: 100 }] },
  { name: "Cologuard Plus", stages: [{ stage: "I", sensitivity: 87 }, { stage: "IV", sensitivity: 100 }] },
  { name: "Galleri", stages: [{ stage: "I", sensitivity: 16.8 }, { stage: "IV", sensitivity: 90.1 }] },
  // ... all tests with stage-specific data
];
```

### Dataset H: ECD Geography (Part 3)
From investigation.md US vs international table:
```js
const ecdGeoData = [
  { name: "Galleri", vendor: "GRAIL", regions: ["US", "UK"] },
  { name: "SPOT-MAS", vendor: "Gene Solutions", regions: ["Singapore", "Vietnam", "Malaysia", "Thailand", "Indonesia", "Philippines"] },
  // ... all ECD tests with region data
];
```

**No step-by-step here** — this is reference data. Each post task below specifies which datasets to embed inline.

---

## Task 1: D3 utility module

**Files:**
- Create: `assets/js/cancer-charts.js`

This shared module provides theme-aware color palettes and tooltip utilities used by all 3 posts.

**Step 1: Write the utility module**

```js
// assets/js/cancer-charts.js
// Shared D3 chart utilities for the cancer diagnostics blog series

(function(window) {
  'use strict';

  const COLORS = {
    light: {
      bg: '#fafaf9',
      text: '#1a1a2e',
      muted: '#6b6b8a',
      grid: '#e5e5e5',
      blue: '#2563eb',
      green: '#059669',
      pink: '#db2777',
      purple: '#7c3aed',
      yellow: '#d97706',
      red: '#dc2626',
      // Category colors
      MRD: '#2563eb',
      ECD: '#059669',
      CGP: '#7c3aed',
      HCT: '#d97706',
      // Approach colors
      'tumor-informed': '#2563eb',
      'tumor-naive': '#dc2626',
      // Indication colors
      'CRC-stool': '#2563eb',
      'CRC-blood': '#059669',
      'MCED': '#dc2626',
      'Liver': '#d97706',
      'Lung': '#7c3aed',
      'Other': '#6b6b8a',
    },
    dark: {
      bg: '#18181b',
      text: '#e4e4e7',
      muted: '#71717a',
      grid: '#3f3f46',
      blue: '#60a5fa',
      green: '#34d399',
      pink: '#f472b6',
      purple: '#a78bfa',
      yellow: '#fbbf24',
      red: '#f87171',
      MRD: '#60a5fa',
      ECD: '#34d399',
      CGP: '#a78bfa',
      HCT: '#fbbf24',
      'tumor-informed': '#60a5fa',
      'tumor-naive': '#f87171',
      'CRC-stool': '#60a5fa',
      'CRC-blood': '#34d399',
      'MCED': '#f87171',
      'Liver': '#fbbf24',
      'Lung': '#a78bfa',
      'Other': '#71717a',
    }
  };

  function getTheme() {
    return document.documentElement.getAttribute('data-theme') || 'light';
  }

  function getColors() {
    return COLORS[getTheme()];
  }

  // Tooltip div (shared, lazily created)
  let tooltipEl = null;
  function tooltip() {
    if (!tooltipEl) {
      tooltipEl = d3.select('body').append('div')
        .attr('class', 'cancer-chart-tooltip')
        .style('position', 'absolute')
        .style('pointer-events', 'none')
        .style('opacity', 0)
        .style('padding', '8px 12px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('line-height', '1.4')
        .style('max-width', '280px')
        .style('z-index', '1000');
    }
    // Update colors on each access
    const c = getColors();
    tooltipEl
      .style('background', c.bg)
      .style('color', c.text)
      .style('border', `1px solid ${c.muted}`);
    return tooltipEl;
  }

  function showTooltip(event, html) {
    const tt = tooltip();
    tt.html(html)
      .style('opacity', 1)
      .style('left', (event.pageX + 12) + 'px')
      .style('top', (event.pageY - 12) + 'px');
  }

  function moveTooltip(event) {
    const tt = tooltip();
    tt.style('left', (event.pageX + 12) + 'px')
      .style('top', (event.pageY - 12) + 'px');
  }

  function hideTooltip() {
    const tt = tooltip();
    tt.style('opacity', 0);
  }

  // Gold star path for FDA-approved markers
  const STAR_PATH = 'M0,-8L2.3,-2.5L8.5,-2.5L3.5,1.5L5.3,7.5L0,4L-5.3,7.5L-3.5,1.5L-8.5,-2.5L-2.3,-2.5Z';

  window.CancerCharts = {
    COLORS, getTheme, getColors, tooltip, showTooltip, moveTooltip, hideTooltip, STAR_PATH
  };

})(window);
```

**Step 2: Verify the file was created correctly**

Run: `wc -l /path/to/assets/js/cancer-charts.js`
Expected: ~90 lines

**Step 3: Commit**

```bash
git add assets/js/cancer-charts.js
git commit -m "Add shared D3 utility module for cancer diagnostics series"
```

---

## Task 2: Part 1 — "The Four Pillars of Cancer Diagnostics"

**Files:**
- Create: `_drafts/2026-02-16-cancer-testing-landscape.md`

**Step 1: Write the post markdown**

Front matter:
```yaml
---
layout: post
title: "Cancer Testing in 2026: The Four Pillars of Molecular Oncology"
date: 2026-02-16 10:00:00 -0700
tags: biotech data-analysis
author: bolu-atx
categories: biotech
---
```

Content outline (write prose for each section):

1. **Hook paragraph** — 155 tests, 75 vendors, 6743 data points. The molecular diagnostics industry mapped.
2. **The four pillars** — One paragraph each for HCT, ECD, CGP, MRD. Concise, no jargon overload. Define each with one sentence + what question it answers.
3. **Mermaid: Patient journey** — Flowchart showing: Healthy Person → HCT (Am I at risk?) → ECD (Do I have cancer?) → CGP (What mutations drive it?) → MRD (Is it gone? Is it back?)
4. **Key concepts sidebar** — Short definitions: cfDNA, methylation, sensitivity/specificity, CHIP, tumor-informed vs tumor-agnostic, LDT vs FDA PMA vs 510(k)
5. **The data** — Introduce OpenOnco. Link to openonco.org. Explain 62% fill rate, 155 tests, 75 vendors.
6. **D3: Market landscape treemap** — Interactive treemap of 155 tests grouped by category, colored by regulatory status. Hover for test name + vendor.
7. **D3: Timeline stacked area** — Tests entering market by year, stacked by category. Annotate: "MRD explosion 2025-26", "ECD FDA breakthroughs 2024", "HCT: no date data".
8. **What's next** — Preview Parts 2 (MRD) and 3 (ECD). CGP and HCT are the mature, commoditized categories; the action is in MRD and ECD.

Use `<!--more-->` after the hook paragraph.

**Step 2: Write the D3 chart code inline**

Include at bottom of post markdown:

```html
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="/assets/js/cancer-charts.js"></script>
```

**Chart 1: Category treemap** (`<div id="treemap"></div>`)
- Data: Dataset B (categoryData) + individual test names from Dataset A context
- Layout: `d3.treemap()` with category groups, tiles for each test
- Color: category color from CancerCharts.getColors()
- Hover: test name, vendor, FDA status
- Theme-aware: listen to `themechange`, re-render with updated colors

**Chart 2: Timeline stacked area** (`<div id="timeline"></div>`)
- Data: Dataset A (timelineData)
- Layout: `d3.stack()` with categories, `d3.area()` for each
- X axis: years 2014-2026, Y axis: test count
- Annotations: text callouts at key years
- Hover: year tooltip with breakdown
- Theme-aware

**Step 3: Test locally**

Run: `bundle exec jekyll serve --drafts --future`
Visit: `localhost:4000` and verify:
- Post renders with correct front matter
- Mermaid patient journey diagram renders in xkcd style
- Both D3 charts render, are interactive (hover tooltips work)
- Light/dark theme toggle updates chart colors
- Charts are responsive on mobile viewport

**Step 4: Commit**

```bash
git add _drafts/2026-02-16-cancer-testing-landscape.md
git commit -m "Add Part 1: Four Pillars of Cancer Diagnostics"
```

---

## Task 3: Part 2 — "MRD: Hunting Invisible Cancer"

**Files:**
- Create: `_drafts/2026-02-17-mrd-hunting-invisible-cancer.md`

**Step 1: Write the post markdown**

Front matter:
```yaml
---
layout: post
title: "Cancer Testing in 2026: MRD — Hunting Invisible Cancer"
date: 2026-02-17 10:00:00 -0700
tags: biotech data-analysis
author: bolu-atx
categories: biotech
---
```

Content outline:

1. **Hook** — You've been declared cancer-free. CT scans are clean. But somewhere, a few thousand cells survived. 44 tests exist to find them — only 1 has FDA clearance.
2. **The problem** — CT/MRI can't see below several mm. Patients relapse despite "clean" scans. ctDNA as molecular surveillance.
3. **Tumor-informed vs tumor-agnostic** — The two philosophies. Bespoke assay built from your tumor's genome vs. pre-designed panel + methylation.
4. **Mermaid: Tumor-informed workflow** — Tissue sample → WES → custom panel design (4-6 weeks) → serial blood draws → ctDNA result
5. **Mermaid: Tumor-agnostic workflow** — Blood draw → fixed panel + 20k methylation signals → result (7-10 days)
6. **D3: ROC scatter** — Sensitivity vs specificity for all MRD tests with data. The visual story: tumor-informed clusters upper-left, tumor-agnostic trades sensitivity for speed.
7. **D3: LoD lollipop chart** — Log-scale limit of detection. 3 orders of magnitude spread. Natera's Foresight at 0.7 ppm vs Caris Assure at 1000 ppm.
8. **The leaders** — Signatera (largest evidence base, PMA submitted), NeXT Personal (best raw sensitivity, small validation), clonoSEQ (only FDA-cleared, 2018 vintage).
9. **D3: Regulatory status bar chart** — 15 CLIA LDT, 12 FDA cleared, 7 FDA approved. The gap.
10. **The business side** — Patent wars (Natera v NeoGenomics injunction), coverage landscape (Medicare coverage for Signatera, BCBS/Humana exclusions), pricing ($349 Signatera self-pay).
11. **Callout** — Link back to Part 1, forward to Part 3.

Use `<!--more-->` after the hook.

**Step 2: Write D3 chart code inline**

**Chart 1: ROC scatter** (`<div id="mrd-roc"></div>`)
- Data: Dataset C
- X axis: 100% - specificity (false positive rate), Y axis: sensitivity
- Points: circles colored by approach (tumor-informed blue, tumor-naive red)
- FDA-cleared: gold star marker (clonoSEQ)
- Hover: test name, vendor, sensitivity, specificity, cohort size
- Diagonal reference line (random classifier)
- Theme-aware

**Chart 2: LoD lollipop** (`<div id="mrd-lod"></div>`)
- Data: Dataset D
- Y axis: test names (sorted by LoD ascending), X axis: log10 LoD in ppm
- Lollipop: line + circle, colored by approach
- Hover: vendor, exact LoD, approach
- Annotate the 3-order-of-magnitude spread
- Theme-aware

**Chart 3: Regulatory bar** (`<div id="mrd-regulatory"></div>`)
- Data: Dataset E
- Horizontal bar chart, sorted by count descending
- Color: green for FDA approved, blue for FDA cleared, gray for CLIA LDT, yellow for other
- Hover: count
- Theme-aware

**Step 3: Test locally**

Run: `bundle exec jekyll serve --drafts --future`
Verify all 3 charts render, tooltips work, theme toggle works, mermaid diagrams render.

**Step 4: Commit**

```bash
git add _drafts/2026-02-17-mrd-hunting-invisible-cancer.md
git commit -m "Add Part 2: MRD deep dive with D3 charts"
```

---

## Task 4: Part 3 — "Early Cancer Detection: The Screening Wars"

**Files:**
- Create: `_drafts/2026-02-18-early-cancer-detection-screening-wars.md`

**Step 1: Write the post markdown**

Front matter:
```yaml
---
layout: post
title: "Cancer Testing in 2026: The Screening Wars"
date: 2026-02-18 10:00:00 -0700
tags: biotech data-analysis
author: bolu-atx
categories: biotech
---
```

Content outline:

1. **Hook** — 70% of cancer deaths have no screening guideline. A blood test that catches 50 cancers at once sounds like science fiction. It's not — but the data tells a complicated story.
2. **The screening gap** — Brief context on what cancers we screen for today (CRC, breast, lung, cervical) vs what kills people (pancreatic, liver, ovarian — no screening).
3. **CRC screening: the most mature battleground** — Stool vs blood. Compliance vs sensitivity tradeoff.
4. **Mermaid: CRC screening decision tree** — Patient → Colonoscopy (gold standard, low compliance ~67%) / Stool test (Cologuard Plus: 94% sens, better compliance) / Blood test (Shield: 83% sens, highest compliance)
5. **D3: ECD ROC scatter** — Sensitivity vs specificity colored by indication. The cluster pattern: stool tests upper-left, blood tests center, MCED far-right (high specificity, lower sensitivity).
6. **The Stage I problem** — The hardest and most valuable detection target.
7. **D3: Stage-sensitivity slope chart** — Lines from Stage I to Stage IV. Every test improves dramatically. Galleri's 17% → 90% cliff is the visual headline. ColoSense at 100% Stage I is the standout.
8. **MCED: Galleri and Cancerguard** — The promise (50+ cancer types) and the reality (17% Stage I). Galleri's NHS trial, PMA filed Jan 2026. Cancerguard targeting the deadliest cancers specifically.
9. **D3: Geographic availability** — Grouped horizontal bar or dot plot. ECD has the most international-only tests (10). Chinese NMPA tests, SPOT-MAS across SE Asia. US-centric HCT vs global CGP.
10. **What's coming** — Freenome CRC blood test, Shield MCD multi-cancer expansion, regulatory pipeline.
11. **Series wrap-up** — The four pillars from Part 1 are converging. Link back to Parts 1 and 2.

Use `<!--more-->` after the hook.

**Step 2: Write D3 chart code inline**

**Chart 1: ECD ROC scatter** (`<div id="ecd-roc"></div>`)
- Data: Dataset F
- Same layout as MRD ROC but colored by indication (CRC-stool blue, CRC-blood green, MCED red, Liver yellow, Lung purple)
- FDA-approved: gold star
- Hover: test name, vendor, sensitivity, specificity, indication
- Theme-aware

**Chart 2: Stage-sensitivity slope** (`<div id="ecd-stages"></div>`)
- Data: Dataset G
- X axis: Stage (I, II, III, IV) — categorical
- Y axis: Sensitivity (0-100%)
- One line per test, colored by indication
- Points at each stage with hover
- Highlight Galleri's cliff (thicker line, annotation)
- Theme-aware

**Chart 3: Geographic availability** (`<div id="ecd-geo"></div>`)
- Data: Dataset H aggregated — for each category (MRD, ECD, CGP, HCT): count of US-only, Intl-only, Both, Unknown
- Grouped horizontal bar chart, 4 groups, 4 bars each
- Color: one color per availability bucket
- Hover: specific test names
- Theme-aware

**Step 3: Test locally**

Run: `bundle exec jekyll serve --drafts --future`
Verify all 3 charts render, tooltips work, theme toggle works, mermaid diagram renders.

**Step 4: Commit**

```bash
git add _drafts/2026-02-18-early-cancer-detection-screening-wars.md
git commit -m "Add Part 3: Early Cancer Detection deep dive with D3 charts"
```

---

## Task 5: Cross-references and series navigation

**Files:**
- Modify: `_drafts/2026-02-16-cancer-testing-landscape.md`
- Modify: `_drafts/2026-02-17-mrd-hunting-invisible-cancer.md`
- Modify: `_drafts/2026-02-18-early-cancer-detection-screening-wars.md`

**Step 1: Add series navigation to each post**

Add to the top of each post (after the excerpt separator):

```markdown
> **This is Part N of a 3-part series on cancer diagnostics in 2026.**
> [Part 1: The Four Pillars](/2026/02/16/cancer-testing-landscape/) |
> [Part 2: MRD](/2026/02/17/mrd-hunting-invisible-cancer/) |
> [Part 3: Screening Wars](/2026/02/18/early-cancer-detection-screening-wars/)
```

Repeat the same block at the bottom of each post.

**Step 2: Verify links resolve locally**

Run: `bundle exec jekyll serve --drafts --future`
Click through all cross-reference links between parts.

**Step 3: Commit**

```bash
git add _drafts/2026-02-1*.md
git commit -m "Add cross-references between cancer diagnostics series posts"
```

---

## Task 6: Final review and polish

**Step 1: Full read-through of all 3 posts**

Check for:
- Consistent terminology across posts
- No broken mermaid diagrams
- All D3 charts load without console errors
- Data accuracy: spot-check 5 numbers against investigation.md and OpenOnco JSON
- Mobile responsiveness: test at 375px viewport width
- Theme toggle: all charts and diagrams update correctly

**Step 2: Fix any issues found**

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "Polish cancer diagnostics blog series"
```
