---
layout: post
title: "Cancer Testing in 2026: MRD — Hunting Invisible Cancer"
date: 2026-02-17 10:00:00 -0700
tags: biotech data-analysis
author: bolu-atx
categories: biotech
---

![The MRD Detection Pipeline: Searching for Needles in a Haystack of Blood & Errors](/assets/posts-media/mrd-infographic.jpg)
*TL;DR: 44 tests race to find one-in-a-million cancer DNA fragments in your blood after treatment --- only one has FDA clearance, and the rest are fighting over patents, payers, and parts-per-million.*

You've been declared cancer-free. The CT scans are clean. Your oncologist uses words like "complete response." But somewhere in your body, a few thousand cells may have survived chemotherapy, evaded the immune system, and are quietly dividing. Of the 44 tests on the market designed to find them, only one has FDA clearance. As I dug into the [OpenOnco](https://openonco.org) data for this category, MRD turned out to be the most fascinating corner of the landscape -- a field exploding with innovation, locked in patent wars, and racing toward a regulatory reckoning.

<!--more-->

*[Part 1: The Four Pillars](/biotech/2026/02/16/cancer-testing-landscape.html) |
Part 2: MRD (this post) |
[Part 3: Screening Wars](/biotech/2026/02/18/early-cancer-detection-screening-wars.html)*

## The problem imaging cannot solve

A CT scan cannot resolve a tumor smaller than about 5mm --- roughly 100 million cells. By the time imaging detects a recurrence, the cancer has had months, sometimes years, to re-establish itself.

The numbers here are sobering: **30--50% of Stage III colorectal cancer patients who achieve complete surgical resection will relapse.** They were "cured" by every metric available. The question that haunts oncology is: *can we catch recurrence when it is a few thousand cells instead of a hundred million?*

The answer, as best I understand it, is circulating tumor DNA (ctDNA). When cancer cells die, they shed fragments of their DNA into the bloodstream. A standard blood draw, processed through the right assay, can detect these fragments at concentrations as low as 1 part per million --- finding one cancer-derived molecule among a million healthy ones. A study in the *New England Journal of Medicine* (April 2025) showed ctDNA detected recurrence at a median of **1.4 months** post-surgery, compared to **6.1 months** for imaging. That's a 4.7-month head start on treatment.

## Two philosophies, one goal

What I found most interesting about the MRD space is that the 44 tests split into two camps defined by a single architectural decision: do you sequence the tumor first? As a software person, this felt like the classic build-vs-buy tradeoff, except the stakes are cancer recurrence.

### Tumor-informed: the custom fingerprint

The tumor-informed approach sequences the patient's actual tumor tissue via whole-exome or whole-genome sequencing, identifies 16 to 200+ somatic mutations unique to that cancer, and builds a custom PCR panel that tracks those exact mutations in subsequent blood draws. Because you're looking for a known fingerprint, sensitivity is high --- you know exactly what to amplify. This approach also filters out clonal hematopoiesis of indeterminate potential (CHIP), the age-related mutations in blood cells that plague tumor-agnostic tests with false positives.

The downside: building a custom panel takes **4--6 weeks** from tissue receipt. You need resected or biopsied tumor tissue. And if the cancer evolves significantly, the original panel may miss new driver mutations.

### Tumor-agnostic: the wide net

Tumor-agnostic tests skip the tissue entirely. They use a pre-built panel probing common mutation sites plus **20,000+** methylation signals to detect cancer-derived DNA without knowing the patient's specific mutational profile. Results come back in **7--10 days**. No tissue required. No custom panel build.

The tradeoff: you're casting a wider net, which means slightly lower sensitivity for any individual patient's cancer. You may also pick up CHIP signals that a tumor-informed approach would have filtered out by design.

## How they work

### Tumor-informed workflow

```mermaid
graph LR
    A["Tumor Tissue<br/>(biopsy/resection)"] --> B["WES / WGS<br/>sequencing"]
    B --> C["Identify somatic<br/>mutations"]
    C --> D["Design custom<br/>PCR panel<br/>(4-6 weeks)"]
    D --> E["Serial blood<br/>draws"]
    E --> F{"ctDNA<br/>detected?"}
    F -->|Yes| G["Recurrence<br/>likely"]
    F -->|No| H["Continue<br/>monitoring"]

    classDef input fill:none,stroke:#60a5fa,stroke-width:2px
    classDef highlight fill:none,stroke:#f472b6,stroke-width:2px
    classDef output fill:none,stroke:#34d399,stroke-width:2px
    classDef negative fill:none,stroke:#f87171,stroke-width:2px
    classDef progress fill:none,stroke:#fbbf24,stroke-width:2px
    class A input
    class B,C highlight
    class D progress
    class E input
    class F highlight
    class G negative
    class H output
```

### Tumor-agnostic workflow

```mermaid
graph LR
    A["Blood draw<br/>(no tissue needed)"] --> B["Fixed panel +<br/>methylation<br/>analysis"]
    B --> C{"ctDNA<br/>detected?"}
    C -->|Yes| D["Recurrence<br/>likely"]
    C -->|No| E["Continue<br/>monitoring"]

    linkStyle 0 stroke:#60a5fa,stroke-width:2px
    linkStyle 1 stroke:#f472b6,stroke-width:2px
    linkStyle 2 stroke:#34d399,stroke-width:2px

    classDef input fill:none,stroke:#60a5fa,stroke-width:2px
    classDef highlight fill:none,stroke:#f472b6,stroke-width:2px
    classDef output fill:none,stroke:#34d399,stroke-width:2px
    classDef negative fill:none,stroke:#f87171,stroke-width:2px
    class A input
    class B highlight
    class C highlight
    class D negative
    class E output
```

Simpler. Faster. But the performance data tells a more nuanced story.

## The performance showdown

The ROC scatter below plots every MRD test for which we have both sensitivity and specificity data from the [OpenOnco](https://openonco.org) dataset. Upper-left is ideal --- high sensitivity, high specificity. Blue dots are tumor-informed tests; red dots are tumor-agnostic. The gold star marks the only FDA-cleared test in the field.

The pattern is clear: **tumor-informed tests cluster in the upper-left**, with Signatera, NeXT Personal, and Pathlight leading. The tumor-agnostic approaches (Reveal MRD, Tempus xM, Latitude) trade sensitivity for the convenience of not requiring tissue. Note the outlier clonoSEQ --- FDA-cleared since 2018, but exclusively for hematologic malignancies (multiple myeloma, ALL, CLL), not solid tumors.

<div id="mrd-roc" style="width:100%;max-width:700px;margin:0 auto"></div>

## The sensitivity spectrum

If the ROC chart shows who can find the signal, the limit of detection (LoD) chart shows *how faint a signal they can find*. LoD is measured in parts per million (ppm) --- the minimum detectable fraction of tumor-derived DNA in a blood sample.

The spread across the field is staggering: **three orders of magnitude** separate the most sensitive test (Foresight CLARITY at 0.7 ppm) from the least (Caris Assure at 500 ppm). All five of the top performers are tumor-informed. The best tumor-agnostic LoD is CancerDetect at 3.5 ppm --- impressive, but still 5x worse than the leader.

<div id="mrd-lod" style="width:100%;max-width:700px;margin:0 auto"></div>

## The leaders

Based on the data, a few tests stood out. I am not endorsing any of these -- I'm just reporting what the numbers and coverage landscape look like.

**Signatera (Natera)** appears to be the de facto standard. 94% sensitivity, 98% specificity, validated in over 300,000 patients across 70+ cancer types. Natera submitted its PMA to the FDA in February 2026. With Signatera Genome (their WGS-based next-gen version), LoD drops to 1 ppm. Medicare covers Signatera for CRC and bladder cancer. Self-pay is \$349; financial assistance brings it to \$0--149.

**NeXT Personal (Personalis)** posts the best raw numbers: 100% sensitivity, 99.9% specificity --- but from a smaller validation cohort (n=493). LoD of 1.7 ppm. The TRACERx lung cancer study published in *Cell* (December 2025) showed ctDNA-positive patients had 5x higher relapse risk. Personalis is positioning aggressively in biopharma partnerships.

**clonoSEQ (Adaptive Biotechnologies)** is the only FDA-cleared MRD test, authorized via De Novo in September 2018 for multiple myeloma and B-ALL, expanded to CLL in 2020. It uses immunosequencing rather than ctDNA, tracking rearranged immunoglobulin or T-cell receptor gene sequences in bone marrow at a sensitivity of 10^-6 (one cell in a million). CE-IVDR certified in August 2024. The limitation: it only works for hematologic cancers with clonal rearrangements.

**Reveal MRD (Guardant Health)** is the tumor-agnostic leader for solid tumors. 81% sensitivity, 98% specificity. No tissue required, results in 7--10 days. The COSMOS study validated its utility for predicting CRC recurrence. For patients who can't provide tumor tissue or need rapid results, Reveal seems like the pragmatic choice --- trading some sensitivity for accessibility.

## The regulatory gap

Forty-four tests. One FDA clearance. The chart below shows the breakdown: the majority of MRD tests are CLIA-validated laboratory-developed tests (LDTs), operating in a regulatory gray zone that the FDA has been signaling it intends to close. Signatera's PMA submission in February 2026 could be the second FDA authorization --- eight years after clonoSEQ.

<div id="mrd-regulatory" style="width:100%;max-width:700px;margin:0 auto"></div>

## The business side

The MRD market is not just a scientific race --- the more I researched, the more it looked like a legal and reimbursement minefield.

**Patent wars:** Natera won a preliminary injunction against NeoGenomics, restricting the sale of NeoGenomics' RaDaR test for certain uses. The dispute centers on patents covering tumor-informed ctDNA monitoring methods. This is not an isolated skirmish; the foundational IP in this space is being contested across multiple fronts.

**Coverage battles:** Medicare covers Signatera for colorectal and bladder cancer under specific CPT codes. But **Humana's liquid biopsy policy explicitly excludes Signatera** (CPT 0340U), and **Blue Cross Blue Shield of Michigan Policy 2002479** (effective January 1, 2026) lists MRD testing as investigational. UnitedHealthcare's 2026 Molecular Oncology Testing policy calls solid tumor MRD "unproven / not medically necessary." The gap between clinical evidence and payer acceptance remains wide.

**Pricing:** Signatera lists at \$349 self-pay with Natera's Patient Assistance Program offering financial assistance at \$0--149 on an income-based scale, plus interest-free 12-month payment plans. Most other tests do not publish patient-facing pricing, leaving costs opaque until the explanation of benefits arrives.

## What's next

MRD testing is the bridge between treatment and surveillance --- the molecular early warning system that imaging never was. The field seems to be consolidating around a few leaders, but the regulatory, reimbursement, and legal landscapes will shape which tests survive the next five years. As an outsider looking in, the gap between the science (which is remarkable) and the business reality (which is messy) is striking.

In **[Part 3: Screening Wars](/biotech/2026/02/18/early-cancer-detection-screening-wars.html)**, I look at the other end of the timeline: **early cancer detection** --- the tests trying to find cancer *before* you ever knew it was there. Multi-cancer early detection (MCED) tests like Galleri promise to screen for dozens of cancer types from a single blood draw, but Stage I sensitivity remains the hard problem. The data tells a complicated story.

<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="/assets/js/cancer-charts.js"></script>
<script>
(function() {
  'use strict';

  // ── Data ──────────────────────────────────────────────────────────────

  const mrdRocData = [
    { name: "NeXT Personal", vendor: "Personalis", sensitivity: 100, specificity: 99.9, approach: "tumor-informed", fdaCleared: false, cohort: 493 },
    { name: "Pathlight", vendor: "SAGA Diagnostics", sensitivity: 100, specificity: 100, approach: "tumor-informed", fdaCleared: false, cohort: 100 },
    { name: "FoundationOne Tracker", vendor: "Foundation Medicine / Natera", sensitivity: 100, specificity: 99.6, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "Haystack MRD", vendor: "Quest Diagnostics", sensitivity: 95, specificity: 100, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "Signatera Genome", vendor: "Natera", sensitivity: 94, specificity: 100, approach: "tumor-informed", fdaCleared: false, cohort: 392 },
    { name: "RaDaR ST", vendor: "NeoGenomics", sensitivity: 95.7, specificity: 91, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "Signatera", vendor: "Natera", sensitivity: 94, specificity: 98, approach: "tumor-informed", fdaCleared: false, cohort: 300000 },
    { name: "clonoSEQ", vendor: "Adaptive Biotechnologies", sensitivity: 95, specificity: 99, approach: "tumor-informed", fdaCleared: true, cohort: null },
    { name: "Labcorp Plasma Detect", vendor: "Labcorp", sensitivity: 95, specificity: 99.4, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "LymphoTrack Dx", vendor: "Invivoscribe", sensitivity: 98, specificity: 99, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "LymphoVista", vendor: "LIQOMICS", sensitivity: 100, specificity: 93, approach: "tumor-informed", fdaCleared: false, cohort: 160 },
    { name: "MRDVision", vendor: "Inocras", sensitivity: 94, specificity: 99, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "Foresight CLARITY", vendor: "Natera", sensitivity: 90.62, specificity: 97.65, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "CanCatch Custom", vendor: "Burning Rock", sensitivity: 98.7, specificity: 99, approach: "tumor-informed", fdaCleared: false, cohort: 181 },
    { name: "Oncodetect", vendor: "Exact Sciences", sensitivity: 91, specificity: 94, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "Foundation TI-WGS", vendor: "Foundation Medicine", sensitivity: 90, specificity: 100, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "K-4CARE", vendor: "Gene Solutions", sensitivity: 79, specificity: 99, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "Veracyte MRD", vendor: "Veracyte (C2i Genomics)", sensitivity: 91, specificity: 92, approach: "tumor-informed", fdaCleared: false, cohort: null },
    { name: "Invitae PCM", vendor: "Labcorp (Invitae)", sensitivity: 76.9, specificity: 100, approach: "tumor-informed", fdaCleared: false, cohort: 61 },
    { name: "Reveal MRD", vendor: "Guardant Health", sensitivity: 81, specificity: 98, approach: "tumor-naive", fdaCleared: false, cohort: null },
    { name: "Caris Assure", vendor: "Caris Life Sciences", sensitivity: 93.8, specificity: 99.6, approach: "tumor-naive", fdaCleared: false, cohort: null },
    { name: "Tempus xM MRD", vendor: "Tempus", sensitivity: 61.1, specificity: 94, approach: "tumor-naive", fdaCleared: false, cohort: 80 },
    { name: "Latitude", vendor: "Natera", sensitivity: 81, specificity: 97, approach: "tumor-naive", fdaCleared: false, cohort: null },
    { name: "CancerVista", vendor: "LIQOMICS", sensitivity: 80, specificity: 96.7, approach: "tumor-naive", fdaCleared: false, cohort: null },
    { name: "BD OneFlow B-ALL", vendor: "BD Biosciences", sensitivity: 96, specificity: 95, approach: "tumor-naive", fdaCleared: false, cohort: null },
    { name: "Guardant LUNAR", vendor: "Guardant Health", sensitivity: 56, specificity: 100, approach: "tumor-naive", fdaCleared: false, cohort: 103 },
    { name: "NavDx", vendor: "Naveris", sensitivity: 90.4, specificity: 98.6, approach: "tumor-naive", fdaCleared: false, cohort: null },
    { name: "CancerDetect", vendor: "IMBdx", sensitivity: 61.9, specificity: 99.9, approach: "tumor-informed", fdaCleared: false, cohort: 98 },
    { name: "Bladder EpiCheck", vendor: "Nucleix", sensitivity: 67, specificity: 84, approach: "tumor-naive", fdaCleared: false, cohort: 449 },
  ];

  const mrdLodData = [
    { name: "Foresight CLARITY", vendor: "Natera", lod: 0.7, approach: "tumor-informed" },
    { name: "Signatera Genome", vendor: "Natera", lod: 1.0, approach: "tumor-informed" },
    { name: "MRDVision", vendor: "Inocras", lod: 1.0, approach: "tumor-informed" },
    { name: "NeXT Personal", vendor: "Personalis", lod: 1.7, approach: "tumor-informed" },
    { name: "CancerDetect", vendor: "IMBdx", lod: 3.5, approach: "tumor-naive" },
    { name: "Haystack MRD", vendor: "Quest Diagnostics", lod: 6, approach: "tumor-informed" },
    { name: "RaDaR ST", vendor: "NeoGenomics", lod: 10, approach: "tumor-informed" },
    { name: "Labcorp Plasma Detect", vendor: "Labcorp", lod: 10, approach: "tumor-informed" },
    { name: "Foundation TI-WGS", vendor: "Foundation Medicine", lod: 10, approach: "tumor-informed" },
    { name: "Pathlight", vendor: "SAGA Diagnostics", lod: 10, approach: "tumor-informed" },
    { name: "Veracyte MRD", vendor: "Veracyte (C2i Genomics)", lod: 10, approach: "tumor-informed" },
    { name: "Oncodetect", vendor: "Exact Sciences", lod: 15, approach: "tumor-informed" },
    { name: "CanCatch Custom", vendor: "Burning Rock", lod: 40, approach: "tumor-informed" },
    { name: "Reveal MRD", vendor: "Guardant Health", lod: 50, approach: "tumor-naive" },
    { name: "Invitae PCM", vendor: "Labcorp (Invitae)", lod: 80, approach: "tumor-informed" },
    { name: "Signatera", vendor: "Natera", lod: 100, approach: "tumor-informed" },
    { name: "Tempus xM MRD", vendor: "Tempus", lod: 100, approach: "tumor-naive" },
    { name: "FoundationOne Tracker", vendor: "Foundation Medicine / Natera", lod: 100, approach: "tumor-informed" },
    { name: "Latitude", vendor: "Natera", lod: 100, approach: "tumor-naive" },
    { name: "Caris Assure", vendor: "Caris Life Sciences", lod: 500, approach: "tumor-naive" },
  ];

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

  // ── Chart 1: ROC Scatter ──────────────────────────────────────────────

  function drawRocChart() {
    var container = d3.select('#mrd-roc');
    container.selectAll('*').remove();
    var c = CancerCharts.getColors();

    var margin = { top: 30, right: 30, bottom: 50, left: 60 };
    var width = 700;
    var height = 500;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // ── Company color mapping ──
    function normalizeVendor(v) {
      if (v.indexOf('Foundation') > -1) return 'Foundation Medicine';
      if (v.indexOf('Natera') > -1) return 'Natera';
      if (v.indexOf('Labcorp') > -1 || v.indexOf('Invitae') > -1) return 'Labcorp';
      if (v.indexOf('Guardant') > -1) return 'Guardant Health';
      if (v.indexOf('LIQOMICS') > -1) return 'LIQOMICS';
      return v;
    }

    var vendorCounts = {};
    mrdRocData.forEach(function(d) {
      var nv = normalizeVendor(d.vendor);
      vendorCounts[nv] = (vendorCounts[nv] || 0) + 1;
    });

    var topVendors = Object.keys(vendorCounts)
      .filter(function(v) { return vendorCounts[v] >= 2; })
      .sort(function(a, b) { return vendorCounts[b] - vendorCounts[a]; });

    var vendorColors = {};
    var palette = [c.blue, c.green, c.pink, c.purple, c.red];
    topVendors.forEach(function(v, i) { vendorColors[v] = palette[i % palette.length]; });

    function getVendorColor(vendor) {
      return vendorColors[normalizeVendor(vendor)] || c.muted;
    }

    // ── Shape paths ──
    var DIAMOND = 'M0,-7 L7,0 L0,7 L-7,0 Z';

    // x = 1 - specificity (FPR), y = sensitivity
    var x = d3.scaleLinear().domain([0, 20]).range([0, plotW]);
    var y = d3.scaleLinear().domain([50, 101]).range([plotH, 0]);

    // Grid
    g.append('g').selectAll('line')
      .data(x.ticks(5))
      .join('line')
        .attr('x1', function(d) { return x(d); })
        .attr('x2', function(d) { return x(d); })
        .attr('y1', 0).attr('y2', plotH)
        .attr('stroke', c.grid).attr('stroke-dasharray', '2,3');

    g.append('g').selectAll('line')
      .data(y.ticks(5))
      .join('line')
        .attr('x1', 0).attr('x2', plotW)
        .attr('y1', function(d) { return y(d); })
        .attr('y2', function(d) { return y(d); })
        .attr('stroke', c.grid).attr('stroke-dasharray', '2,3');

    // Diagonal reference line
    g.append('line')
      .attr('x1', x(0)).attr('y1', y(100))
      .attr('x2', x(20)).attr('y2', y(80))
      .attr('stroke', c.muted).attr('stroke-dasharray', '6,4').attr('stroke-width', 1);

    // Axes
    var xAxis = d3.axisBottom(x).ticks(5).tickFormat(function(d) { return d + '%'; });
    var yAxis = d3.axisLeft(y).ticks(6).tickFormat(function(d) { return d + '%'; });

    g.append('g').attr('transform', 'translate(0,' + plotH + ')').call(xAxis)
      .selectAll('text,line,path').attr('color', c.text).attr('stroke', c.text);
    g.append('g').call(yAxis)
      .selectAll('text,line,path').attr('color', c.text).attr('stroke', c.text);

    // Axis labels
    svg.append('text')
      .attr('x', margin.left + plotW / 2).attr('y', height - 6)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', 13)
      .text('False Positive Rate (1 - Specificity)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + plotH / 2)).attr('y', 16)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', 13)
      .text('Sensitivity');

    // Title
    svg.append('text')
      .attr('x', margin.left + plotW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('MRD Test Performance (Sensitivity vs Specificity)');

    // ── Points ──
    mrdRocData.forEach(function(d) {
      var fpr = 100 - d.specificity;
      var px = x(fpr);
      var py = y(d.sensitivity);
      var color = getVendorColor(d.vendor);
      var isNaive = d.approach === 'tumor-naive';

      var point;
      if (isNaive) {
        point = g.append('path')
          .attr('d', DIAMOND)
          .attr('transform', 'translate(' + px + ',' + py + ')')
          .attr('fill', color).attr('fill-opacity', 0.7)
          .attr('stroke', color).attr('stroke-width', 1.5);
      } else {
        point = g.append('circle')
          .attr('cx', px).attr('cy', py).attr('r', 7)
          .attr('fill', color).attr('fill-opacity', 0.7)
          .attr('stroke', color).attr('stroke-width', 1.5);
      }

      if (d.fdaCleared) {
        g.append('path')
          .attr('d', CancerCharts.STAR_PATH)
          .attr('transform', 'translate(' + px + ',' + py + ') scale(1.8)')
          .attr('fill', 'none').attr('stroke', c.yellow).attr('stroke-width', 2)
          .style('pointer-events', 'none');
      }

      point.style('cursor', 'pointer')
        .on('mouseover', function(event) {
          if (isNaive) {
            d3.select(this).attr('transform', 'translate(' + px + ',' + py + ') scale(1.3)');
          } else {
            d3.select(this).attr('r', 10);
          }
          CancerCharts.showTooltip(event, tooltipHtml(d));
        })
        .on('mousemove', function(event) { CancerCharts.moveTooltip(event); })
        .on('mouseout', function() {
          if (isNaive) {
            d3.select(this).attr('transform', 'translate(' + px + ',' + py + ')');
          } else {
            d3.select(this).attr('r', 7);
          }
          CancerCharts.hideTooltip();
        });
    });

    function tooltipHtml(d) {
      var color = getVendorColor(d.vendor);
      var html = '<strong>' + d.name + '</strong><br/>';
      html += '<span style="color:' + color + '">' + d.vendor + '</span><br/>';
      html += 'Sensitivity: ' + d.sensitivity + '%<br/>';
      html += 'Specificity: ' + d.specificity + '%<br/>';
      html += 'Approach: ' + d.approach.replace('tumor-informed', 'Tumor-informed').replace('tumor-naive', 'Tumor-agnostic') + '<br/>';
      if (d.cohort) html += 'Cohort: ' + d.cohort.toLocaleString() + ' patients<br/>';
      if (d.fdaCleared) html += '<strong style="color:' + c.yellow + '">FDA Cleared</strong>';
      return html;
    }

    // ── Legend (top-right) ──
    var legX = plotW - 150;
    var leg = g.append('g').attr('transform', 'translate(' + legX + ',0)');
    var row = 0;
    var lh = 16;

    // Background (sized after building legend)
    var legBg = leg.append('rect')
      .attr('x', -8).attr('y', -8)
      .attr('fill', c.bg).attr('fill-opacity', 0.85)
      .attr('stroke', c.grid).attr('rx', 4);

    // Company colors
    topVendors.forEach(function(v) {
      leg.append('circle').attr('cx', 5).attr('cy', row * lh).attr('r', 5).attr('fill', vendorColors[v]);
      leg.append('text').attr('x', 16).attr('y', row * lh + 4).attr('fill', c.text).attr('font-size', 11).text(v);
      row++;
    });
    leg.append('circle').attr('cx', 5).attr('cy', row * lh).attr('r', 5).attr('fill', c.muted);
    leg.append('text').attr('x', 16).attr('y', row * lh + 4).attr('fill', c.text).attr('font-size', 11).text('Other');
    row += 1.6;

    // Shape: approach
    leg.append('circle').attr('cx', 5).attr('cy', row * lh).attr('r', 5)
      .attr('fill', 'none').attr('stroke', c.text).attr('stroke-width', 1.5);
    leg.append('text').attr('x', 16).attr('y', row * lh + 4).attr('fill', c.text).attr('font-size', 11).text('Tumor-informed');
    row++;
    leg.append('path').attr('d', 'M0,-5 L5,0 L0,5 L-5,0 Z')
      .attr('transform', 'translate(5,' + (row * lh) + ')')
      .attr('fill', 'none').attr('stroke', c.text).attr('stroke-width', 1.5);
    leg.append('text').attr('x', 16).attr('y', row * lh + 4).attr('fill', c.text).attr('font-size', 11).text('Tumor-agnostic');
    row += 1.6;

    // FDA cleared
    leg.append('path').attr('d', CancerCharts.STAR_PATH)
      .attr('transform', 'translate(5,' + (row * lh) + ') scale(0.8)')
      .attr('fill', 'none').attr('stroke', c.yellow).attr('stroke-width', 2);
    leg.append('text').attr('x', 16).attr('y', row * lh + 4).attr('fill', c.text).attr('font-size', 11).text('FDA Cleared');

    // Size background to fit
    legBg.attr('width', 162).attr('height', row * lh + 16);
  }

  // ── Chart 2: LoD Lollipop ─────────────────────────────────────────────

  function drawLodChart() {
    var container = d3.select('#mrd-lod');
    container.selectAll('*').remove();
    var c = CancerCharts.getColors();

    var margin = { top: 30, right: 50, bottom: 50, left: 170 };
    var width = 700;
    var height = 40 * mrdLodData.length + margin.top + margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('height', 'auto')
      .style('font-family', 'inherit');

    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var sorted = mrdLodData.slice().sort(function(a, b) { return a.lod - b.lod; });

    var x = d3.scaleLog().domain([0.5, 700]).range([0, plotW]);
    var yBand = d3.scaleBand().domain(sorted.map(function(d) { return d.name; })).range([0, plotH]).padding(0.3);

    // Grid
    var ticks = [1, 10, 100, 500];
    g.selectAll('.grid-line')
      .data(ticks)
      .join('line')
        .attr('x1', function(d) { return x(d); })
        .attr('x2', function(d) { return x(d); })
        .attr('y1', 0).attr('y2', plotH)
        .attr('stroke', c.grid).attr('stroke-dasharray', '2,3');

    // Axes
    var xAxis = d3.axisBottom(x).tickValues(ticks).tickFormat(function(d) { return d + ' ppm'; });
    g.append('g').attr('transform', 'translate(0,' + plotH + ')').call(xAxis)
      .selectAll('text,line,path').attr('color', c.text).attr('stroke', c.text);

    g.append('g').call(d3.axisLeft(yBand).tickSize(0))
      .selectAll('text').attr('fill', c.text).attr('font-size', 11);
    g.select('.domain').attr('stroke', c.text);

    // Lollipops
    sorted.forEach(function(d) {
      var color = c[d.approach] || c.muted;
      var yPos = yBand(d.name) + yBand.bandwidth() / 2;

      g.append('line')
        .attr('x1', x(0.5)).attr('x2', x(d.lod))
        .attr('y1', yPos).attr('y2', yPos)
        .attr('stroke', color).attr('stroke-width', 2);

      g.append('circle')
        .attr('cx', x(d.lod)).attr('cy', yPos).attr('r', 6)
        .attr('fill', color).attr('fill-opacity', 0.8)
        .attr('stroke', color).attr('stroke-width', 1.5)
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          d3.select(this).attr('r', 9);
          CancerCharts.showTooltip(event,
            '<strong>' + d.name + '</strong><br/>' +
            '<span style="color:' + c.muted + '">' + d.vendor + '</span><br/>' +
            'LoD: ' + d.lod + ' ppm<br/>' +
            'Approach: ' + d.approach.replace('tumor-informed', 'Tumor-informed').replace('tumor-naive', 'Tumor-agnostic')
          );
        })
        .on('mousemove', function(event) { CancerCharts.moveTooltip(event); })
        .on('mouseout', function() {
          d3.select(this).attr('r', 6);
          CancerCharts.hideTooltip();
        });
    });

    // Title
    svg.append('text')
      .attr('x', margin.left + plotW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('MRD Limit of Detection (lower = more sensitive)');

    // Axis label
    svg.append('text')
      .attr('x', margin.left + plotW / 2).attr('y', height - 6)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', 13)
      .text('Limit of Detection (ppm, log scale)');

    // 3 orders of magnitude annotation
    var arrowY = plotH + 32;
    g.append('line')
      .attr('x1', x(0.7)).attr('x2', x(500))
      .attr('y1', arrowY).attr('y2', arrowY)
      .attr('stroke', c.muted).attr('stroke-width', 1);
    g.append('line')
      .attr('x1', x(0.7)).attr('x2', x(0.7))
      .attr('y1', arrowY - 4).attr('y2', arrowY + 4)
      .attr('stroke', c.muted).attr('stroke-width', 1);
    g.append('line')
      .attr('x1', x(500)).attr('x2', x(500))
      .attr('y1', arrowY - 4).attr('y2', arrowY + 4)
      .attr('stroke', c.muted).attr('stroke-width', 1);

    // Legend
    var leg = svg.append('g').attr('transform', 'translate(' + (margin.left + plotW - 140) + ',' + (margin.top + 6) + ')');
    leg.append('circle').attr('cx', 0).attr('cy', 0).attr('r', 5).attr('fill', c['tumor-informed']);
    leg.append('text').attr('x', 10).attr('y', 4).attr('fill', c.text).attr('font-size', 11).text('Tumor-informed');
    leg.append('circle').attr('cx', 0).attr('cy', 18).attr('r', 5).attr('fill', c['tumor-naive']);
    leg.append('text').attr('x', 10).attr('y', 22).attr('fill', c.text).attr('font-size', 11).text('Tumor-agnostic');
  }

  // ── Chart 3: Regulatory Bar ───────────────────────────────────────────

  function drawRegChart() {
    var container = d3.select('#mrd-regulatory');
    container.selectAll('*').remove();
    var c = CancerCharts.getColors();

    var margin = { top: 30, right: 60, bottom: 40, left: 180 };
    var width = 700;
    var barHeight = 32;
    var sorted = mrdRegData.slice().sort(function(a, b) { return b.count - a.count; });
    var total = sorted.reduce(function(s, d) { return s + d.count; }, 0);
    var height = sorted.length * barHeight + margin.top + margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var x = d3.scaleLinear().domain([0, 16]).range([0, plotW]);
    var yBand = d3.scaleBand().domain(sorted.map(function(d) { return d.status; })).range([0, plotH]).padding(0.25);

    function barColor(status) {
      if (status.indexOf('Approved') > -1) return c.green;
      if (status.indexOf('Cleared') > -1 || status.indexOf('Authorized') > -1) return c.green;
      if (status === 'FDA De Novo' || status === 'FDA 510(k)') return c.green;
      if (status === 'CE-IVD/IVDR') return c.blue;
      if (status === 'Breakthrough Designation') return c.yellow;
      if (status === 'Research Use Only') return c.purple;
      if (status === 'Discontinued') return c.red;
      return c.muted;
    }

    // Grid
    g.selectAll('.grid-line')
      .data(x.ticks(4))
      .join('line')
        .attr('x1', function(d) { return x(d); })
        .attr('x2', function(d) { return x(d); })
        .attr('y1', 0).attr('y2', plotH)
        .attr('stroke', c.grid).attr('stroke-dasharray', '2,3');

    // Bars
    sorted.forEach(function(d) {
      var yPos = yBand(d.status);
      var bColor = barColor(d.status);

      g.append('rect')
        .attr('x', 0).attr('y', yPos)
        .attr('width', x(d.count)).attr('height', yBand.bandwidth())
        .attr('fill', bColor).attr('fill-opacity', 0.75)
        .attr('rx', 3)
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          d3.select(this).attr('fill-opacity', 1);
          CancerCharts.showTooltip(event,
            '<strong>' + d.status + '</strong><br/>' +
            'Count: ' + d.count + ' tests<br/>' +
            'Share: ' + Math.round(d.count / total * 100) + '% of MRD tests'
          );
        })
        .on('mousemove', function(event) { CancerCharts.moveTooltip(event); })
        .on('mouseout', function() {
          d3.select(this).attr('fill-opacity', 0.75);
          CancerCharts.hideTooltip();
        });

      // Count label
      g.append('text')
        .attr('x', x(d.count) + 6).attr('y', yPos + yBand.bandwidth() / 2 + 4)
        .attr('fill', c.text).attr('font-size', 12).attr('font-weight', 600)
        .text(d.count);
    });

    // Y axis labels
    g.append('g').call(d3.axisLeft(yBand).tickSize(0))
      .selectAll('text').attr('fill', c.text).attr('font-size', 11);
    g.select('.domain').attr('stroke', c.text);

    // Title
    svg.append('text')
      .attr('x', margin.left + plotW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('MRD Test Regulatory Status (44 tests)');
  }

  // ── Init & theme listener ─────────────────────────────────────────────

  function drawAll() {
    drawRocChart();
    drawLodChart();
    drawRegChart();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', drawAll);
  } else {
    drawAll();
  }

  document.addEventListener('themechange', drawAll);
})();
</script>
