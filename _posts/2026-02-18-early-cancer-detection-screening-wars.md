---
layout: post
title: "Cancer Testing in 2026: The Screening Wars"
date: 2026-02-18 10:00:00 -0700
tags: biotech data-analysis
author: bolu-atx
categories: biotech
---

![The ECD Screening Wars: A Bioinformatician's Guide to Early Cancer Detection](/assets/posts-media/cancer-screening-wars.jpg)
*TL;DR: 31 tests fight over stool vs. blood, single-cancer vs. multi-cancer, and sensitivity vs. compliance --- stool wins on accuracy, blood wins on uptake, and nobody has cracked Stage I detection for the cancers that kill the most.*

Seventy percent of cancer deaths occur in organs with no screening guideline. Pancreatic cancer, ovarian cancer, liver cancer -- by the time symptoms appear, the survival window has closed. A blood test that catches 50 cancers at once sounds like science fiction. It is not. But as I dug into the [OpenOnco](https://openonco.org) data for this category, the story turned out to be more complicated than the headlines suggest.

<!--more-->

*This is Part 3 of a 3-part series on cancer diagnostics in 2026:
[Part 1: The Four Pillars](/biotech/2026/02/16/cancer-testing-landscape.html) |
[Part 2: MRD](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html) |
Part 3: Screening Wars (this post)*

I covered the [landscape overview in Part 1](/biotech/2026/02/16/cancer-testing-landscape.html) and [MRD in Part 2](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html). This final post digs into the most publicly visible and commercially contested category: **Early Cancer Detection (ECD)**.

## The screening gap

As far as I can tell, the US medical system has established screening programs for exactly four cancers: colorectal (colonoscopy, stool tests), breast (mammography), cervical (Pap/HPV), and lung (low-dose CT for heavy smokers). These are the cancers where decades of randomized controlled trials proved that catching it early saves lives.

But the top cancer killers tell a different story. Pancreatic cancer has a 12% five-year survival rate. Ovarian cancer is caught late in 60% of cases. Liver cancer incidence has tripled in the US since 1980. None of these have a recommended screening test for the general population.

This gap -- between what we *can* screen for and what actually kills people -- is what drives the entire ECD industry. The OpenOnco database lists **31 ECD tests** across stool, blood, urine, and saliva modalities. Some are FDA-approved and covered by Medicare. Many are lab-developed tests operating in regulatory gray zones. And performance varies by an order of magnitude depending on what you measure and when you measure it.

## CRC screening: the most mature battleground

Colorectal cancer has the most diagnostic competition of any cancer type, and for good reason. It is the second-leading cause of cancer death in the US, but it is also one of the most preventable -- if you catch it early or at the precancerous polyp stage.

Three modalities are now fighting for market share.

**Stool-based: Cologuard Plus (Exact Sciences).** The reigning champion. FDA approved October 2024 with 93.9% CRC sensitivity and 91% specificity. But the real number that matters: 43.4% sensitivity for advanced adenomas -- precancerous polyps that haven't turned malignant yet. Finding and removing these prevents cancer entirely. No blood test comes close to this precancer detection rate.

**Blood-based: Shield (Guardant Health).** The first FDA-approved blood test for primary CRC screening (July 2024). 83.1% CRC sensitivity, 89.6% specificity. Its advanced adenoma sensitivity is just 13.2% -- a fraction of the stool-based tests. But Shield has a different advantage: *compliance*. One in three Americans eligible for CRC screening skip it entirely. A simple blood draw at a routine checkup catches the people who would never mail back a stool sample or schedule a colonoscopy.

**The tradeoff is real.** A test that catches 83% of cancers but is taken by everyone may prevent more deaths than a 94% test taken by 67% of the eligible population. This is the central tension of the blood-vs-stool debate, and it will not be resolved by sensitivity data alone -- it requires population-level outcome studies.

**Freenome (Freenome / Exact Sciences).** The multiomics entrant. Combines DNA methylation, protein biomarkers, and immune signals in a single blood draw. Their PREEMPT CRC study (n=48,995 enrolled, 27,010 analyzed) reported 79.2% CRC sensitivity with 91.5% specificity. Stage I sensitivity of 57.1% -- better than Shield's 54.5% but still well below stool-based tests. Exact Sciences acquired an exclusive US license in August 2025. PMA submitted; FDA review pending.

**ColoSense (Geneoscopy).** The RNA-based wildcard. FDA approved May 2024 with 93% CRC sensitivity and a remarkable 100% Stage I detection rate. Uses stool-derived RNA transcripts rather than DNA methylation. Still early in commercial rollout via Labcorp.

## CRC screening decision tree

```mermaid
graph TD
    P["Patient age 45+<br/>Average CRC risk"] --> D{"Screening<br/>options"}

    D -->|"Gold standard"| C["Colonoscopy<br/>~67% compliance<br/>10yr interval"]
    D -->|"At-home stool"| S["Cologuard Plus<br/>94% CRC sens<br/>3yr interval"]
    D -->|"Blood draw"| B["Shield<br/>83% CRC sens<br/>3yr interval"]
    D -->|"Emerging"| F["Freenome / ColoSense<br/>79-93% sens<br/>pending scale"]

    C -->|"Positive"| DX["Diagnostic<br/>colonoscopy"]
    S -->|"Positive"| DX
    B -->|"Positive"| DX
    F -->|"Positive"| DX

    DX --> TX["Treatment<br/>or surveillance"]

    classDef input fill:none,stroke:#60a5fa,stroke-width:2px
    classDef highlight fill:none,stroke:#f472b6,stroke-width:2px
    classDef output fill:none,stroke:#34d399,stroke-width:2px
    classDef result fill:none,stroke:#a78bfa,stroke-width:2px
    class P input
    class C,S,B,F highlight
    class DX result
    class TX output
```

Every screening pathway converges on the same endpoint: if the test is positive, the patient gets a diagnostic colonoscopy. The debate is about how many cancers get caught on the way in, and how many patients actually show up.

## Sensitivity vs. specificity across the ECD landscape

Before going further, here's the full picture. The scatter plot below places every ECD test with published performance data in ROC space -- sensitivity on the Y axis, false positive rate (1 - specificity) on the X axis. Upper-left is best. Gold stars mark FDA-approved tests.

<div id="ecd-roc" style="width:100%;max-width:700px;margin:0 auto"></div>

A few patterns jumped out as I was putting this together:

- **CRC stool-based tests cluster in the upper-left** -- high sensitivity, high specificity. This is the most mature modality.
- **CRC blood-based tests trade sensitivity for compliance** -- they sit lower on the Y axis.
- **MCED tests push specificity to extremes** (Galleri at 99.5%, OverC at 98.9%) because when you screen millions of healthy people for dozens of cancers, even a 1% false positive rate generates an unacceptable number of false alarms.
- **Lung tests have notably lower specificity** -- FirstLook Lung at 58%, ProVue Lung at 55% -- because they are designed as pre-LDCT triage tools, not standalone screening tests.

## The Stage I problem

This is the chart that I think defines the entire ECD category.

Every test in this space exists to answer one question: *can you catch cancer before it spreads?* Stage I means the tumor is localized -- often curable with surgery alone. Stage IV means metastatic -- often fatal regardless of treatment. The clinical value of a screening test is almost entirely determined by its Stage I sensitivity.

And every single test shows a dramatic drop from Stage IV to Stage I detection. From what I've read, this is basically physics, not engineering failure -- early-stage tumors shed less DNA, fewer proteins, and fewer cells into the bloodstream. The signal-to-noise problem is immense.

<div id="ecd-stages" style="width:100%;max-width:700px;margin:0 auto"></div>

The slope of each line tells the story:

- **ColoSense** achieves 100% Stage I sensitivity -- but only for CRC via stool, where tumor DNA is shed directly into the collection medium rather than diluted through the bloodstream.
- **Cologuard Plus** drops from 100% (Stage IV) to 87% (Stage I) -- a remarkably gentle slope for a non-invasive test.
- **Shield** drops from 100% to 54.5% -- the blood-based penalty for CRC detection.
- **Galleri** falls off a cliff: 90.1% at Stage IV to 16.8% at Stage I. This is the number that Galleri's critics cite most often. When only 1 in 6 Stage I cancers is detected, the clinical utility of "early detection" becomes debatable.
- **EPISEEK** shows a flatter profile (45% Stage I/II to 74% Stage IV) for an MCED test -- but at much lower overall sensitivity.

The fundamental challenge: the cancers that benefit most from early detection are the hardest to detect early.

## MCED: Multi-Cancer Early Detection

Two platforms are trying to solve the screening gap with a single blood test.

### Galleri (GRAIL)

Arguably the most ambitious test in oncology. Galleri analyzes cfDNA methylation patterns to detect signals from 50+ cancer types -- including pancreatic, ovarian, and liver cancers that have no screening guideline. It also predicts the cancer's tissue of origin with 93% accuracy, telling your doctor where to look.

The specificity story is strong: 99.5%. In population screening, this matters enormously. Screen one million healthy people with a 99% specific test and you get 10,000 false positives -- each requiring expensive, anxiety-inducing follow-up imaging and biopsies. At 99.5%, you cut that to 5,000. Still a lot, but the math works better.

The sensitivity story is complicated: 51.5% overall, 16.8% at Stage I. The PATHFINDER 2 study (n=23,000+) showed that Galleri detected cancers 7x more often than standard USPSTF screening -- but most detected cancers were later-stage. More than 50% of Galleri-detected cancers were early-stage in PATHFINDER 2, and 75% were cancer types without any recommended screening. That last number is the real argument for Galleri: it finds cancers that no other test is looking for.

The NHS-Galleri trial (140,000 participants) and a PMA submission filed January 29, 2026, will determine whether Galleri becomes the first FDA-approved MCED test. Price: ~$949 out-of-pocket. Medicare does not cover it.

### Cancerguard (Exact Sciences)

Launched September 2025 through Quest Diagnostics' 7,000-site network. Takes a different philosophical approach: rather than detecting every possible cancer, Cancerguard focuses on the deadliest ones -- pancreatic, ovarian, liver, lung, colorectal, and esophageal. These are the cancers where early detection has the highest impact on survival.

64% overall sensitivity. 68% for the six deadliest cancers. 97.4% specificity. Price: $689 -- notably cheaper than Galleri. Built on the CancerSEEK and DETECT-A study foundation (10,000 participants), now enrolling 25,000 more in the FALCON registry.

### The MCED paradox

This is where the math gets interesting (and where my software brain perks up). High specificity is non-negotiable for population screening. If you screen 100 million Americans annually and your false positive rate is 2.6% (Cancerguard) instead of 0.5% (Galleri), that's 2.1 million additional false alarms per year. Each false alarm means imaging, possible biopsies, patient anxiety, and healthcare costs. But if your sensitivity for the earliest stages remains below 50%, you're missing the cancers that matter most.

No MCED test has solved this paradox yet. From what I can tell, the field is converging on a layered approach: MCED as a complement to existing single-cancer screening, not a replacement.

## Geographic availability: ECD's international dimension

One of the more surprising things I found in the OpenOnco data is that ECD has the most international-only tests of any category. While MRD and CGP are dominated by US-based companies, the ECD space includes a significant non-US contingent.

<div id="ecd-geo" style="width:100%;max-width:700px;margin:0 auto"></div>

Ten ECD tests are available internationally but not in the US. The biggest bloc: **Wuhan Ammunition Life Technology** from China, with five NMPA-approved tests covering CRC (IColocomf, IColohunter), esophageal (IEsohunter), liver (IHepcomf), and urothelial (IUrisure) cancer. Their published performance data is strong -- IColocomf reports 95.3% CRC sensitivity with 96.7% specificity, IHepcomf shows 92.3% liver cancer sensitivity -- though cohort sizes and study designs vary.

**Gene Solutions' SPOT-MAS** is the standout in Southeast Asia: a multi-cancer blood test validated in the K-DETEK trial (n=9,024) with 70.8% sensitivity and 99.7% specificity, available across Singapore, Vietnam, Malaysia, Thailand, Indonesia, and the Philippines. It is the first clinically validated MCED test in Asia.

**Burning Rock's OverC** holds a unique distinction: it is the only MCED test to receive Breakthrough Device Designation from both the US FDA and the Chinese NMPA. 69.1% sensitivity, 98.9% specificity across six cancer types, available in China and the EU.

This geographic fragmentation means the global ECD landscape looks very different depending on where you are standing. A patient in Shenzhen has access to organ-specific methylation tests that don't exist in the US. A patient in Ho Chi Minh City can get a 10-cancer blood test that isn't available in Europe. Regulatory harmonization is nowhere on the horizon.

## What is coming

Based on what I've seen in the data and filings, the next 12-18 months could reshape this category.

**Galleri's PMA review** is the biggest regulatory event in ECD history. If the FDA approves an MCED test for population screening, it opens the floodgates for Medicare coverage and commercial payer adoption. Congressional bills (H.R. 2407, S. 2085) are already pending to mandate Medicare MCED coverage.

**Freenome's CRC blood test** is headed for FDA review through its Exact Sciences partnership. If approved, it would be the second blood-based CRC screening option alongside Shield, intensifying the compliance-vs-sensitivity debate.

**Shield MCD** -- Guardant is expanding from CRC to multi-cancer detection using the same blood draw. Already launched nationally (October 2025) with FDA Breakthrough Device Designation and selected for the NCI Vanguard Study (24,000 participants). 60% sensitivity across 10 tumor types, 98.5% specificity.

**The blood-vs-stool war will intensify.** Cologuard Plus has better numbers. Shield has better compliance. The resolution will come from population outcome studies, not head-to-head sensitivity comparisons. The test that prevents the most deaths wins, and that depends as much on who takes the test as on what the test can detect.

## Series conclusion

I started this project because I stumbled onto OpenOnco and thought the dataset was too interesting not to explore. Three posts later, here's what stands out to me as a software person looking in from the outside.

Across the four pillars -- CGP for profiling, HCT for inherited risk, MRD for recurrence monitoring, and ECD for early detection -- every category seems to be converging on the same patient: a person whose molecular profile is tracked continuously from inherited risk assessment through screening, diagnosis, treatment, and survivorship.

The technical barriers are falling fast. Sub-1 ppm MRD detection. 50-cancer blood tests. Stool RNA with 100% Stage I CRC sensitivity. The remaining barriers are regulatory (only 1 FDA-approved MRD test, zero approved MCED tests), economic (Medicare coverage drives adoption), and infrastructural (integrating results from a dozen tests into a coherent clinical workflow).

That last one is where I think people like me might eventually have something to contribute. A patient who gets a Galleri MCED screen, a Cologuard Plus CRC screen, a Shield blood draw, and annual MRD monitoring after treatment is generating a stream of molecular data that no EHR system is designed to handle. The next wave of cancer diagnostics innovation may not come from the lab at all -- it may come from the data layer that connects everything together. That's a software problem.

*All data in this series comes from [OpenOnco](https://openonco.org) (v. Feb 15, 2026) -- an open-access database of 155 cancer diagnostic tests, 75 vendors, and 6,743 data points. Domain research was done with the help of Claude and Gemini, cross-referenced against published papers and FDA filings. Corrections welcome -- I am learning in public here.*

---

*[Part 1: The Four Pillars](/biotech/2026/02/16/cancer-testing-landscape.html) |
[Part 2: MRD](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html) |
Part 3: Screening Wars (this post)*


<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="/assets/js/cancer-charts.js"></script>
<script>
(function() {
  'use strict';

  // ============================================================
  // Chart 1: ECD ROC Scatter (Sensitivity vs Specificity)
  // ============================================================
  function drawRocChart() {
    var c = CancerCharts.getColors();
    var container = d3.select('#ecd-roc');
    container.selectAll('*').remove();

    var data = [
      { name: "Cologuard Plus", vendor: "Exact Sciences", sensitivity: 93.9, specificity: 91, indication: "CRC-stool", fda: true },
      { name: "ColoSense", vendor: "Geneoscopy", sensitivity: 93, specificity: 88, indication: "CRC-stool", fda: true },
      { name: "Cologuard", vendor: "Exact Sciences", sensitivity: 92, specificity: 87, indication: "CRC-stool", fda: true },
      { name: "IColocomf", vendor: "Wuhan Ammunition", sensitivity: 95.3, specificity: 96.7, indication: "CRC-stool", fda: false },
      { name: "Shield", vendor: "Guardant Health", sensitivity: 83.1, specificity: 89.6, indication: "CRC-blood", fda: true },
      { name: "Freenome CRC", vendor: "Freenome", sensitivity: 79.2, specificity: 91.5, indication: "CRC-blood", fda: false },
      { name: "IColohunter", vendor: "Wuhan Ammunition", sensitivity: 91.2, specificity: 92.4, indication: "CRC-blood", fda: false },
      { name: "Signal-C", vendor: "Universal DX", sensitivity: 93, specificity: 92, indication: "CRC-blood", fda: false },
      { name: "Epi proColon", vendor: "Epigenomics", sensitivity: 68, specificity: 80, indication: "CRC-blood", fda: true },
      { name: "Galleri", vendor: "GRAIL", sensitivity: 51.5, specificity: 99.5, indication: "MCED", fda: false },
      { name: "Shield MCD", vendor: "Guardant Health", sensitivity: 60, specificity: 98.5, indication: "MCED", fda: false },
      { name: "OverC", vendor: "Burning Rock", sensitivity: 69.1, specificity: 98.9, indication: "MCED", fda: false },
      { name: "EPISEEK", vendor: "Precision Epigenomics", sensitivity: 54, specificity: 99.5, indication: "MCED", fda: false },
      { name: "Cancerguard", vendor: "Exact Sciences", sensitivity: 64, specificity: 97.4, indication: "MCED", fda: false },
      { name: "SPOT-MAS", vendor: "Gene Solutions", sensitivity: 70.8, specificity: 99.7, indication: "MCED", fda: false },
      { name: "Trucheck Intelli", vendor: "Datar Cancer Genetics", sensitivity: 88.2, specificity: 96.3, indication: "MCED", fda: false },
      { name: "Oncoguard Liver", vendor: "Exact Sciences", sensitivity: 70, specificity: 81.9, indication: "Liver", fda: false },
      { name: "HelioLiver", vendor: "Helio Genomics", sensitivity: 47.8, specificity: 88, indication: "Liver", fda: false },
      { name: "IHepcomf", vendor: "Wuhan Ammunition", sensitivity: 92.3, specificity: 93.4, indication: "Liver", fda: false },
      { name: "MethylScan HCC", vendor: "EarlyDx", sensitivity: 84.5, specificity: 92, indication: "Liver", fda: false },
      { name: "FirstLook Lung", vendor: "DELFI Diagnostics", sensitivity: 80, specificity: 58, indication: "Lung", fda: false },
      { name: "ProVue Lung", vendor: "PrognomiQ", sensitivity: 85, specificity: 55, indication: "Lung", fda: false },
      { name: "IEsohunter", vendor: "Wuhan Ammunition", sensitivity: 87.4, specificity: 93.3, indication: "Other", fda: false },
      { name: "IUrisure", vendor: "Wuhan Ammunition", sensitivity: 93.9, specificity: 92, indication: "Other", fda: false },
      { name: "GALEAS Bladder", vendor: "Nonacus", sensitivity: 90, specificity: 85, indication: "Other", fda: false },
      { name: "CancerDetect", vendor: "Viome", sensitivity: 90, specificity: 95, indication: "Other", fda: false },
      { name: "Avantect Pancreatic", vendor: "ClearNote Health", sensitivity: 68.3, specificity: 96.9, indication: "Other", fda: false },
      { name: "ClarityDX Prostate", vendor: "Nanostics", sensitivity: 95, specificity: 35, indication: "Other", fda: false },
    ];

    var margin = { top: 35, right: 20, bottom: 50, left: 55 };
    var width = 700, height = 460;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // Axes: x = 1 - specificity (FPR), y = sensitivity
    var x = d3.scaleLinear().domain([0, 70]).range([0, innerW]);
    var y = d3.scaleLinear().domain([0, 100]).range([innerH, 0]);

    // Grid
    g.append('g').selectAll('line')
      .data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', innerW)
      .attr('y1', function(d) { return y(d); })
      .attr('y2', function(d) { return y(d); })
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    g.append('g').selectAll('line')
      .data(x.ticks(7)).enter().append('line')
      .attr('y1', 0).attr('y2', innerH)
      .attr('x1', function(d) { return x(d); })
      .attr('x2', function(d) { return x(d); })
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    // Axes
    g.append('g').attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x).ticks(7).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) { g.selectAll('text').attr('fill', c.text); g.selectAll('line,path').attr('stroke', c.muted); });

    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) { g.selectAll('text').attr('fill', c.text); g.selectAll('line,path').attr('stroke', c.muted); });

    // Axis labels
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', height - 6)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('False Positive Rate (1 - Specificity)');

    svg.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + innerH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('Sensitivity');

    // Title
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', 15).attr('font-weight', 600)
      .text('ECD Tests: Sensitivity vs. False Positive Rate');

    // "Better" arrow
    g.append('text').attr('x', 4).attr('y', 14)
      .attr('fill', c.muted).attr('font-size', 11).attr('font-style', 'italic')
      .text('\u2190 better');

    // Points
    var indicationColor = function(d) { return c[d.indication] || c.muted; };

    // Non-FDA circles
    var nonFda = data.filter(function(d) { return !d.fda; });
    g.selectAll('.dot-circle')
      .data(nonFda).enter().append('circle')
      .attr('cx', function(d) { return x(100 - d.specificity); })
      .attr('cy', function(d) { return y(d.sensitivity); })
      .attr('r', 6)
      .attr('fill', indicationColor)
      .attr('fill-opacity', 0.7)
      .attr('stroke', indicationColor)
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        CancerCharts.showTooltip(event,
          '<strong>' + d.name + '</strong><br/>' + d.vendor +
          '<br/>Sensitivity: ' + d.sensitivity + '%' +
          '<br/>Specificity: ' + d.specificity + '%' +
          '<br/>Indication: ' + d.indication);
      })
      .on('mousemove', CancerCharts.moveTooltip)
      .on('mouseout', CancerCharts.hideTooltip);

    // FDA stars
    var fdaTests = data.filter(function(d) { return d.fda; });
    g.selectAll('.dot-star')
      .data(fdaTests).enter().append('path')
      .attr('d', CancerCharts.STAR_PATH)
      .attr('transform', function(d) {
        return 'translate(' + x(100 - d.specificity) + ',' + y(d.sensitivity) + ')';
      })
      .attr('fill', c.yellow)
      .attr('stroke', indicationColor)
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        CancerCharts.showTooltip(event,
          '<strong>' + d.name + ' \u2605 FDA</strong><br/>' + d.vendor +
          '<br/>Sensitivity: ' + d.sensitivity + '%' +
          '<br/>Specificity: ' + d.specificity + '%' +
          '<br/>Indication: ' + d.indication);
      })
      .on('mousemove', CancerCharts.moveTooltip)
      .on('mouseout', CancerCharts.hideTooltip);

    // Legend
    var legendItems = [
      { label: 'CRC (stool)', color: c['CRC-stool'] },
      { label: 'CRC (blood)', color: c['CRC-blood'] },
      { label: 'MCED', color: c['MCED'] },
      { label: 'Liver', color: c['Liver'] },
      { label: 'Lung', color: c['Lung'] },
      { label: 'Other', color: c['Other'] },
      { label: 'FDA approved', color: c.yellow, star: true },
    ];
    var legend = g.append('g').attr('transform', 'translate(' + (innerW - 115) + ', 4)');
    legendItems.forEach(function(item, i) {
      var row = legend.append('g').attr('transform', 'translate(0,' + (i * 18) + ')');
      if (item.star) {
        row.append('path').attr('d', CancerCharts.STAR_PATH)
          .attr('transform', 'scale(0.7)').attr('fill', item.color);
      } else {
        row.append('circle').attr('r', 5).attr('fill', item.color).attr('fill-opacity', 0.7);
      }
      row.append('text').attr('x', 12).attr('y', 4)
        .attr('fill', c.text).attr('font-size', 11).text(item.label);
    });
  }

  // ============================================================
  // Chart 2: Stage-Sensitivity Slope Chart
  // ============================================================
  function drawStageChart() {
    var c = CancerCharts.getColors();
    var container = d3.select('#ecd-stages');
    container.selectAll('*').remove();

    var data = [
      { name: "ColoSense", indication: "CRC-stool", stages: { "I": 100, "II": 71.4, "III": 100 } },
      { name: "Cologuard Plus", indication: "CRC-stool", stages: { "I": 87, "II": 94, "III": 97, "IV": 100 } },
      { name: "Cologuard", indication: "CRC-stool", stages: { "I": 90, "II": 100, "III": 90, "IV": 75 } },
      { name: "Shield", indication: "CRC-blood", stages: { "I": 54.5, "II": 100, "III": 100, "IV": 100 } },
      { name: "Freenome CRC", indication: "CRC-blood", stages: { "I": 57.1, "II": 100, "III": 82.4, "IV": 100 } },
      { name: "FirstLook Lung", indication: "Lung", stages: { "I": 71, "II": 89, "III": 88, "IV": 98 } },
      { name: "Galleri", indication: "MCED", stages: { "I": 16.8, "II": 40.4, "III": 77, "IV": 90.1 } },
      { name: "EPISEEK", indication: "MCED", stages: { "I": 45, "II": 45, "III": 73, "IV": 74 } },
      { name: "IEsohunter", indication: "Other", stages: { "I": 78.5, "II": 87.3, "III": 92.5, "IV": 96.9 } },
      { name: "Epi proColon", indication: "CRC-blood", stages: { "I": 57, "II": 72, "III": 85, "IV": 78 } },
      { name: "SPOT-MAS", indication: "MCED", stages: { "I": 62.3 } },
      { name: "Signal-C", indication: "CRC-blood", stages: { "I": 91, "II": 92, "III": 91, "IV": 93 } },
    ];

    var stageLabels = ["I", "II", "III", "IV"];
    var margin = { top: 35, right: 140, bottom: 45, left: 55 };
    var width = 700, height = 440;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var x = d3.scalePoint().domain(stageLabels).range([0, innerW]).padding(0.15);
    var y = d3.scaleLinear().domain([0, 105]).range([innerH, 0]);

    // Grid
    g.append('g').selectAll('line')
      .data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', innerW)
      .attr('y1', function(d) { return y(d); })
      .attr('y2', function(d) { return y(d); })
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    // Axes
    g.append('g').attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x))
      .call(function(g) { g.selectAll('text').attr('fill', c.text).attr('font-size', 13); g.selectAll('line,path').attr('stroke', c.muted); });

    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) { g.selectAll('text').attr('fill', c.text); g.selectAll('line,path').attr('stroke', c.muted); });

    // Axis labels
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', height - 4)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('Cancer Stage at Detection');

    svg.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + innerH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('Sensitivity');

    // Title
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', 15).attr('font-weight', 600)
      .text('Stage-Specific Sensitivity: The Early Detection Cliff');

    var line = d3.line()
      .defined(function(d) { return d.val !== undefined; })
      .x(function(d) { return x(d.stage); })
      .y(function(d) { return y(d.val); });

    data.forEach(function(test) {
      var points = stageLabels.map(function(s) {
        return { stage: s, val: test.stages[s] };
      }).filter(function(p) { return p.val !== undefined; });
      if (points.length < 2) return;

      var color = c[test.indication] || c.muted;
      var isGalleri = test.name === "Galleri";
      var sw = isGalleri ? 3 : 1.5;
      var opacity = isGalleri ? 1 : 0.7;

      g.append('path')
        .datum(points)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', sw)
        .attr('stroke-opacity', opacity);

      g.selectAll('.pt-' + test.name.replace(/\s+/g, ''))
        .data(points).enter().append('circle')
        .attr('cx', function(d) { return x(d.stage); })
        .attr('cy', function(d) { return y(d.val); })
        .attr('r', isGalleri ? 5 : 3.5)
        .attr('fill', color)
        .attr('fill-opacity', opacity)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
          CancerCharts.showTooltip(event,
            '<strong>' + test.name + '</strong>' +
            '<br/>Stage ' + d.stage + ': ' + d.val + '%');
        })
        .on('mousemove', CancerCharts.moveTooltip)
        .on('mouseout', CancerCharts.hideTooltip);

      // Galleri annotation
      if (isGalleri && test.stages["I"] !== undefined) {
        g.append('text')
          .attr('x', x("I") + 6).attr('y', y(test.stages["I"]) - 10)
          .attr('fill', c['MCED']).attr('font-size', 11).attr('font-weight', 600)
          .text('16.8% at Stage I');
      }
    });

    // Legend (right side)
    var legendData = data.filter(function(d) {
      var pts = stageLabels.filter(function(s) { return d.stages[s] !== undefined; });
      return pts.length >= 2;
    });
    var legendG = g.append('g').attr('transform', 'translate(' + (innerW + 10) + ', 0)');
    legendData.forEach(function(test, i) {
      var row = legendG.append('g').attr('transform', 'translate(0,' + (i * 17) + ')');
      var color = c[test.indication] || c.muted;
      row.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 0).attr('y2', 0)
        .attr('stroke', color).attr('stroke-width', test.name === "Galleri" ? 3 : 1.5);
      row.append('text').attr('x', 20).attr('y', 4)
        .attr('fill', c.text).attr('font-size', 10).text(test.name);
    });
  }

  // ============================================================
  // Chart 3: Geographic Availability (grouped horizontal bars)
  // ============================================================
  function drawGeoChart() {
    var c = CancerCharts.getColors();
    var container = d3.select('#ecd-geo');
    container.selectAll('*').remove();

    var data = [
      { category: "MRD", usOnly: 6, intlOnly: 5, both: 9, unknown: 24 },
      { category: "ECD", usOnly: 6, intlOnly: 10, both: 4, unknown: 11 },
      { category: "CGP", usOnly: 5, intlOnly: 2, both: 7, unknown: 32 },
      { category: "HCT", usOnly: 9, intlOnly: 0, both: 3, unknown: 22 },
    ];

    var buckets = ["usOnly", "intlOnly", "both", "unknown"];
    var bucketLabels = { usOnly: "US only", intlOnly: "Intl only", both: "Both", unknown: "Unknown" };
    var bucketColors = {
      usOnly: c.blue,
      intlOnly: c.green,
      both: c.purple,
      unknown: c.muted
    };

    var margin = { top: 35, right: 30, bottom: 40, left: 55 };
    var width = 700, height = 340;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var categories = data.map(function(d) { return d.category; });
    var maxVal = d3.max(data, function(d) {
      return d3.max(buckets, function(b) { return d[b]; });
    });

    var y0 = d3.scaleBand().domain(categories).range([0, innerH]).padding(0.25);
    var y1 = d3.scaleBand().domain(buckets).range([0, y0.bandwidth()]).padding(0.08);
    var x = d3.scaleLinear().domain([0, maxVal + 4]).range([0, innerW]);

    // Grid
    g.append('g').selectAll('line')
      .data(x.ticks(6)).enter().append('line')
      .attr('x1', function(d) { return x(d); })
      .attr('x2', function(d) { return x(d); })
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    // Axes
    g.append('g')
      .call(d3.axisLeft(y0))
      .call(function(g) { g.selectAll('text').attr('fill', c.text).attr('font-size', 13).attr('font-weight', 600); g.selectAll('line,path').attr('stroke', c.muted); });

    g.append('g').attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x).ticks(6).tickFormat(function(d) { return d; }))
      .call(function(g) { g.selectAll('text').attr('fill', c.text); g.selectAll('line,path').attr('stroke', c.muted); });

    // Axis label
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', height - 4)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('Number of Tests');

    // Title
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text).attr('font-size', 15).attr('font-weight', 600)
      .text('Geographic Availability by Category');

    // Bars
    data.forEach(function(d) {
      var isECD = d.category === "ECD";
      buckets.forEach(function(b) {
        var barY = y0(d.category) + y1(b);
        var barW = x(d[b]);
        var barH = y1.bandwidth();
        var isHighlight = isECD && b === "intlOnly";

        g.append('rect')
          .attr('x', 0).attr('y', barY)
          .attr('width', barW).attr('height', barH)
          .attr('fill', bucketColors[b])
          .attr('fill-opacity', isHighlight ? 1 : 0.7)
          .attr('stroke', isHighlight ? c.pink : 'none')
          .attr('stroke-width', isHighlight ? 2 : 0)
          .attr('rx', 2)
          .style('cursor', 'pointer')
          .on('mouseover', function(event) {
            CancerCharts.showTooltip(event,
              '<strong>' + d.category + '</strong><br/>' +
              bucketLabels[b] + ': ' + d[b] + ' tests');
          })
          .on('mousemove', CancerCharts.moveTooltip)
          .on('mouseout', CancerCharts.hideTooltip);

        if (d[b] > 0) {
          g.append('text')
            .attr('x', barW + 4).attr('y', barY + barH / 2 + 4)
            .attr('fill', c.text).attr('font-size', 10)
            .text(d[b]);
        }
      });
    });

    // ECD annotation
    var ecdIntlY = y0("ECD") + y1("intlOnly") + y1.bandwidth() / 2;
    g.append('text')
      .attr('x', x(10) + 22).attr('y', ecdIntlY + 4)
      .attr('fill', c.pink).attr('font-size', 11).attr('font-weight', 600)
      .text('\u2190 Most intl-only tests');

    // Legend
    var legend = g.append('g').attr('transform', 'translate(' + (innerW - 195) + ',' + (innerH - 25) + ')');
    buckets.forEach(function(b, i) {
      var lx = i * 80;
      legend.append('rect').attr('x', lx).attr('y', 0).attr('width', 12).attr('height', 12)
        .attr('fill', bucketColors[b]).attr('fill-opacity', 0.7).attr('rx', 2);
      legend.append('text').attr('x', lx + 16).attr('y', 10)
        .attr('fill', c.text).attr('font-size', 10).text(bucketLabels[b]);
    });
  }

  // ============================================================
  // Init and theme change
  // ============================================================
  function drawAll() {
    drawRocChart();
    drawStageChart();
    drawGeoChart();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', drawAll);
  } else {
    drawAll();
  }
  document.addEventListener('themechange', drawAll);
})();
</script>
