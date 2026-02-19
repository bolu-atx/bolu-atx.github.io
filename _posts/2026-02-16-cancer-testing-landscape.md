---
layout: post
title: "OpenOnco Assay Insights: The Four Pillars of Molecular Oncology"
date: 2026-02-16 10:00:00 -0700
tags: biotech data-analysis
author: bolu-atx
categories: biotech
---

I stumbled onto [OpenOnco](https://openonco.org) a few weeks ago and couldn't stop scrolling. It's an open-source database that catalogs the molecular oncology testing landscape -- 155 tests, 75 vendors, 6,743 trackable data points covering everything from turnaround times to FDA statuses to reimbursement codes. For someone like me -- a software engineer who has spent time adjacent to bioinformatics but has never designed an assay -- it was a goldmine. I could finally see the shape of an industry I'd been curious about for years.

I am not an assay scientist. Most of the domain context in this series comes from hours of research with the help of Claude and Gemini, cross-referenced against the OpenOnco dataset, published papers, and FDA filings. Think of this as a software person's field guide to molecular oncology testing -- what I found when I tried to make sense of the landscape, with all the caveats that implies.

This is the first post in a three-part series. Part 1 (this post) maps the four categories of cancer molecular testing and introduces the dataset. Part 2 dives into MRD -- the fastest-moving category where a single test (Signatera) dominates reimbursement while 43 competitors fight for clinical evidence. Part 3 covers the early cancer detection wars -- blood vs. stool, single-cancer vs. multi-cancer, and the FDA's unprecedented approval streak in 2024.

<!--more-->

*Part 1: The Four Pillars (this post) |
[Part 2: MRD](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html) |
[Part 3: Screening Wars](/biotech/2026/02/18/early-cancer-detection-screening-wars.html)*

## The four pillars

Every molecular cancer test falls into one of four categories. They correspond to four questions a patient might ask across the cancer continuum.

### HCT -- Hereditary Cancer Testing

*"Am I at genetic risk for cancer?"*

HCT tests sequence your germline DNA -- the DNA you inherited from your parents -- to look for pathogenic variants in genes like BRCA1, BRCA2, and the Lynch syndrome mismatch repair genes (MLH1, MSH2, MSH6, PMS2). If you carry one of these variants, your lifetime cancer risk can be 5-10x higher than average. There are **34 tests** in this category, from Myriad's MyRisk (the original) to newer panels from Ambry, Invitae/Labcorp, Color Health, and even 23andMe. Almost every one is a CLIA LDT. Only two have FDA clearance: Myriad's BRACAnalysis CDx (a companion diagnostic for PARP inhibitors) and 23andMe's DTC BRCA report. The technology is mature and commoditized -- all of these tests cluster at 99%+ analytical sensitivity and specificity. The differentiation is in panel breadth, turnaround time, and price.

### ECD -- Early Cancer Detection

*"Do I have cancer right now, even though I feel fine?"*

ECD tests screen asymptomatic people for cancer. This is the most controversial and fastest-growing category. There are **31 tests** here, split between stool-based colorectal screening (Cologuard, ColoSense) and blood-based approaches. The blood tests further divide into single-cancer screens (Shield for CRC, HelioLiver for HCC, FirstLook Lung) and multi-cancer early detection tests (Galleri, Shield MCD, OverC). The blood-based tests rely on cfDNA methylation patterns, fragmentomics, or protein biomarkers to detect cancer signal in people who have no symptoms. 2024 was a breakout year: the FDA approved Shield, Cologuard Plus, and ColoSense in rapid succession. GRAIL filed its PMA for Galleri in January 2026. The fundamental tension in this category is sensitivity vs. specificity at population scale -- even a 0.5% false positive rate means thousands of unnecessary follow-ups when you screen millions of healthy people.

### CGP -- Comprehensive Genomic Profiling

*"What mutations drive my tumor, and which drugs might work?"*

CGP tests come after diagnosis. They profile the tumor's genomic landscape -- somatic mutations, fusions, amplifications, microsatellite instability, tumor mutational burden -- to match patients with targeted therapies or immunotherapy. This is the most mature and most regulated category, with **46 tests**. Foundation Medicine's FoundationOne CDx was the first FDA-approved broad companion diagnostic (2017). Since then, Guardant360 CDx, Tempus xT CDx, MI Cancer Seek, TruSight Oncology Comprehensive, and several others have gained FDA approval or clearance. The category includes both tissue-based panels (FoundationOne CDx, MSK-IMPACT) and liquid biopsy panels (FoundationOne Liquid CDx, Guardant360 CDx) that profile circulating tumor DNA from a blood draw.

### MRD -- Molecular Residual Disease

*"After treatment, is my cancer truly gone?"*

MRD tests detect trace amounts of circulating tumor DNA (ctDNA) in the blood after surgery or treatment. The idea: if you can find tumor DNA fragments at parts-per-million concentrations, you can catch recurrence months before imaging (CT scans, MRI) would show anything visible. There are **44 tests** in this category -- the second-largest, and the fastest-moving. But only one has FDA clearance: Adaptive Biotechnologies' clonoSEQ, which is cleared for blood cancers (multiple myeloma, ALL, CLL) -- not solid tumors. Natera's Signatera dominates the solid tumor MRD space as a CLIA LDT with ~70% commercial payer coverage. The category splits between tumor-informed approaches (Signatera, Haystack MRD, RaDaR) that require sequencing the original tumor to build a custom assay, and tumor-agnostic approaches (Guardant Reveal, Oncodetect) that use fixed panels plus methylation.


## The patient journey

These four pillars map onto a patient's trajectory through the cancer care continuum:

```mermaid
graph LR
    A["Healthy Person"] --> B["Am I at risk?<br/><b>HCT</b>"]
    B --> C["Do I have cancer?<br/><b>ECD</b>"]
    C --> D["What drives it?<br/><b>CGP</b>"]
    D --> E["Is it gone?<br/><b>MRD</b>"]
    E --> F["Surveillance"]
    F -.->|"recurrence?"| D

    classDef default fill:none,stroke-width:2px
    classDef hct fill:none,stroke:#fbbf24,stroke-width:2px
    classDef ecd fill:none,stroke:#34d399,stroke-width:2px
    classDef cgp fill:none,stroke:#a78bfa,stroke-width:2px
    classDef mrd fill:none,stroke:#60a5fa,stroke-width:2px
    classDef neutral fill:none,stroke:#9ca3af,stroke-width:2px

    class A neutral
    class B hct
    class C ecd
    class D cgp
    class E mrd
    class F neutral
```

Not every patient moves through all four stages. Someone with a BRCA2 variant (HCT) might get enhanced screening but never develop cancer. A patient diagnosed via conventional imaging skips ECD entirely. And MRD monitoring only applies after curative-intent treatment (surgery, chemo, radiation) -- metastatic patients are typically managed with CGP-guided targeted therapy instead.


## Key concepts

Before diving into the data, a few definitions that will recur throughout this series. These are simplified -- if you're an assay developer, you'll find these reductive, but they're good enough for the data exploration that follows.

**cfDNA (cell-free DNA).** When cells die -- normal turnover, inflammation, or tumor cell death -- they release fragments of their DNA into the bloodstream. These fragments are short (~160 base pairs) and get cleared within hours. Tumor-derived cfDNA (ctDNA) carries the mutations and methylation patterns of the cancer, which is what these tests detect. The challenge: ctDNA is a needle in a haystack, often less than 0.1% of total cfDNA in early-stage disease.

**Sensitivity vs. Specificity.** Sensitivity is the true positive rate -- what fraction of actual cancers does the test catch? Specificity is the true negative rate -- what fraction of cancer-free people get a clean result? Both matter, but they trade off. A test can achieve 100% sensitivity by calling everything positive, but specificity drops to zero. In population screening (ECD), specificity matters more than most people realize: screening 1 million people with a 99.5% specific test still generates 5,000 false positives.

**CHIP (Clonal Hematopoiesis of Indeterminate Potential).** As you age, some of your blood stem cells accumulate mutations and expand clonally -- not cancer, but a pre-cancerous state. CHIP-associated mutations (DNMT3A, TET2, ASXL1) show up in cfDNA and can fool ctDNA-based tests into calling a false positive. This is the single biggest technical challenge for both MRD and ECD tests. Tumor-informed MRD assays sidestep CHIP by only tracking mutations known to come from the tumor. Tumor-agnostic assays must filter CHIP computationally, which is imperfect.

**Tumor-informed vs. Tumor-agnostic.** Two approaches to MRD testing. Tumor-informed assays (Signatera, Haystack MRD, RaDaR) start by sequencing the patient's tumor and matched normal tissue, then build a custom PCR panel tracking 16-50 tumor-specific variants. This gives higher sensitivity (~0.001% tumor fraction) but requires tissue and takes 2-6 weeks to build. Tumor-agnostic assays (Guardant Reveal, Oncodetect) use a fixed gene panel plus methylation markers -- no tissue needed, results in ~7 days, but generally lower sensitivity for small tumors.

**LDT vs. FDA.** An LDT (Lab Developed Test) is a test built and validated by a single CLIA-certified lab. No FDA review required -- the lab self-validates under CLIA regulations. An FDA-approved test (PMA, 510(k), or De Novo) has undergone formal FDA analytical and clinical validation review. As of early 2026, the vast majority of cancer molecular tests are LDTs. The FDA finalized its rule to regulate LDTs in 2024, with a phased enforcement timeline starting in 2025 -- this is reshaping the entire industry, as labs race to either submit for FDA approval or argue for exemptions.


## The data

All of the analysis in this series is built on data from [OpenOnco](https://openonco.org). It's genuinely impressive work -- someone (or a team) has methodically cataloged the entire molecular oncology testing landscape and made it open-source. The dataset covers **155 tests** from **75 vendors** across the four categories. There are **6,743 total data points** tracked per test (clinical performance, regulatory status, reimbursement, sample requirements, turnaround time, etc.) with a **62% fill rate** -- meaning 4,202 fields have data. Data quality is tiered: Tier 1 data (sensitivity, specificity, regulatory status) has a 99% citation rate. About **20% of tests** (31 out of 155) have vendor-verified data, where the test manufacturer reviewed and corrected OpenOnco's entries.

The coverage is not uniform, which itself is interesting. CGP has the deepest data (longest-established category with the most FDA-approved tests). MRD has the most active changelog (fastest-moving category). ECD has the most contested data points (vendors actively disputing each other's claimed performance). HCT has the shallowest data -- the tests are commoditized and the vendors have little incentive to differentiate on published metrics.


## The landscape at a glance

The treemap below shows every test in the dataset, grouped by category. Tile size is uniform -- this is a map of the market's breadth, not revenue. The number on each tile is the count of cancer types the test covers. Opacity indicates regulatory status: fully opaque tiles are FDA approved or cleared; translucent tiles are CLIA LDTs or earlier-stage. Hover for details on cancer types, sample requirements, and clinical performance.

<div id="treemap" style="width:100%;max-width:800px;margin:0 auto"></div>


## The acceleration curve

New cancer tests are not launching at a steady rate. The chart below shows tests with identifiable launch or first-availability dates, stacked by category. The pattern is clear: CGP dominated 2017-2022, then MRD and ECD exploded starting in 2023.

<div id="timeline" style="width:100%;max-width:800px;margin:0 auto"></div>
<p style="font-size:0.85em;color:var(--text-muted, #6b6b8a);text-align:center;margin-top:4px;">
  100 tests had no identifiable launch year (including all 34 HCT tests).
</p>


## What's next

In **[Part 2: MRD](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html)**, I dig into MRD -- the fastest-moving category. I compare tumor-informed vs. tumor-agnostic approaches, look at the clinical evidence landscape, try to understand Signatera's dominance (and its vulnerabilities), and visualize the sensitivity-vs-turnaround tradeoff across all 44 tests.

In **[Part 3: Screening Wars](/biotech/2026/02/18/early-cancer-detection-screening-wars.html)**, I tackle the early cancer detection wars -- blood vs. stool, single-cancer vs. multi-cancer screening, the FDA's 2024 approval streak, and the population-scale math that makes specificity the most important number in medicine.

---

I've been having way too much fun with Gemini's image generation lately -- the xkcd-style hand-drawn look is irresistible for summarizing dense topics. Here's my attempt at a one-page takeaway for this post.

![Cancer Detection Molecular Assays: A Crash Course for Code Monkeys](/assets/posts-media/cancer-detection.jpg)
*TL;DR: 155 tests from 75 vendors now cover the full cancer timeline --- risk, screening, profiling, and recurrence monitoring --- but most are unregulated LDTs and the data is a mess.*

<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="/assets/js/cancer-charts.js"></script>

<script>
(function() {
  'use strict';

  // ── Treemap data ──
  const treemapData = {
    name: "Cancer Tests",
    children: [
      { name: "CGP", fullName: "Comprehensive Genomic Profiling", children: [
        { name: "FoundationOne CDx", vendor: "Foundation Medicine", status: "FDA Approved" },
        { name: "FoundationOne Liquid CDx", vendor: "Foundation Medicine", status: "FDA Approved" },
        { name: "FoundationOne Heme", vendor: "Foundation Medicine", status: "CLIA LDT" },
        { name: "Guardant360 CDx", vendor: "Guardant Health", status: "FDA Approved" },
        { name: "Guardant360 Liquid", vendor: "Guardant Health", status: "CLIA LDT" },
        { name: "Tempus xT CDx", vendor: "Tempus AI", status: "FDA Approved" },
        { name: "Tempus xF", vendor: "Tempus AI", status: "CLIA LDT" },
        { name: "Tempus xF+", vendor: "Tempus AI", status: "CLIA LDT" },
        { name: "MSK-IMPACT", vendor: "Memorial Sloan Kettering", status: "FDA De Novo" },
        { name: "MI Cancer Seek", vendor: "Caris Life Sciences", status: "FDA Approved" },
        { name: "OncoExTra", vendor: "Exact Sciences", status: "CLIA LDT" },
        { name: "OmniSeq INSIGHT", vendor: "Labcorp Oncology", status: "CLIA LDT" },
        { name: "StrataNGS", vendor: "Strata Oncology", status: "CLIA LDT" },
        { name: "MI Profile", vendor: "Caris Life Sciences", status: "CLIA LDT" },
        { name: "NEO PanTracer Tissue", vendor: "NeoGenomics", status: "CLIA LDT" },
        { name: "Northstar Select", vendor: "BillionToOne", status: "CLIA LDT" },
        { name: "IsoPSA", vendor: "Cleveland Diagnostics", status: "FDA Approved" },
        { name: "Oncotype DX", vendor: "Exact Sciences", status: "CLIA LDT" },
        { name: "Liquid Trace Solid Tumor", vendor: "Genomic Testing Cooperative", status: "CLIA LDT" },
        { name: "Liquid Trace Hematology", vendor: "Genomic Testing Cooperative", status: "CLIA LDT" },
        { name: "LiquidHALLMARK", vendor: "Lucence", status: "CLIA LDT" },
        { name: "Resolution ctDx FIRST", vendor: "Agilent / Resolution Bioscience", status: "FDA Approved" },
        { name: "OncoCompass Target", vendor: "Burning Rock Dx", status: "CLIA LDT" },
        { name: "OncoScreen Focus CDx", vendor: "Burning Rock Dx", status: "CLIA LDT" },
        { name: "TruSight Oncology Comprehensive", vendor: "Illumina", status: "FDA Approved" },
        { name: "GeneseeqPrime", vendor: "Geneseeq Technology", status: "FDA Cleared" },
        { name: "PGDx elio tissue complete", vendor: "Labcorp/PGDx", status: "FDA Cleared" },
        { name: "Oncomine Comprehensive Assay Plus", vendor: "Thermo Fisher Scientific", status: "RUO" },
        { name: "TSO 500", vendor: "Illumina", status: "RUO" },
        { name: "cobas EGFR Mutation Test v2", vendor: "Roche", status: "FDA Approved" },
        { name: "cobas KRAS Mutation Test", vendor: "Roche", status: "FDA Approved" },
        { name: "therascreen EGFR RGQ PCR Kit", vendor: "QIAGEN", status: "FDA Approved" },
        { name: "OncoBEAM RAS CRC Kit", vendor: "Sysmex Inostics", status: "CE-IVD" },
        { name: "PGDx elio plasma focus Dx", vendor: "Labcorp/PGDx", status: "FDA De Novo" },
        { name: "LeukoStrat CDx FLT3", vendor: "Invivoscribe", status: "FDA Approved" },
        { name: "Hedera Profiling 2 ctDNA", vendor: "Hedera Dx", status: "CE-IVD" },
        { name: "OncoScreen Focus CDx Tissue Kit", vendor: "Burning Rock Dx", status: "CE-IVD" },
        { name: "OncoScreen Plus Tissue Kit", vendor: "Burning Rock Dx", status: "CLIA LDT" },
        { name: "MSK-IMPACT SOPHiA", vendor: "SOPHiA GENETICS", status: "RUO" },
        { name: "MSK-ACCESS SOPHiA", vendor: "SOPHiA GENETICS", status: "RUO" },
        { name: "MSK-IMPACT Flex SOPHiA", vendor: "SOPHiA GENETICS", status: "RUO" },
        { name: "CellSight DNA", vendor: "Cancer Cell Dx", status: "CLIA LDT" },
        { name: "CancerVision", vendor: "Inocras", status: "CLIA LDT" },
        { name: "K-4CARE", vendor: "Gene Solutions", status: "CLIA LDT" },
        { name: "Decipher Prostate", vendor: "Veracyte", status: "CLIA LDT" },
        { name: "PGDx elio plasma focus Dx (2)", vendor: "Labcorp/PGDx", status: "FDA De Novo" },
      ]},
      { name: "MRD", fullName: "Molecular Residual Disease", children: [
        { name: "Haystack MRD", vendor: "Quest Diagnostics", status: "Breakthrough" },
        { name: "NeXT Personal Dx", vendor: "Personalis", status: "CLIA LDT" },
        { name: "Oncodetect", vendor: "Exact Sciences", status: "CLIA LDT" },
        { name: "Pathlight", vendor: "SAGA Diagnostics", status: "CLIA LDT" },
        { name: "RaDaR ST", vendor: "NeoGenomics", status: "CLIA LDT" },
        { name: "Reveal MRD", vendor: "Guardant Health", status: "CLIA LDT" },
        { name: "Signatera", vendor: "Natera", status: "PMA Submitted" },
        { name: "Tempus xM MRD", vendor: "Tempus", status: "CLIA LDT" },
        { name: "Labcorp Plasma Detect", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "FoundationOne Tracker", vendor: "Foundation Medicine", status: "Breakthrough" },
        { name: "Foundation TI-WGS MRD", vendor: "Foundation Medicine", status: "RUO" },
        { name: "Veracyte MRD (C2i)", vendor: "Veracyte", status: "RUO" },
        { name: "Guardant LUNAR", vendor: "Guardant Health", status: "RUO" },
        { name: "NavDx", vendor: "Naveris", status: "CLIA LDT" },
        { name: "Foresight CLARITY", vendor: "Natera", status: "CLIA LDT" },
        { name: "Invitae PCM", vendor: "Labcorp (Invitae)", status: "Breakthrough" },
        { name: "Caris Assure", vendor: "Caris Life Sciences", status: "CLIA LDT" },
        { name: "clonoSEQ", vendor: "Adaptive Biotechnologies", status: "FDA Cleared" },
        { name: "Signatera Genome", vendor: "Natera", status: "CLIA LDT" },
        { name: "Latitude", vendor: "Natera", status: "CLIA LDT" },
        { name: "CancerDetect", vendor: "IMBdx", status: "CLIA LDT" },
        { name: "LymphoVista", vendor: "LIQOMICS", status: "CLIA LDT" },
        { name: "CancerVista", vendor: "LIQOMICS", status: "CLIA LDT" },
        { name: "CanCatch Custom", vendor: "Burning Rock Dx", status: "CLIA LDT" },
        { name: "Guardant360 Response", vendor: "Guardant Health", status: "CLIA LDT" },
        { name: "Signatera IO Monitoring", vendor: "Natera", status: "CLIA LDT" },
        { name: "NeXT Personal", vendor: "Personalis", status: "CLIA LDT" },
        { name: "Tempus xM for TRM", vendor: "Tempus", status: "RUO" },
        { name: "RaDaR", vendor: "NeoGenomics", status: "CLIA LDT" },
        { name: "FoundationOne Tracker (TRM)", vendor: "Foundation Medicine", status: "CLIA LDT" },
        { name: "FoundationOne Monitor", vendor: "Foundation Medicine", status: "CLIA LDT" },
        { name: "Northstar Response", vendor: "BillionToOne", status: "CLIA LDT" },
        { name: "Caris Assure (TRM)", vendor: "Caris Life Sciences", status: "CLIA LDT" },
        { name: "Reveal TRM", vendor: "Guardant Health", status: "CLIA LDT" },
        { name: "Liquid Trace Monitoring", vendor: "Genomic Testing Cooperative", status: "CLIA LDT" },
        { name: "xM for TRM", vendor: "Tempus AI", status: "RUO" },
        { name: "clonoSEQ Assay (Kit)", vendor: "Adaptive Biotechnologies", status: "FDA Cleared" },
        { name: "LymphoTrack Dx IGH", vendor: "Invivoscribe", status: "CE-IVD" },
        { name: "BD OneFlow B-ALL MRD", vendor: "BD Biosciences", status: "CE-IVD" },
        { name: "MRDVision", vendor: "Inocras", status: "CLIA LDT" },
        { name: "Bladder EpiCheck", vendor: "Nucleix", status: "FDA Cleared" },
        { name: "K-4CARE (MRD)", vendor: "Gene Solutions", status: "CLIA LDT" },
        { name: "EasyM", vendor: "Rapid Novor", status: "CLIA LDT" },
        { name: "Quest Flow Cytometry MRD", vendor: "Quest Diagnostics", status: "CLIA LDT" },
      ]},
      { name: "HCT", fullName: "Hereditary Cancer Testing", children: [
        { name: "MyRisk", vendor: "Myriad Genetics", status: "CLIA LDT" },
        { name: "Invitae Multi-Cancer Panel", vendor: "Invitae (Labcorp)", status: "CLIA LDT" },
        { name: "CancerNext-Expanded", vendor: "Ambry Genetics", status: "CLIA LDT" },
        { name: "Empower Hereditary Cancer", vendor: "Natera", status: "CLIA LDT" },
        { name: "Color Hereditary Cancer", vendor: "Color Health", status: "CLIA LDT" },
        { name: "Comprehensive Hereditary Cancer", vendor: "Quest Diagnostics", status: "CLIA LDT" },
        { name: "VistaSeq Hereditary Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "xG / xG+ Hereditary Cancer", vendor: "Tempus", status: "CLIA LDT" },
        { name: "Comprehensive Common Cancer", vendor: "GeneDx", status: "CLIA LDT" },
        { name: "Full Comprehensive Cancer", vendor: "Fulgent Genetics", status: "CLIA LDT" },
        { name: "Invitae Hereditary Breast", vendor: "Invitae", status: "CLIA LDT" },
        { name: "Invitae Hereditary Breast Guidelines", vendor: "Invitae", status: "CLIA LDT" },
        { name: "Invitae BRCA1/2 STAT", vendor: "Invitae", status: "CLIA LDT" },
        { name: "Invitae Hereditary Thyroid", vendor: "Invitae", status: "CLIA LDT" },
        { name: "Invitae Hereditary Pancreatic", vendor: "Invitae", status: "CLIA LDT" },
        { name: "VistaSeq Breast Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "VistaSeq Colorectal Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "VistaSeq Ovarian Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "VistaSeq Endometrial Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "VistaSeq Gastric Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "VistaSeq Pancreatic Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "VistaSeq Prostate Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "VistaSeq Melanoma Cancer", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "BRCAssure: BRCA1", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "BRCAssure: BRCA2", vendor: "Labcorp", status: "CLIA LDT" },
        { name: "Full Comprehensive (Pan-Cancer)", vendor: "Fulgent Genetics", status: "CLIA LDT" },
        { name: "Gastric Cancer Panel", vendor: "PreventionGenetics", status: "CLIA LDT" },
        { name: "PTEN Hamartoma Syndrome", vendor: "PreventionGenetics", status: "CLIA LDT" },
        { name: "FoundationOne Germline", vendor: "Foundation Medicine", status: "CLIA LDT" },
        { name: "FoundationOne Germline More", vendor: "Foundation Medicine", status: "CLIA LDT" },
        { name: "BRACAnalysis CDx", vendor: "Myriad Genetics", status: "FDA Approved" },
        { name: "23andMe BRCA1/BRCA2", vendor: "23andMe", status: "FDA Authorized" },
        { name: "AbsoluteDx", vendor: "Allelica", status: "CLIA LDT" },
        { name: "Risk MAPS", vendor: "Protean BioDiagnostics", status: "CLIA LDT" },
      ]},
      { name: "ECD", fullName: "Early Cancer Detection", children: [
        { name: "Shield", vendor: "Guardant Health", status: "FDA Approved" },
        { name: "Galleri", vendor: "GRAIL", status: "PMA Filed" },
        { name: "Cologuard Plus", vendor: "Exact Sciences", status: "FDA Approved" },
        { name: "ColoSense", vendor: "Geneoscopy", status: "FDA Approved" },
        { name: "Freenome CRC Blood Test", vendor: "Freenome", status: "PMA Filed" },
        { name: "FirstLook Lung", vendor: "DELFI Diagnostics", status: "CLIA LDT" },
        { name: "HelioLiver", vendor: "Helio Genomics", status: "PMA Filed" },
        { name: "Oncoguard Liver", vendor: "Exact Sciences", status: "Breakthrough" },
        { name: "Shield MCD", vendor: "Guardant Health", status: "Breakthrough" },
        { name: "EPISEEK", vendor: "Precision Epigenomics", status: "CLIA LDT" },
        { name: "ProVue Lung", vendor: "PrognomiQ", status: "CLIA LDT" },
        { name: "Signal-C", vendor: "Universal DX", status: "CLIA LDT" },
        { name: "IColocomf", vendor: "Wuhan Ammunition Life Tech", status: "CE-IVD" },
        { name: "GALEAS Bladder", vendor: "Nonacus", status: "CE-IVD" },
        { name: "OverC MCED", vendor: "Burning Rock Dx", status: "Breakthrough" },
        { name: "Cancerguard", vendor: "Exact Sciences", status: "CLIA LDT" },
        { name: "Freenome CRC (dev)", vendor: "Exact Sciences", status: "CLIA LDT" },
        { name: "Cologuard", vendor: "Exact Sciences", status: "FDA Approved" },
        { name: "Epi proColon", vendor: "Epigenomics", status: "FDA Approved" },
        { name: "IColohunter", vendor: "Wuhan Ammunition Life Tech", status: "CE-IVD" },
        { name: "IEsohunter", vendor: "Wuhan Ammunition Life Tech", status: "CE-IVD" },
        { name: "IHepcomf", vendor: "Wuhan Ammunition Life Tech", status: "CE-IVD" },
        { name: "IUrisure", vendor: "Wuhan Ammunition Life Tech", status: "CE-IVD" },
        { name: "Avantect Pancreatic", vendor: "ClearNote Health", status: "CLIA LDT" },
        { name: "CancerDetect Oral", vendor: "Viome", status: "Breakthrough" },
        { name: "Trucheck Intelli", vendor: "Datar Cancer Genetics", status: "CE-IVD" },
        { name: "OncoXPLORE+", vendor: "OncoDNA", status: "RUO" },
        { name: "MethylScan HCC", vendor: "EarlyDx", status: "CLIA LDT" },
        { name: "SPOT-MAS", vendor: "Gene Solutions", status: "CLIA LDT" },
        { name: "ClarityDX Prostate", vendor: "Protean BioDiagnostics", status: "CLIA LDT" },
        { name: "OnkoSkan", vendor: "Protean BioDiagnostics", status: "CLIA LDT" },
      ]},
    ]
  };

  // ── Timeline data ──
  const timelineData = [
    { year: 2014, MRD: 0, ECD: 1, CGP: 1, HCT: 0 },
    { year: 2016, MRD: 0, ECD: 1, CGP: 0, HCT: 0 },
    { year: 2017, MRD: 0, ECD: 0, CGP: 1, HCT: 0 },
    { year: 2018, MRD: 2, ECD: 0, CGP: 0, HCT: 0 },
    { year: 2019, MRD: 0, ECD: 1, CGP: 0, HCT: 0 },
    { year: 2020, MRD: 0, ECD: 1, CGP: 1, HCT: 0 },
    { year: 2021, MRD: 2, ECD: 2, CGP: 1, HCT: 0 },
    { year: 2022, MRD: 3, ECD: 0, CGP: 3, HCT: 0 },
    { year: 2023, MRD: 2, ECD: 2, CGP: 2, HCT: 0 },
    { year: 2024, MRD: 0, ECD: 3, CGP: 5, HCT: 0 },
    { year: 2025, MRD: 6, ECD: 4, CGP: 3, HCT: 0 },
    { year: 2026, MRD: 3, ECD: 1, CGP: 0, HCT: 0 },
  ];

  // ── Cancer info lookup (from OpenOnco) ──
  const cancerInfo = {
    "FoundationOne CDx": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "FoundationOne Liquid CDx": {n:1, t:["Pan-cancer"], s:"Blood"},
    "FoundationOne Heme": {n:2, t:["Heme","Sarcoma"], s:"Tissue/blood/marrow"},
    "Guardant360 CDx": {n:1, t:["Pan-cancer"], s:"Blood"},
    "Guardant360 Liquid": {n:1, t:["Pan-cancer"], s:"Blood", sp:99.9},
    "Tempus xT CDx": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "Tempus xF": {n:1, t:["Pan-cancer"], s:"Blood", se:99, sp:99.9},
    "Tempus xF+": {n:1, t:["Pan-cancer"], s:"Blood"},
    "MSK-IMPACT": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "MI Cancer Seek": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "OncoExTra": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "OmniSeq INSIGHT": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "StrataNGS": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "MI Profile": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "NEO PanTracer Tissue": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "Northstar Select": {n:17, t:["Pan-cancer"], s:"Blood", sp:99.9},
    "IsoPSA": {n:1, t:["Prostate"], s:"Blood", se:90.2, sp:45.5},
    "Oncotype DX": {n:1, t:["Breast"], s:"Tissue"},
    "Liquid Trace Solid Tumor": {n:10, t:["Lung","Brain","Breast","Thyroid","CRC","Oropharyngeal","Pancreas","Ovarian","Prostate","+1 more"], s:"Blood"},
    "Liquid Trace Hematology": {n:10, t:["MM","Lymphoma","ALL","AML","MDS","CMML","MPN","VEXAS","+2 more"], s:"Blood"},
    "LiquidHALLMARK": {n:15, t:["Lung","Breast","CRC","Prostate","Ovarian","Gastric","Liver","Pancreas"], s:"Blood", se:99.4, sp:99},
    "Resolution ctDx FIRST": {n:1, t:["NSCLC"], s:"Blood", se:66.2, sp:100},
    "OncoCompass Target": {n:2, t:["NSCLC","Pan-cancer"], s:"Blood", se:96, sp:99.9},
    "OncoScreen Focus CDx": {n:3, t:["NSCLC","CRC","GIST"], s:"Tissue"},
    "TruSight Oncology Comprehensive": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "GeneseeqPrime": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "PGDx elio tissue complete": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "Oncomine Comprehensive Assay Plus": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "TSO 500": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "cobas EGFR Mutation Test v2": {n:1, t:["NSCLC"], s:"Blood", se:85, sp:98},
    "cobas KRAS Mutation Test": {n:1, t:["CRC"], s:"Blood", se:95, sp:99},
    "therascreen EGFR RGQ PCR Kit": {n:1, t:["NSCLC"], s:"Blood", se:80, sp:98},
    "OncoBEAM RAS CRC Kit": {n:1, t:["CRC"], s:"Blood", se:95, sp:99},
    "PGDx elio plasma focus Dx": {n:1, t:["Pan-cancer"], s:"Blood", se:96},
    "LeukoStrat CDx FLT3": {n:1, t:["AML"], s:"Bone marrow/blood"},
    "Hedera Profiling 2 ctDNA": {n:4, t:["Pan-cancer","NSCLC","Breast","CRC"], s:"Blood", se:97, sp:99.7},
    "OncoScreen Focus CDx Tissue Kit": {n:3, t:["NSCLC","CRC","GIST"], s:"Tissue", se:100, sp:100},
    "OncoScreen Plus Tissue Kit": {n:1, t:["Pan-cancer"], s:"Tissue", se:97, sp:99},
    "MSK-IMPACT SOPHiA": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "MSK-ACCESS SOPHiA": {n:1, t:["Pan-cancer"], s:"Blood"},
    "MSK-IMPACT Flex SOPHiA": {n:1, t:["Pan-cancer"], s:"Tissue"},
    "CellSight DNA": {n:1, t:["Pan-cancer"], s:"Blood", se:99},
    "CancerVision": {n:1, t:["Pan-cancer"], s:"Tissue", se:100, sp:99},
    "K-4CARE": {n:1, t:["Pan-cancer"], s:"Tissue", se:99, sp:99},
    "Decipher Prostate": {n:1, t:["Prostate"], s:"Tissue"},
    "PGDx elio plasma focus Dx (2)": {n:1, t:["Pan-cancer"], s:"Blood", se:96},
    "Haystack MRD": {n:1, t:["Multi-solid"], s:"Blood, 30 mL", se:95, sp:100},
    "NeXT Personal Dx": {n:4, t:["Breast","CRC","NSCLC","Cervical"], s:"Blood, 20 mL", se:100, sp:99.9},
    "Oncodetect": {n:1, t:["Multi-solid"], s:"Blood", se:91, sp:94},
    "Pathlight": {n:2, t:["Breast","Multi-solid"], s:"Blood", se:100, sp:100},
    "RaDaR ST": {n:3, t:["Breast","H&N","Multi-solid"], s:"Blood", se:95.7, sp:91},
    "Reveal MRD": {n:3, t:["CRC","Breast","NSCLC"], s:"Blood, 20 mL", se:81, sp:98},
    "Signatera": {n:6, t:["CRC","Breast","Bladder","NSCLC","Ovarian","Pan-solid"], s:"Blood, 20 mL", se:94, sp:98},
    "Tempus xM MRD": {n:1, t:["CRC"], s:"Blood, 17 mL", se:61.1, sp:94},
    "Labcorp Plasma Detect": {n:4, t:["CRC","Lung","Bladder","Multi-solid"], s:"Blood, 20 mL", se:95, sp:99.4},
    "FoundationOne Tracker": {n:1, t:["Multi-solid"], s:"Blood", se:100, sp:99.6},
    "Foundation TI-WGS MRD": {n:1, t:["Multi-solid"], s:"Blood", se:90, sp:100},
    "Veracyte MRD (C2i)": {n:2, t:["Bladder","Multi-solid"], s:"Blood, 4 mL", se:91, sp:92},
    "Guardant LUNAR": {n:2, t:["CRC","Multi-solid"], s:"Blood, 4 mL", se:56, sp:100},
    "NavDx": {n:3, t:["HPV+ H&N","Anal SCC","HPV gynecologic"], s:"Blood, 10 mL", se:90.4, sp:98.6},
    "Foresight CLARITY": {n:5, t:["DLBCL","LBCL","Follicular","Hodgkin","MM"], s:"Blood", se:90.6, sp:97.7},
    "Invitae PCM": {n:6, t:["NSCLC","Breast","CRC","Pancreas","H&N","Multi-solid"], s:"Blood, 20 mL", se:76.9, sp:100},
    "Caris Assure": {n:1, t:["Pan-cancer"], s:"Blood, 20 mL", se:93.8, sp:100},
    "clonoSEQ": {n:6, t:["MM","B-ALL","CLL","DLBCL","Mantle cell","Other lymphoid"], s:"Bone marrow/blood", se:95, sp:99},
    "Signatera Genome": {n:6, t:["Breast","NSCLC","Melanoma","Kidney","CRC","Multi-solid"], s:"Blood", se:94, sp:100},
    "Latitude": {n:1, t:["CRC"], s:"Blood", se:81, sp:97},
    "CancerDetect": {n:3, t:["CRC","Breast","Gastric"], s:"Blood, 20 mL", se:61.9, sp:99.9},
    "LymphoVista": {n:7, t:["DLBCL","Follicular","Mantle cell","Marginal zone","Burkitt","CNS","Hodgkin"], s:"Blood, 20 mL", se:100, sp:93},
    "CancerVista": {n:17, t:["Bladder","Brain","Breast","Cervical","CRC","Esophageal","H&N","Kidney","Liver","Lung","Ovarian","Pancreas","Prostate","Skin","Gastric","Thyroid","Uterine"], s:"Blood, 20 mL", se:80, sp:96.7},
    "CanCatch Custom": {n:4, t:["NSCLC","CRC","Esophageal","GIST"], s:"Blood, 16 mL", se:98.7, sp:99},
    "Guardant360 Response": {n:4, t:["NSCLC","Bladder","Breast","GI"], s:"Blood"},
    "Signatera IO Monitoring": {n:1, t:["Pan-solid"], s:"Blood"},
    "NeXT Personal": {n:5, t:["Breast","CRC","NSCLC","Melanoma","Kidney"], s:"Blood"},
    "Tempus xM for TRM": {n:1, t:["Pan-cancer"], s:"Blood"},
    "RaDaR": {n:5, t:["Breast","Melanoma","CRC","H&N","Lung"], s:"Blood"},
    "FoundationOne Tracker (TRM)": {n:1, t:["Pan-solid"], s:"Blood"},
    "FoundationOne Monitor": {n:1, t:["Pan-cancer"], s:"Blood"},
    "Northstar Response": {n:3, t:["CRC","Pancreas","GI"], s:"Blood"},
    "Caris Assure (TRM)": {n:1, t:["Pan-cancer"], s:"Blood, 20 mL", se:93.8, sp:100},
    "Reveal TRM": {n:1, t:["Pan-cancer"], s:"Blood, 20 mL"},
    "Liquid Trace Monitoring": {n:1, t:["Pan-cancer"], s:"Blood, 10 mL"},
    "xM for TRM": {n:1, t:["Pan-cancer"], s:"Blood"},
    "clonoSEQ Assay (Kit)": {n:4, t:["MM","B-ALL","CLL","Mantle cell"], s:"Bone marrow/blood", se:95, sp:99},
    "LymphoTrack Dx IGH": {n:3, t:["B-cell lymphoma","CLL","B-ALL"], s:"Bone marrow/blood", se:98, sp:99},
    "BD OneFlow B-ALL MRD": {n:1, t:["B-ALL"], s:"Bone marrow", se:96, sp:95},
    "MRDVision": {n:1, t:["Multi-solid"], s:"Blood", se:94, sp:99},
    "Bladder EpiCheck": {n:1, t:["Bladder"], s:"Urine, 10 mL", se:67, sp:84},
    "K-4CARE (MRD)": {n:1, t:["Pan-cancer"], s:"Tissue", se:99, sp:99},
    "EasyM": {n:1, t:["MM"], s:"Blood, 0.1 mL", sp:100},
    "Quest Flow Cytometry MRD": {n:1, t:["MM"], s:"Blood"},
    "MyRisk": {n:8, t:["Breast","Ovarian","CRC","Endometrial","Pancreas","Prostate","Gastric","Melanoma"], s:"Blood/saliva", se:99.9, sp:99.9},
    "Invitae Multi-Cancer Panel": {n:12, t:["Breast","Ovarian","Uterine","CRC","Gastric","Pancreas","Prostate","Melanoma","Thyroid","Kidney","Brain","Sarcoma"], s:"Blood/saliva", se:99.5, sp:99.9},
    "CancerNext-Expanded": {n:9, t:["Breast","Ovarian","CRC","Uterine","Pancreas","Prostate","Kidney","Thyroid","Paraganglioma"], s:"Blood/saliva", se:99, sp:99.9},
    "Empower Hereditary Cancer": {n:8, t:["Breast","Ovarian","Uterine","CRC","Gastric","Pancreas","Prostate","Melanoma"], s:"Blood/saliva", se:99, sp:99.9},
    "Color Hereditary Cancer": {n:8, t:["Breast","Ovarian","Uterine","CRC","Melanoma","Pancreas","Prostate","Gastric"], s:"Saliva", se:99, sp:99},
    "Comprehensive Hereditary Cancer": {n:7, t:["Breast","Ovarian","CRC","Endometrial","Pancreas","Prostate","Gastric"], s:"Blood", se:99, sp:99},
    "VistaSeq Hereditary Cancer": {n:5, t:["Breast","Ovarian","CRC","Pancreas","Prostate"], s:"Blood/saliva", se:99, sp:99},
    "xG / xG+ Hereditary Cancer": {n:6, t:["Breast","Ovarian","CRC","Pancreas","Prostate","Endometrial"], s:"Blood/saliva", se:99, sp:99.9},
    "Comprehensive Common Cancer": {n:7, t:["Breast","Ovarian","CRC","Pancreas","Prostate","Uterine","Melanoma"], s:"Blood/saliva", se:99, sp:99.9},
    "Full Comprehensive Cancer": {n:12, t:["Breast","Ovarian","CRC","Endometrial","Pancreas","Prostate","Gastric","Kidney","Thyroid","Brain","Sarcoma","Heme"], s:"Blood/saliva", se:99.9, sp:99.9},
    "Invitae Hereditary Breast": {n:0, s:"Blood/saliva"},
    "Invitae Hereditary Breast Guidelines": {n:0, s:"Blood/saliva"},
    "Invitae BRCA1/2 STAT": {n:0, s:"Blood/saliva"},
    "Invitae Hereditary Thyroid": {n:1, t:["Thyroid"], s:"Blood/saliva"},
    "Invitae Hereditary Pancreatic": {n:1, t:["Pancreas"], s:"Blood/saliva"},
    "VistaSeq Breast Cancer": {n:1, t:["Breast"], s:"Blood/saliva"},
    "VistaSeq Colorectal Cancer": {n:1, t:["CRC"], s:"Blood/saliva"},
    "VistaSeq Ovarian Cancer": {n:1, t:["Ovarian"], s:"Blood/saliva"},
    "VistaSeq Endometrial Cancer": {n:1, t:["Endometrial"], s:"Blood/saliva"},
    "VistaSeq Gastric Cancer": {n:1, t:["Gastric"], s:"Blood/saliva"},
    "VistaSeq Pancreatic Cancer": {n:1, t:["Pancreas"], s:"Blood/saliva"},
    "VistaSeq Prostate Cancer": {n:1, t:["Prostate"], s:"Blood/saliva"},
    "VistaSeq Melanoma Cancer": {n:1, t:["Melanoma"], s:"Blood/saliva"},
    "BRCAssure: BRCA1": {n:0, s:"Blood/saliva"},
    "BRCAssure: BRCA2": {n:0, s:"Blood/saliva"},
    "Full Comprehensive (Pan-Cancer)": {n:0, s:"Blood"},
    "Gastric Cancer Panel": {n:1, t:["Gastric"], s:"Blood/saliva"},
    "PTEN Hamartoma Syndrome": {n:0, s:"Blood/saliva"},
    "FoundationOne Germline": {n:0, s:"Blood/saliva"},
    "FoundationOne Germline More": {n:0, s:"Blood/saliva"},
    "BRACAnalysis CDx": {n:0, s:"Blood/saliva"},
    "23andMe BRCA1/BRCA2": {n:0, s:"Saliva"},
    "AbsoluteDx": {n:5, t:["Breast","Prostate","CRC","Ovarian","Pancreas"], s:"Saliva"},
    "Risk MAPS": {n:0, s:"Saliva"},
    "Shield": {n:1, t:["CRC"], s:"Blood, 40 mL", se:83.1, sp:89.6},
    "Galleri": {n:50, t:["Multi-cancer"], s:"Blood, 20 mL", se:51.5, sp:99.5},
    "Cologuard Plus": {n:1, t:["CRC"], s:"Stool", se:93.9, sp:91},
    "ColoSense": {n:1, t:["CRC"], s:"Stool", se:93, sp:88},
    "Freenome CRC Blood Test": {n:1, t:["CRC"], s:"Blood", se:79.2, sp:91.5},
    "FirstLook Lung": {n:1, t:["Lung"], s:"Blood, 1 mL", se:80, sp:58},
    "HelioLiver": {n:1, t:["HCC"], s:"Blood, 10 mL", se:47.8, sp:88},
    "Oncoguard Liver": {n:1, t:["HCC"], s:"Blood, 10 mL", se:70, sp:81.9},
    "Shield MCD": {n:10, t:["Bladder","CRC","Esophageal","Gastric","Liver","Lung","Ovarian","Pancreas","Breast","Prostate"], s:"Blood, 40 mL", se:60, sp:98.5},
    "EPISEEK": {n:60, t:["Multi-cancer"], s:"Blood, 20 mL", se:54, sp:99.5},
    "ProVue Lung": {n:1, t:["Lung"], s:"Blood, 10 mL", se:85, sp:55},
    "Signal-C": {n:1, t:["CRC"], s:"Blood", se:93, sp:92},
    "IColocomf": {n:1, t:["CRC"], s:"Stool", se:95.3, sp:96.7},
    "GALEAS Bladder": {n:1, t:["Bladder"], s:"Urine, 50 mL", se:90, sp:85},
    "OverC MCED": {n:6, t:["CRC","Esophageal","Liver","Lung","Ovarian","Pancreas"], s:"Blood", se:69.1, sp:98.9},
    "Cancerguard": {n:50, t:["Multi-cancer"], s:"Blood, 10 mL", se:64, sp:97.4},
    "Freenome CRC (dev)": {n:1, t:["CRC"], s:"Blood"},
    "Cologuard": {n:1, t:["CRC"], s:"Stool", se:92, sp:87},
    "Epi proColon": {n:1, t:["CRC"], s:"Blood", se:68, sp:80},
    "IColohunter": {n:1, t:["CRC"], s:"Blood", se:91.2, sp:92.4},
    "IEsohunter": {n:1, t:["Esophageal"], s:"Blood, 10 mL", se:87.4, sp:93.3},
    "IHepcomf": {n:1, t:["Liver"], s:"Blood, 10 mL", se:92.3, sp:93.4},
    "IUrisure": {n:1, t:["Urothelial"], s:"Urine, 1.8 mL", se:93.9, sp:92},
    "Avantect Pancreatic": {n:1, t:["Pancreas"], s:"Blood", se:68.3, sp:96.9},
    "CancerDetect Oral": {n:2, t:["Oral SCC","Oropharyngeal"], s:"Saliva", se:90, sp:95},
    "Trucheck Intelli": {n:9, t:["Adeno","Adenosquamous","CNS","GI","Melanoma","Mesothelioma","NET","SCC","Transitional"], s:"Blood, 10 mL", se:88.2, sp:96.3},
    "OncoXPLORE+": {n:2, t:["Multi-solid","Heme"], s:"Blood", se:53.5, sp:99},
    "MethylScan HCC": {n:1, t:["HCC"], s:"Blood, 10 mL", se:84.5, sp:92},
    "SPOT-MAS": {n:10, t:["Breast","CRC","Gastric","Lung","Liver","Ovarian","Pancreas","Esophageal","H&N","Endometrial"], s:"Blood, 10 mL", se:70.8, sp:99.7},
    "ClarityDX Prostate": {n:1, t:["Prostate"], s:"Blood", se:95, sp:35},
    "OnkoSkan": {n:10, t:["CRC","Pancreas","Liver","Lung","Gastric","Kidney","Breast","Ovarian","NET","GBM"], s:"Blood"},
  };

  // ── Helpers ──
  function statusOpacity(status) {
    if (!status) return 0.5;
    const s = status.toLowerCase();
    if (s.includes('fda approved') || s.includes('fda cleared') || s.includes('fda authorized') || s.includes('de novo')) return 1.0;
    if (s === 'clia ldt') return 0.5;
    return 0.7;
  }

  // ── Treemap ──
  function renderTreemap() {
    const container = document.getElementById('treemap');
    if (!container) return;
    container.innerHTML = '';
    const c = CancerCharts.getColors();

    const containerWidth = container.clientWidth || 800;
    const width = containerWidth;
    const height = Math.max(400, width * 0.55);

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('height', 'auto')
      .style('font-family', 'inherit');

    const root = d3.hierarchy(treemapData)
      .sum(d => d.children ? 0 : 1)
      .sort((a, b) => b.value - a.value);

    d3.treemap()
      .size([width, height])
      .padding(2)
      .paddingTop(22)
      .round(true)(root);

    // Category groups
    const categories = svg.selectAll('g.category')
      .data(root.children)
      .join('g')
      .attr('class', 'category');

    // Category background
    categories.append('rect')
      .attr('x', d => d.x0)
      .attr('y', d => d.y0)
      .attr('width', d => d.x1 - d.x0)
      .attr('height', d => d.y1 - d.y0)
      .attr('fill', 'none')
      .attr('stroke', d => c[d.data.name] || c.muted)
      .attr('stroke-width', 1.5)
      .attr('rx', 3);

    // Category label
    categories.append('text')
      .attr('x', d => d.x0 + 6)
      .attr('y', d => d.y0 + 15)
      .text(d => `${d.data.name} (${d.data.children.length})`)
      .attr('fill', d => c[d.data.name] || c.text)
      .attr('font-size', '12px')
      .attr('font-weight', '600');

    // Tiles
    const leaves = svg.selectAll('g.leaf')
      .data(root.leaves())
      .join('g')
      .attr('class', 'leaf');

    leaves.append('rect')
      .attr('x', d => d.x0)
      .attr('y', d => d.y0)
      .attr('width', d => d.x1 - d.x0)
      .attr('height', d => d.y1 - d.y0)
      .attr('fill', d => c[d.parent.data.name] || c.muted)
      .attr('opacity', d => statusOpacity(d.data.status))
      .attr('rx', 2)
      .attr('stroke', c.bg)
      .attr('stroke-width', 0.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke', c.text).attr('stroke-width', 1.5);
        const info = cancerInfo[d.data.name];
        let html = `<strong>${d.data.name}</strong><br/>${d.data.vendor}<br/>` +
          `<span style="opacity:0.7">${d.data.status}</span>`;
        if (info) {
          if (info.s) html += `<br/><span style="opacity:0.7">Sample:</span> ${info.s}`;
          if (info.t && info.t.length) html += `<br/><span style="opacity:0.7">Cancers (${info.n}):</span> ${info.t.join(', ')}`;
          if (info.se != null || info.sp != null) {
            const parts = [];
            if (info.se != null) parts.push(`Sens ${info.se}%`);
            if (info.sp != null) parts.push(`Spec ${info.sp}%`);
            html += `<br/>${parts.join(' · ')}`;
          }
        }
        CancerCharts.showTooltip(event, html);
      })
      .on('mousemove', function(event) {
        CancerCharts.moveTooltip(event);
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke', c.bg).attr('stroke-width', 0.5);
        CancerCharts.hideTooltip();
      });

    // Cancer count number (centered in tile)
    leaves.append('text')
      .attr('x', d => (d.x0 + d.x1) / 2)
      .attr('y', d => (d.y0 + d.y1) / 2 + 1)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .text(d => {
        const info = cancerInfo[d.data.name];
        const w = d.x1 - d.x0, h = d.y1 - d.y0;
        if (!info || info.n === 0 || w < 18 || h < 14) return '';
        return info.n;
      })
      .attr('fill', c.bg)
      .attr('font-size', d => {
        const w = d.x1 - d.x0, h = d.y1 - d.y0;
        const sz = Math.min(w, h);
        if (sz > 30) return '12px';
        if (sz > 22) return '10px';
        return '8px';
      })
      .attr('font-weight', '700')
      .attr('pointer-events', 'none');

    // Tile name labels (below center, only if space)
    leaves.append('text')
      .attr('x', d => d.x0 + 3)
      .attr('y', d => d.y0 + 12)
      .text(d => (d.x1 - d.x0) > 60 && (d.y1 - d.y0) > 28 ? d.data.name : '')
      .attr('fill', c.bg)
      .attr('font-size', '8px')
      .attr('font-weight', '400')
      .attr('opacity', 0.8)
      .attr('pointer-events', 'none')
      .each(function(d) {
        const textWidth = this.getComputedTextLength();
        const tileWidth = d.x1 - d.x0 - 6;
        if (textWidth > tileWidth) {
          d3.select(this).text('');
        }
      });
  }

  // ── Timeline stacked area ──
  function renderTimeline() {
    const container = document.getElementById('timeline');
    if (!container) return;
    container.innerHTML = '';
    const c = CancerCharts.getColors();

    const margin = { top: 30, right: 30, bottom: 40, left: 50 };
    const containerWidth = container.clientWidth || 800;
    const width = containerWidth;
    const height = 380;
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('height', 'auto')
      .style('font-family', 'inherit');

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const keys = ["HCT", "CGP", "ECD", "MRD"];

    const stack = d3.stack().keys(keys);
    const series = stack(timelineData);

    const x = d3.scaleLinear()
      .domain([2014, 2026])
      .range([0, innerW]);

    const yMax = d3.max(series, s => d3.max(s, d => d[1]));
    const y = d3.scaleLinear()
      .domain([0, yMax + 1])
      .range([innerH, 0]);

    // Grid lines
    g.selectAll('line.grid')
      .data(y.ticks(5))
      .join('line')
      .attr('class', 'grid')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', d => y(d))
      .attr('y2', d => y(d))
      .attr('stroke', c.grid)
      .attr('stroke-dasharray', '3,3');

    // Area generator
    const area = d3.area()
      .x(d => x(d.data.year))
      .y0(d => y(d[0]))
      .y1(d => y(d[1]))
      .curve(d3.curveMonotoneX);

    // Draw areas
    g.selectAll('path.area')
      .data(series)
      .join('path')
      .attr('class', 'area')
      .attr('d', area)
      .attr('fill', d => c[d.key])
      .attr('opacity', 0.7)
      .attr('stroke', d => c[d.key])
      .attr('stroke-width', 1.5);

    // X axis
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x)
        .tickValues([2014, 2016, 2018, 2020, 2022, 2024, 2026])
        .tickFormat(d3.format('d'))
      );
    xAxis.selectAll('text').attr('fill', c.text).attr('font-size', '12px');
    xAxis.selectAll('line').attr('stroke', c.muted);
    xAxis.select('.domain').attr('stroke', c.muted);

    // Y axis
    const yAxis = g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('d')));
    yAxis.selectAll('text').attr('fill', c.text).attr('font-size', '12px');
    yAxis.selectAll('line').attr('stroke', c.muted);
    yAxis.select('.domain').attr('stroke', c.muted);

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('fill', c.muted)
      .attr('font-size', '12px')
      .text('Tests launched');

    // Annotations
    const annotations = [
      { text: "MRD explosion", year: 2025, yOffset: -15 },
      { text: "ECD FDA breakthroughs", year: 2024, yOffset: 25 },
    ];

    annotations.forEach(a => {
      const yearData = timelineData.find(d => d.year === a.year);
      if (!yearData) return;
      const total = keys.reduce((sum, k) => sum + yearData[k], 0);
      g.append('text')
        .attr('x', x(a.year))
        .attr('y', y(total) + a.yOffset)
        .attr('text-anchor', 'middle')
        .attr('fill', c.text)
        .attr('font-size', '11px')
        .attr('font-style', 'italic')
        .text(a.text);
    });

    // Legend
    const legendG = svg.append('g')
      .attr('transform', `translate(${margin.left + 10}, 8)`);

    keys.slice().reverse().forEach((key, i) => {
      const lg = legendG.append('g')
        .attr('transform', `translate(${i * 90}, 0)`);
      lg.append('rect')
        .attr('width', 12)
        .attr('height', 12)
        .attr('rx', 2)
        .attr('fill', c[key])
        .attr('opacity', 0.7);
      lg.append('text')
        .attr('x', 16)
        .attr('y', 10)
        .attr('fill', c.text)
        .attr('font-size', '11px')
        .text(key);
    });

    // Hover overlay
    const bisect = d3.bisector(d => d.year).left;
    const hoverLine = g.append('line')
      .attr('stroke', c.muted)
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,4')
      .attr('y1', 0)
      .attr('y2', innerH)
      .style('opacity', 0);

    const hoverRect = g.append('rect')
      .attr('width', innerW)
      .attr('height', innerH)
      .attr('fill', 'transparent')
      .style('cursor', 'crosshair');

    hoverRect.on('mousemove', function(event) {
      const [mx] = d3.pointer(event);
      const yearVal = x.invert(mx);
      const idx = bisect(timelineData, yearVal, 1);
      const d0 = timelineData[idx - 1];
      const d1 = timelineData[idx];
      const d = (!d1 || (yearVal - d0.year < d1.year - yearVal)) ? d0 : d1;
      if (!d) return;

      hoverLine
        .attr('x1', x(d.year))
        .attr('x2', x(d.year))
        .style('opacity', 1);

      const total = keys.reduce((sum, k) => sum + d[k], 0);
      const breakdown = keys
        .filter(k => d[k] > 0)
        .map(k => `<span style="color:${c[k]}">${k}: ${d[k]}</span>`)
        .join('<br/>');

      CancerCharts.showTooltip(event,
        `<strong>${d.year}</strong> (${total} tests)<br/>${breakdown}`
      );
    });

    hoverRect.on('mouseout', function() {
      hoverLine.style('opacity', 0);
      CancerCharts.hideTooltip();
    });
  }

  // ── Init and theme handling ──
  function renderAll() {
    renderTreemap();
    renderTimeline();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderAll);
  } else {
    renderAll();
  }

  document.addEventListener('themechange', renderAll);

  // Resize handling
  let resizeTimer;
  const ro = new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderAll, 150);
  });
  const treemapEl = document.getElementById('treemap');
  const timelineEl = document.getElementById('timeline');
  if (treemapEl) ro.observe(treemapEl);
  if (timelineEl) ro.observe(timelineEl);

})();
</script>
