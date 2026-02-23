---
layout: post
title: "Why a 99% Accurate Test Is Often Wrong"
date: 2026-02-19 10:00:00 -0700
tags: biotech data-analysis
author: bolu-atx
categories: biotech
---

Is 99% accuracy good enough? If you read [Part 3 of this series](/biotech/2026/02/18/early-cancer-detection-screening-wars.html), you saw that Galleri achieves 99.5% specificity, Shield hits 89.6%, and Cologuard Plus reaches 91%. These sound like excellent numbers. But here is the uncomfortable truth: **a positive result from a 99% accurate test might mean you have less than a 1% chance of actually being sick.**

<!--more-->

*Companion to the 3-part cancer diagnostics series:
[Part 1: The Four Pillars](/biotech/2026/02/16/cancer-testing-landscape.html) |
[Part 2: MRD](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html) |
[Part 3: Screening Wars](/biotech/2026/02/18/early-cancer-detection-screening-wars.html)*

## The 99% trap

Let's walk through the math with a concrete example. Imagine screening 100,000 people for pancreatic cancer -- one of the deadliest cancers with no established screening guideline.

- **Prevalence**: 0.01% (about 10 out of 100,000 people have undiagnosed pancreatic cancer)
- **Sensitivity**: 80% (the test catches 80% of true cancers)
- **Specificity**: 99% (the test correctly clears 99% of healthy people)

What happens when everybody gets tested?

- **True Positives**: 10 × 80% = **8** cancers detected
- **False Negatives**: 10 × 20% = **2** cancers missed
- **False Positives**: 99,990 × 1% = **999** healthy people told they might have cancer
- **True Negatives**: 99,990 × 99% = **98,991** correctly cleared

Now you test positive. There are 8 + 999 = 1,007 positive results in total. Only 8 of them actually have cancer. Your chance of actually having pancreatic cancer given a positive result is **8 / 1,007 = 0.8%**.

A 99% specific, 80% sensitive test -- and a positive result means less than a 1% chance of disease. That is not a flaw in the test. It is the base-rate problem, and it catches everyone from patients to physicians off guard.

## Bayes' theorem as belief updating

The result above feels wrong because our intuition conflates two different questions: "how accurate is this test?" and "how likely am I to be sick given a positive result?" Bayes' theorem is the bridge between them.

In its general form, Bayes' theorem relates a **prior** belief about some hypothesis $H$ to a **posterior** belief after observing evidence $E$:

$$
P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}
$$

The denominator $P(E)$ -- the total probability of observing the evidence -- expands via the law of total probability:

$$
P(E) = P(E \mid H) \cdot P(H) + P(E \mid \neg H) \cdot P(\neg H)
$$

In a diagnostic testing context, the terms map directly to clinical quantities:

| Bayes' Theorem | Diagnostic Testing |
|---|---|
| $H$ | Patient has the disease |
| $E$ | Test returns positive |
| $P(H)$ | **Prevalence** (prior probability) |
| $P(E \mid H)$ | **Sensitivity** (true positive rate) |
| $P(E \mid \neg H)$ | **1 - Specificity** (false positive rate) |
| $P(H \mid E)$ | **PPV** (posterior probability) |

Substituting these into Bayes' theorem gives us the **Positive Predictive Value**:

$$
PPV = \frac{\text{Sensitivity} \times \text{Prevalence}}{\text{Sensitivity} \times \text{Prevalence} + (1 - \text{Specificity}) \times (1 - \text{Prevalence})}
$$

The insight is that PPV depends on all three inputs -- sensitivity, specificity, *and* prevalence. A test does not have a fixed PPV; it has a PPV *for a given population*. The same blood draw that produces a 60% PPV when screening high-risk patients can produce a 2% PPV when screening the general population. The test did not get worse. The prior changed.

Your **prior** belief is the prevalence -- how likely is this person to have cancer before any test? The test result is **evidence** that updates that belief. The **posterior** is the PPV -- your revised estimate after seeing the result.

When the prior is very low (rare disease), even strong evidence (a 99% specific test) is not enough to produce high confidence. The false positives from the enormous healthy population swamp the true positives from the tiny sick population. This is why specificity matters so much more than sensitivity for population screening -- and why the MCED tests from [Part 3](/biotech/2026/02/18/early-cancer-detection-screening-wars.html) push specificity to 99%+ at the expense of sensitivity.

<style>
.bayes-controls { margin: 1.2rem 0 0.8rem; }
.bayes-slider-row {
  display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0;
}
.bayes-slider-row label {
  min-width: 90px; font-family: var(--font-mono); font-size: 0.82rem;
  color: var(--color-ink-muted);
}
.bayes-slider-row input[type="range"] {
  flex: 1; accent-color: var(--color-accent); height: 6px;
}
.bayes-slider-row .slider-val {
  min-width: 52px; text-align: right; font-family: var(--font-mono);
  font-size: 0.82rem; color: var(--color-ink); font-weight: 600;
}
.bayes-stat {
  text-align: center; margin: 0.8rem 0 1.2rem; padding: 0.8rem 1rem;
  background: var(--color-paper-warm); border-radius: 8px;
}
.bayes-stat .stat-value {
  font-size: 2.2rem; font-weight: 700; font-family: var(--font-mono);
  color: var(--color-accent); line-height: 1.2;
}
.bayes-stat .stat-label {
  font-size: 0.82rem; color: var(--color-ink-muted); display: block;
  margin-bottom: 0.2rem;
}
.bayes-stat .stat-detail {
  font-size: 0.8rem; color: var(--color-ink-muted); margin-top: 0.3rem;
}
.bayes-stat-pair {
  display: flex; gap: 1rem; margin: 0.8rem 0 1.2rem;
}
.bayes-stat-pair .bayes-stat { flex: 1; margin: 0; }
.preset-btns {
  display: flex; gap: 0.5rem; margin: 0.6rem 0 0.2rem;
  flex-wrap: wrap;
}
.preset-btn {
  padding: 0.3rem 0.75rem; border-radius: 6px; border: 1.5px solid var(--color-accent);
  background: transparent; color: var(--color-accent); font-family: var(--font-mono);
  font-size: 0.78rem; cursor: pointer; transition: background 0.15s, color 0.15s;
}
.preset-btn:hover, .preset-btn.active {
  background: var(--color-accent); color: var(--color-paper);
}
</style>

## The 1000 people

Each dot below represents one person in a screening population of 1,000. Drag the sliders to see how prevalence, sensitivity, and specificity reshape the balance between true positives, false positives, and everything in between.

<div style="max-width:700px;margin:0 auto">
<div class="bayes-controls">
  <div class="bayes-slider-row">
    <label for="dot-prev">Prevalence</label>
    <input type="range" id="dot-prev" min="0.1" max="50" step="0.1" value="1">
    <span class="slider-val" id="dot-prev-val">1%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="dot-sens">Sensitivity</label>
    <input type="range" id="dot-sens" min="50" max="100" step="1" value="80">
    <span class="slider-val" id="dot-sens-val">80%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="dot-spec">Specificity</label>
    <input type="range" id="dot-spec" min="50" max="100" step="0.5" value="99">
    <span class="slider-val" id="dot-spec-val">99%</span>
  </div>
</div>
<div id="dot-grid"></div>
<div class="bayes-stat" id="dot-ppv-display">
  <span class="stat-label">Positive Predictive Value</span>
  <span class="stat-value" id="dot-ppv-value">--</span>
  <div class="stat-detail" id="dot-ppv-detail"></div>
</div>
</div>

At the defaults (1% prevalence, 80% sensitivity, 99% specificity), about 8 green dots (true positives) share the "positive result" category with about 10 pink dots (false positives). Even at 99% specificity, more than half of positive results are wrong. Now drag prevalence down to 0.1% and watch the green dots vanish.

## PPV vs. prevalence

The chart below makes the relationship explicit. Each curve shows PPV across the full prevalence range for a different level of specificity. The data points mark where real cancer tests from [Part 3](/biotech/2026/02/18/early-cancer-detection-screening-wars.html) actually land when applied at their intended screening prevalence.

<div style="max-width:700px;margin:0 auto">
<div class="bayes-controls">
  <div class="bayes-slider-row">
    <label for="ppv-sens">Sensitivity</label>
    <input type="range" id="ppv-sens" min="50" max="100" step="1" value="80">
    <span class="slider-val" id="ppv-sens-val">80%</span>
  </div>
</div>
<div id="ppv-curves"></div>
</div>

Notice how the curves collapse to near-zero PPV below 0.1% prevalence regardless of specificity. This is why Galleri's 99.5% specificity still produces a low PPV for rare individual cancers like pancreatic (0.013% prevalence) -- even though it performs well when screening for *any* cancer at combined ~1.4% prevalence.

## The power of re-testing

If a single positive test leaves you uncertain, what happens when you test again? Each round of testing uses the **posterior** from the previous round as the new **prior**. The PPV climbs dramatically with each consecutive positive result.

<div style="max-width:700px;margin:0 auto">
<div class="bayes-controls">
  <div class="bayes-slider-row">
    <label for="bu-prev">Prior</label>
    <input type="range" id="bu-prev" min="0.1" max="50" step="0.1" value="1">
    <span class="slider-val" id="bu-prev-val">1%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="bu-sens">Sensitivity</label>
    <input type="range" id="bu-sens" min="50" max="100" step="1" value="80">
    <span class="slider-val" id="bu-sens-val">80%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="bu-spec">Specificity</label>
    <input type="range" id="bu-spec" min="50" max="100" step="0.5" value="99">
    <span class="slider-val" id="bu-spec-val">99%</span>
  </div>
</div>
<div id="bayes-update"></div>
</div>

This is the mathematical case for **confirmatory testing**. A screening test with moderate PPV followed by a highly specific diagnostic test (like colonoscopy for CRC, or imaging + biopsy for MCED) can produce near-certainty. This is exactly why every positive Cologuard, Shield, and Galleri result triggers a follow-up procedure.

## The cancer test reality check

Here is the punchline. Each bar below shows the Positive Predictive Value (PPV) for a real test from [Part 3](/biotech/2026/02/18/early-cancer-detection-screening-wars.html), computed at its intended screening prevalence using the Bayes' theorem formula above. The dashed line marks the coin-flip threshold: below 50%, a positive result is more likely to be wrong than right.

<div id="reality-check" style="width:100%;max-width:700px;margin:0 auto"></div>

A few things stand out:

- **SPOT-MAS** and **Galleri** lead because their ultra-high specificity (99.7% and 99.5%) compensates for relatively modest sensitivity. This is by design -- when you screen millions of healthy people, every tenth of a percent of specificity counts.
- **CRC blood tests** (Shield, Cologuard Plus) have low PPV despite strong sensitivity because CRC-specific prevalence (~0.5%) is lower than combined all-cancer prevalence (~1.4%), and their specificities (89.6%, 91%) allow more false positives through.
- **Galleri for pancreatic cancer** -- arguably the cancer where early detection would save the most lives -- has a PPV of about 1.3%. In a population of 100,000 people, a positive Galleri result for pancreatic cancer is wrong 99 times out of 100. This is the base-rate wall.

## What this means for screening

**The MCED paradox from Part 3 is mathematically inevitable.** When I wrote about Galleri's 51.5% sensitivity and Cancerguard's 64% sensitivity, it sounded like the real bottleneck was catching more cancers. But Bayes' theorem reveals that the binding constraint for population screening is specificity, not sensitivity. Improving Galleri's sensitivity from 51.5% to 90% while keeping specificity at 99.5% would only raise its all-cancer PPV from ~59% to ~72%. Improving its specificity from 99.5% to 99.9% at the same 51.5% sensitivity would raise PPV from ~59% to ~88%.

**Pre-test probability changes everything.** These PPV calculations assume population-level screening of asymptomatic people. A patient who walks into a clinic with symptoms -- weight loss, jaundice, a palpable mass -- has a much higher pre-test probability than the general population. For symptomatic patients, even a moderately specific test can produce high PPV. This is why the compliance argument for blood-based CRC tests (Shield vs. Cologuard Plus) extends beyond just getting people tested -- it also captures people who might have early symptoms but would never schedule a colonoscopy.

**The layered approach works.** As the re-testing chart shows, sequential testing dramatically improves confidence. The clinical workflow for most screening positives already reflects this: positive screen → confirmatory imaging → biopsy. Each step has higher specificity than the last, and Bayesian updating does the heavy lifting. The real challenge is ensuring patients actually complete the full cascade -- a positive Galleri result that never gets followed up with imaging provides no clinical benefit.

---

## When the prior flips: MRD monitoring

Everything above assumes we are screening healthy people where the prior is low -- 0.01% for pancreatic cancer, 1.4% for any cancer. Now flip the scenario entirely. A Stage III colorectal cancer patient finishes chemo. Historical data says roughly 40% of these patients will relapse within three years. The prior is not 1.4%. It is **40%**.

This is the world of **Minimal Residual Disease (MRD)** testing from [Part 2](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html). The question is no longer "does this healthy person have cancer?" but "does this treated cancer patient still have microscopic disease?" And because the prior is high, the entire Bayesian calculus inverts.

When the prior is high, PPV is already strong -- a positive MRD result at 40% prior and 98% specificity gives a PPV above 95%. The clinical anxiety is not "is this positive real?" but rather "can I trust a negative?" This is still Bayes' theorem, but now the evidence $E$ is a *negative* test result. Earlier we asked $P(disease \mid positive)$ and got PPV. Now we ask $P(disease\text{-}free \mid negative)$:

$$
P(\neg H \mid E^-) = \frac{P(E^- \mid \neg H) \cdot P(\neg H)}{P(E^- \mid \neg H) \cdot P(\neg H) + P(E^- \mid H) \cdot P(H)}
$$

The terms map to the same clinical quantities, just flipped: $P(E^- \mid \neg H)$ is **specificity** (healthy people correctly testing negative), $P(E^- \mid H)$ is **1 - sensitivity** (sick people missed), $P(\neg H)$ is $1 - prevalence$. Substituting gives us the **Negative Predictive Value**:

$$
NPV = \frac{Spec \times (1 - Prev)}{Spec \times (1 - Prev) + (1 - Sens) \times Prev}
$$

The Alpha-CORRECT trial showed this in action: ctDNA-negative patients had 96.1% three-year relapse-free survival versus 54.5% for ctDNA-positive patients (HR 9.6). A negative MRD test is powerfully reassuring -- but only if sensitivity is high enough.

<div style="max-width:700px;margin:0 auto">
<div class="bayes-controls">
  <div class="bayes-slider-row">
    <label for="mrd-prev">Prior</label>
    <input type="range" id="mrd-prev" min="0.1" max="60" step="0.1" value="40">
    <span class="slider-val" id="mrd-prev-val">40%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="mrd-sens">Sensitivity</label>
    <input type="range" id="mrd-sens" min="50" max="100" step="1" value="94">
    <span class="slider-val" id="mrd-sens-val">94%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="mrd-spec">Specificity</label>
    <input type="range" id="mrd-spec" min="50" max="100" step="0.5" value="98">
    <span class="slider-val" id="mrd-spec-val">98%</span>
  </div>
  <div class="preset-btns">
    <button class="preset-btn" data-prev="1.4" data-sens="80" data-spec="99">Screening (1.4%)</button>
    <button class="preset-btn active" data-prev="40" data-sens="94" data-spec="98">MRD (40%)</button>
  </div>
</div>
<div class="bayes-stat-pair">
  <div class="bayes-stat">
    <span class="stat-label">Positive Predictive Value</span>
    <span class="stat-value" id="mrd-ppv-value">--</span>
    <div class="stat-detail" id="mrd-ppv-detail"></div>
  </div>
  <div class="bayes-stat">
    <span class="stat-label">Negative Predictive Value</span>
    <span class="stat-value" id="mrd-npv-value">--</span>
    <div class="stat-detail" id="mrd-npv-detail"></div>
  </div>
</div>
</div>

Hit the preset buttons to toggle between screening and MRD contexts. Same Bayes' theorem, same formula -- radically different clinical story. At screening prevalence, PPV collapses and a positive means almost nothing. At MRD prevalence, PPV is near-certain and the question becomes whether NPV is high enough to safely de-escalate treatment.

## The MRD test landscape

Not all MRD tests are created equal. The chart below shows PPV and NPV for five commercial MRD assays at an adjustable recurrence rate. The tests are sorted by NPV, because that is the metric that matters most in MRD -- can a negative result actually reassure you?

<div style="max-width:700px;margin:0 auto">
<div class="bayes-controls">
  <div class="bayes-slider-row">
    <label for="mrd-land-prev">Recurrence</label>
    <input type="range" id="mrd-land-prev" min="5" max="60" step="1" value="40">
    <span class="slider-val" id="mrd-land-prev-val">40%</span>
  </div>
</div>
<div id="mrd-landscape"></div>
</div>

Notice how sensitivity differences map directly to NPV gaps. NeXT Personal's 100% analytical sensitivity gives it a perfect NPV -- every negative is a true negative. Drop to Reveal's 81% sensitivity and NPV falls to around 86% at 40% recurrence, meaning roughly 1 in 7 "all clear" results are wrong. For a cancer patient deciding whether to skip additional chemo, that gap is the difference between confidence and anxiety.

*A few caveats on these numbers. NeXT Personal's 100% sensitivity comes from a validation cohort of n=493 -- impressive but small; larger clinical cohorts are ongoing. Signatera's 94% is a longitudinal/surveillance figure that varies by cancer type (CRC 88-93%, bladder 99%, lung 80-99%). Oncodetect's 91%/94% are the surveillance monitoring values from the Alpha-CORRECT CRC trial; the post-surgical landmark timepoint is lower (78%/80%). Reveal's 81% is the COSMOS 2024 longitudinal figure for stage II+ CRC; earlier landmark data showed 55-63%. All numbers from [OpenOnco](https://openonco.org) (v. Feb 15, 2026).*

## Serial negative tests: the power of re-monitoring

In screening, we saw that consecutive *positive* tests drive PPV toward certainty. In MRD monitoring, the mirror image matters: consecutive *negative* results drive the posterior probability of residual disease toward zero. Each negative updates the prior downward:

$$
P(disease \mid negative) = \frac{(1 - Sens) \times P(disease)}{(1 - Sens) \times P(disease) + Spec \times (1 - P(disease))}
$$

This is why MRD monitoring is not a single test but a longitudinal program -- blood draws every 3-6 months for years. A single negative result at 94% sensitivity still leaves meaningful residual uncertainty. But three or four consecutive negatives compound the evidence, each one shrinking the posterior further.

<div style="max-width:700px;margin:0 auto">
<div class="bayes-controls">
  <div class="bayes-slider-row">
    <label for="sn-prev">Prior</label>
    <input type="range" id="sn-prev" min="1" max="60" step="1" value="40">
    <span class="slider-val" id="sn-prev-val">40%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="sn-sens">Sensitivity</label>
    <input type="range" id="sn-sens" min="50" max="100" step="1" value="94">
    <span class="slider-val" id="sn-sens-val">94%</span>
  </div>
  <div class="bayes-slider-row">
    <label for="sn-spec">Specificity</label>
    <input type="range" id="sn-spec" min="50" max="100" step="0.5" value="98">
    <span class="slider-val" id="sn-spec-val">98%</span>
  </div>
</div>
<div id="serial-neg"></div>
</div>

At the defaults (40% prior, 94% sensitivity), a single negative drops the probability of residual disease from 40% to about 3.9%. A second negative pushes it below 0.3%. By the fourth consecutive negative, you are well under 0.01%. This is the mathematical case for serial MRD monitoring -- and why ctDNA-guided approaches show a median 1.4-month lead time over conventional imaging.

For MRD monitoring, sensitivity is the binding constraint -- exactly the mirror of screening. Every percentage point of sensitivity translates directly into NPV, the confidence a patient and their oncologist can place in a negative result. This is why the MRD assays from [Part 2](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html) compete on sensitivity (81% to 100%), and why serial monitoring compounds the benefit of even moderate sensitivity into strong cumulative evidence.

## The same theorem, opposite constraints

Bayes' theorem is one equation, but it produces two completely different clinical stories depending on the prior. For population screening where the prior is low, specificity is everything -- false positives overwhelm true positives, and PPV is the metric that matters. For MRD monitoring where the prior is high, sensitivity takes over -- false negatives erode trust in a clean result, and NPV becomes the metric that matters. And in both contexts, serial testing is the great equalizer: consecutive positives rescue a weak PPV, consecutive negatives rescue a weak NPV, each round compounding the evidence through the same Bayesian update.

The practical upshot: when evaluating any cancer test, the first question is not "how accurate is it?" but "what is the prior?" A 99% accurate test is a clinical disaster for pancreatic screening and a clinical triumph for post-chemo MRD monitoring. The test did not change. The prior did.

---

And because no post in this series is complete without a Gemini-drawn xkcd-style cheat sheet -- here's Bayes' theorem in oncology on a napkin.

![The Prior Is Everything: Screening vs. MRD Monitoring — same 99% test, opposite clinical stories](/assets/posts-media/mrd-vs-mced-xkcd.jpg)
*TL;DR: The prior is everything. Screening healthy people (low prior)? False positives swamp true positives, specificity is king, and PPV is your metric. Monitoring a post-chemo patient (high prior)? False negatives erode trust in a clean result, sensitivity is king, and NPV is your metric. Same test, same 99% -- opposite constraints, opposite failure modes. Serial testing rescues both.*

*Companion to the cancer diagnostics series:
[Part 1: The Four Pillars](/biotech/2026/02/16/cancer-testing-landscape.html) |
[Part 2: MRD](/biotech/2026/02/17/mrd-hunting-invisible-cancer.html) |
[Part 3: Screening Wars](/biotech/2026/02/18/early-cancer-detection-screening-wars.html).
Data from [OpenOnco](https://openonco.org) (v. Feb 15, 2026).*

<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="/assets/js/cancer-charts.js"></script>
<script>
(function() {
  'use strict';

  // ============================================================
  // Shared Math Utilities
  // ============================================================

  function ppv(prev, sens, spec) {
    var num = sens * prev;
    var den = num + (1 - spec) * (1 - prev);
    return den === 0 ? 0 : num / den;
  }

  function computeConfusion(prev, sens, spec, N) {
    var sick = Math.round(N * prev);
    var healthy = N - sick;
    var tp = Math.round(sick * sens);
    var fn = sick - tp;
    var fp = Math.round(healthy * (1 - spec));
    var tn = healthy - fp;
    return { TP: tp, FN: fn, FP: fp, TN: tn };
  }

  function sliderVal(id) {
    var el = document.getElementById(id);
    return el ? parseFloat(el.value) : 0;
  }

  function setDisplay(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // Seeded Fisher-Yates shuffle for stable dot grid
  function computePermutation(n, seed) {
    var arr = [];
    for (var i = 0; i < n; i++) arr.push(i);
    var s = seed;
    function rand() {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    }
    for (var i = n - 1; i > 0; i--) {
      var j = Math.floor(rand() * (i + 1));
      var tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
  }

  // ============================================================
  // Chart 1: Dot Grid (1000 People)
  // ============================================================

  var dotPerm = null;
  var dotCircles = null;

  function drawDotGrid() {
    var c = CancerCharts.getColors();
    var container = d3.select('#dot-grid');
    container.selectAll('*').remove();

    dotPerm = computePermutation(1000, 42);

    var cols = 40, rows = 25;
    var dotR = 5, gapX = 15.5, gapY = 14;
    var margin = { top: 8, right: 8, bottom: 40, left: 8 };
    var gridW = cols * gapX;
    var gridH = rows * gapY;
    var width = gridW + margin.left + margin.right;
    var height = gridH + margin.top + margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var dotsData = [];
    for (var i = 0; i < 1000; i++) {
      dotsData.push({
        idx: i,
        x: (i % cols) * gapX + gapX / 2,
        y: Math.floor(i / cols) * gapY + gapY / 2
      });
    }

    dotCircles = g.selectAll('circle')
      .data(dotsData).enter().append('circle')
      .attr('cx', function(d) { return d.x; })
      .attr('cy', function(d) { return d.y; })
      .attr('r', dotR)
      .attr('fill', c.grid);

    // Legend
    var legendItems = [
      { label: 'True Positive', key: 'green' },
      { label: 'False Positive', key: 'pink' },
      { label: 'False Negative', key: 'yellow' },
      { label: 'True Negative', key: 'grid' }
    ];
    var legend = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + (margin.top + gridH + 14) + ')');
    legendItems.forEach(function(item, i) {
      var lx = i * (gridW / 4);
      legend.append('circle')
        .attr('cx', lx + 6).attr('cy', 0).attr('r', 4)
        .attr('fill', c[item.key]).attr('fill-opacity', item.key === 'grid' ? 1 : 0.85);
      legend.append('text')
        .attr('x', lx + 14).attr('y', 4)
        .attr('fill', c.text).attr('font-size', 10).text(item.label);
    });

    updateDotGrid(false);
  }

  function updateDotGrid(animate) {
    if (!dotCircles) return;
    var c = CancerCharts.getColors();

    var prev = sliderVal('dot-prev') / 100;
    var sens = sliderVal('dot-sens') / 100;
    var spec = sliderVal('dot-spec') / 100;
    var cm = computeConfusion(prev, sens, spec, 1000);
    var nSick = Math.round(1000 * prev);

    var catColor = { TP: c.green, FN: c.yellow, FP: c.pink, TN: c.grid };

    var sel = animate !== false ? dotCircles.transition().duration(250) : dotCircles;
    sel.attr('fill', function(d) {
      var pi = dotPerm[d.idx];
      if (pi < cm.TP) return catColor.TP;
      if (pi < nSick) return catColor.FN;
      if (pi < nSick + cm.FP) return catColor.FP;
      return catColor.TN;
    });

    var totalPos = cm.TP + cm.FP;
    var ppvPct = totalPos > 0 ? (cm.TP / totalPos * 100) : 0;
    setDisplay('dot-ppv-value', ppvPct.toFixed(1) + '%');
    setDisplay('dot-ppv-detail',
      cm.TP + ' true positive' + (cm.TP !== 1 ? 's' : '') +
      ' out of ' + totalPos + ' positive result' + (totalPos !== 1 ? 's' : ''));
    setDisplay('dot-prev-val', sliderVal('dot-prev') + '%');
    setDisplay('dot-sens-val', sliderVal('dot-sens') + '%');
    setDisplay('dot-spec-val', sliderVal('dot-spec') + '%');
  }

  // ============================================================
  // Chart 2: PPV vs. Prevalence Curves
  // ============================================================

  var ppvSvgG = null;
  var ppvX = null, ppvY = null;
  var ppvInnerW = 0, ppvInnerH = 0;

  function drawPPVCurves() {
    var c = CancerCharts.getColors();
    var container = d3.select('#ppv-curves');
    container.selectAll('*').remove();

    var margin = { top: 52, right: 25, bottom: 50, left: 55 };
    var width = 700, height = 457;
    ppvInnerW = width - margin.left - margin.right;
    ppvInnerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    ppvX = d3.scaleLog().domain([0.01, 50]).range([0, ppvInnerW]).clamp(true);
    ppvY = d3.scaleLinear().domain([0, 100]).range([ppvInnerH, 0]);

    // Grid
    g.append('g').selectAll('line')
      .data(ppvY.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', ppvInnerW)
      .attr('y1', function(d) { return ppvY(d); })
      .attr('y2', function(d) { return ppvY(d); })
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    [0.01, 0.1, 1, 10].forEach(function(v) {
      g.append('line')
        .attr('x1', ppvX(v)).attr('x2', ppvX(v))
        .attr('y1', 0).attr('y2', ppvInnerH)
        .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');
    });

    // Axes
    var xTicks = [0.01, 0.1, 1, 10, 50];
    g.append('g').attr('transform', 'translate(0,' + ppvInnerH + ')')
      .call(d3.axisBottom(ppvX).tickValues(xTicks)
        .tickFormat(function(d) { return d < 1 ? d + '%' : d + '%'; }))
      .call(function(g) {
        g.selectAll('text').attr('fill', c.text);
        g.selectAll('line,path').attr('stroke', c.muted);
      });

    g.append('g')
      .call(d3.axisLeft(ppvY).ticks(5).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) {
        g.selectAll('text').attr('fill', c.text);
        g.selectAll('line,path').attr('stroke', c.muted);
      });

    // Axis labels
    svg.append('text').attr('x', margin.left + ppvInnerW / 2).attr('y', height - 6)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('Prevalence (log scale)');

    svg.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + ppvInnerH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('Positive Predictive Value');

    // Title
    svg.append('text').attr('x', margin.left + ppvInnerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('PPV vs. Prevalence at Different Specificity Levels');

    // Legend below title
    var specLegend = [
      { label: '90%', color: c.muted },
      { label: '95%', color: c.yellow },
      { label: '99%', color: c.blue },
      { label: '99.5%', color: c.green },
      { label: '99.9%', color: c.pink }
    ];
    var lgG = svg.append('g')
      .attr('transform', 'translate(' + (margin.left + ppvInnerW / 2) + ',32)');
    var lgSpacing = 80;
    specLegend.forEach(function(sl, i) {
      var lx = (i - (specLegend.length - 1) / 2) * lgSpacing;
      lgG.append('line')
        .attr('x1', lx - 18).attr('x2', lx - 4)
        .attr('y1', 0).attr('y2', 0)
        .attr('stroke', sl.color).attr('stroke-width', 2.5);
      lgG.append('text')
        .attr('x', lx).attr('y', 4)
        .attr('fill', sl.color).attr('font-size', 10).attr('font-weight', 600)
        .text(sl.label);
    });

    // Content group for curves + data points (cleared on update)
    ppvSvgG = g.append('g').attr('class', 'ppv-content');

    updatePPVCurves();
  }

  function updatePPVCurves() {
    if (!ppvSvgG) return;
    var c = CancerCharts.getColors();
    ppvSvgG.selectAll('*').remove();

    var sens = sliderVal('ppv-sens') / 100;
    setDisplay('ppv-sens-val', sliderVal('ppv-sens') + '%');

    var specLevels = [
      { val: 0.90, label: '90%', color: c.muted },
      { val: 0.95, label: '95%', color: c.yellow },
      { val: 0.99, label: '99%', color: c.blue },
      { val: 0.995, label: '99.5%', color: c.green },
      { val: 0.999, label: '99.9%', color: c.pink }
    ];

    // Generate prevalence points (log-spaced)
    var prevPoints = [];
    for (var lp = Math.log10(0.01); lp <= Math.log10(50); lp += 0.015) {
      prevPoints.push(Math.pow(10, lp));
    }
    prevPoints.push(50);

    var line = d3.line()
      .x(function(d) { return ppvX(d.prev); })
      .y(function(d) { return ppvY(d.ppv); });

    specLevels.forEach(function(sl) {
      var data = prevPoints.map(function(p) {
        return { prev: p, ppv: ppv(p / 100, sens, sl.val) * 100 };
      });

      ppvSvgG.append('path')
        .datum(data).attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', sl.color)
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.85);

    });

    // Overlay real test data points
    var realTests = [
      { name: 'Galleri', sens: 0.515, spec: 0.995, prev: 1.4, cat: 'MCED' },
      { name: 'Cancerguard', sens: 0.64, spec: 0.974, prev: 1.4, cat: 'MCED' },
      { name: 'Shield MCD', sens: 0.60, spec: 0.985, prev: 1.4, cat: 'MCED' },
      { name: 'SPOT-MAS', sens: 0.708, spec: 0.997, prev: 1.4, cat: 'MCED' },
      { name: 'Cologuard Plus', sens: 0.939, spec: 0.91, prev: 0.5, cat: 'CRC-stool' },
      { name: 'Shield', sens: 0.831, spec: 0.896, prev: 0.5, cat: 'CRC-blood' }
    ];

    realTests.forEach(function(t) {
      var testPPV = ppv(t.prev / 100, t.sens, t.spec) * 100;
      ppvSvgG.append('circle')
        .attr('cx', ppvX(t.prev))
        .attr('cy', ppvY(testPPV))
        .attr('r', 5)
        .attr('fill', c[t.cat] || c.muted)
        .attr('fill-opacity', 0.9)
        .attr('stroke', c.text)
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          CancerCharts.showTooltip(event,
            '<strong>' + t.name + '</strong>' +
            '<br/>Prevalence: ' + t.prev + '%' +
            '<br/>Sensitivity: ' + (t.sens * 100).toFixed(1) + '%' +
            '<br/>Specificity: ' + (t.spec * 100).toFixed(1) + '%' +
            '<br/>PPV: ' + testPPV.toFixed(1) + '%');
        })
        .on('mousemove', CancerCharts.moveTooltip)
        .on('mouseout', CancerCharts.hideTooltip);

      ppvSvgG.append('text')
        .attr('x', ppvX(t.prev))
        .attr('y', ppvY(testPPV) - 9)
        .attr('text-anchor', 'middle')
        .attr('fill', c[t.cat] || c.muted)
        .attr('font-size', 9)
        .attr('font-weight', 600)
        .text(t.name);
    });
  }

  // ============================================================
  // Chart 3: Sequential Bayesian Updating
  // ============================================================

  function drawBayesUpdate() {
    var c = CancerCharts.getColors();
    var container = d3.select('#bayes-update');
    container.selectAll('*').remove();

    var margin = { top: 35, right: 60, bottom: 40, left: 120 };
    var width = 700, height = 280;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('Bayesian Updating: Consecutive Positive Tests');

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var labels = ['Prior', 'After Test 1', 'After Test 2', 'After Test 3'];
    var y = d3.scaleBand().domain(labels).range([0, innerH]).padding(0.25);
    var x = d3.scaleLinear().domain([0, 100]).range([0, innerW]);

    // Grid
    g.append('g').selectAll('line')
      .data(x.ticks(5)).enter().append('line')
      .attr('x1', function(d) { return x(d); })
      .attr('x2', function(d) { return x(d); })
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    // Axes
    g.append('g').attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x).ticks(5).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) {
        g.selectAll('text').attr('fill', c.text);
        g.selectAll('line,path').attr('stroke', c.muted);
      });

    // Row labels
    labels.forEach(function(label) {
      g.append('text')
        .attr('x', -10)
        .attr('y', y(label) + y.bandwidth() / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', c.text)
        .attr('font-size', 12)
        .text(label);
    });

    // Bars (will be updated)
    var barColors = [c.muted, c.yellow, c.pink, c.green];
    labels.forEach(function(label, i) {
      g.append('rect')
        .attr('class', 'bu-bar-' + i)
        .attr('x', 0)
        .attr('y', y(label))
        .attr('height', y.bandwidth())
        .attr('width', 0)
        .attr('fill', barColors[i])
        .attr('fill-opacity', 0.8)
        .attr('rx', 3);

      g.append('text')
        .attr('class', 'bu-label-' + i)
        .attr('x', 0)
        .attr('y', y(label) + y.bandwidth() / 2 + 5)
        .attr('fill', c.text)
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text('');
    });

    // Store references for updates
    container.node().__buG = g;
    container.node().__buX = x;
    container.node().__buY = y;
    container.node().__buLabels = labels;

    updateBayesUpdate();
  }

  function updateBayesUpdate() {
    var container = d3.select('#bayes-update');
    var node = container.node();
    if (!node || !node.__buG) return;

    var g = node.__buG;
    var x = node.__buX;
    var c = CancerCharts.getColors();

    var prior = sliderVal('bu-prev') / 100;
    var sens = sliderVal('bu-sens') / 100;
    var spec = sliderVal('bu-spec') / 100;

    // Compute sequential posteriors
    var probs = [prior];
    for (var i = 0; i < 3; i++) {
      probs.push(ppv(probs[i], sens, spec));
    }

    probs.forEach(function(p, i) {
      var pPct = p * 100;
      g.select('.bu-bar-' + i)
        .transition().duration(300)
        .attr('width', x(pPct));

      g.select('.bu-label-' + i)
        .transition().duration(300)
        .attr('x', x(pPct) + 6)
        .tween('text', function() {
          var that = d3.select(this);
          var current = parseFloat(that.text()) || 0;
          var interp = d3.interpolateNumber(current, pPct);
          return function(t) {
            that.text(interp(t).toFixed(1) + '%');
          };
        });
    });

    setDisplay('bu-prev-val', sliderVal('bu-prev') + '%');
    setDisplay('bu-sens-val', sliderVal('bu-sens') + '%');
    setDisplay('bu-spec-val', sliderVal('bu-spec') + '%');
  }

  // ============================================================
  // Chart 4: Reality Check (static horizontal bar chart)
  // ============================================================

  function drawRealityCheck() {
    var c = CancerCharts.getColors();
    var container = d3.select('#reality-check');
    container.selectAll('*').remove();

    var tests = [
      { name: 'SPOT-MAS', detail: 'any cancer · 1.4% prev', sens: 0.708, spec: 0.997, prev: 0.014, cat: 'MCED' },
      { name: 'Galleri', detail: 'any cancer · 1.4% prev', sens: 0.515, spec: 0.995, prev: 0.014, cat: 'MCED' },
      { name: 'Shield MCD', detail: 'any cancer · 1.4% prev', sens: 0.60, spec: 0.985, prev: 0.014, cat: 'MCED' },
      { name: 'Galleri', detail: 'CRC only · 0.5% prev', sens: 0.515, spec: 0.995, prev: 0.005, cat: 'MCED' },
      { name: 'Cancerguard', detail: 'any cancer · 1.4% prev', sens: 0.64, spec: 0.974, prev: 0.014, cat: 'MCED' },
      { name: 'Cologuard Plus', detail: 'CRC · 0.5% prev', sens: 0.939, spec: 0.91, prev: 0.005, cat: 'CRC-stool' },
      { name: 'Shield', detail: 'CRC · 0.5% prev', sens: 0.831, spec: 0.896, prev: 0.005, cat: 'CRC-blood' },
      { name: 'Galleri', detail: 'pancreatic · 0.013% prev', sens: 0.515, spec: 0.995, prev: 0.00013, cat: 'MCED' }
    ];

    // Compute PPV for each
    tests.forEach(function(t) {
      t.ppv = ppv(t.prev, t.sens, t.spec) * 100;
    });

    // Sort descending by PPV
    tests.sort(function(a, b) { return b.ppv - a.ppv; });

    var margin = { top: 35, right: 55, bottom: 40, left: 185 };
    var width = 700, height = 380;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var yLabels = tests.map(function(t) { return t.name + ' (' + t.detail + ')'; });
    var y = d3.scaleBand().domain(yLabels).range([0, innerH]).padding(0.2);
    var x = d3.scaleLinear().domain([0, 100]).range([0, innerW]);

    // Grid
    g.append('g').selectAll('line')
      .data(x.ticks(5)).enter().append('line')
      .attr('x1', function(d) { return x(d); })
      .attr('x2', function(d) { return x(d); })
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    // Coin flip line at 50%
    g.append('line')
      .attr('x1', x(50)).attr('x2', x(50))
      .attr('y1', -5).attr('y2', innerH + 5)
      .attr('stroke', c.pink)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4');

    g.append('text')
      .attr('x', x(50) + 4).attr('y', -8)
      .attr('fill', c.pink)
      .attr('font-size', 10)
      .attr('font-style', 'italic')
      .text('coin flip');

    // Axes
    g.append('g').attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x).ticks(5).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) {
        g.selectAll('text').attr('fill', c.text);
        g.selectAll('line,path').attr('stroke', c.muted);
      });

    // Row labels
    tests.forEach(function(t, i) {
      var label = yLabels[i];
      g.append('text')
        .attr('x', -6)
        .attr('y', y(label) + y.bandwidth() / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', c.text)
        .attr('font-size', 10)
        .text(label);
    });

    // Bars
    tests.forEach(function(t, i) {
      var label = yLabels[i];
      var barColor = c[t.cat] || c.muted;

      g.append('rect')
        .attr('x', 0)
        .attr('y', y(label))
        .attr('width', x(t.ppv))
        .attr('height', y.bandwidth())
        .attr('fill', barColor)
        .attr('fill-opacity', 0.8)
        .attr('rx', 3)
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          CancerCharts.showTooltip(event,
            '<strong>' + t.name + '</strong>' +
            '<br/>' + t.detail +
            '<br/>Sensitivity: ' + (t.sens * 100).toFixed(1) + '%' +
            '<br/>Specificity: ' + (t.spec * 100).toFixed(1) + '%' +
            '<br/>PPV: ' + t.ppv.toFixed(1) + '%');
        })
        .on('mousemove', CancerCharts.moveTooltip)
        .on('mouseout', CancerCharts.hideTooltip);

      // PPV value label
      g.append('text')
        .attr('x', x(t.ppv) + 5)
        .attr('y', y(label) + y.bandwidth() / 2 + 4)
        .attr('fill', c.text)
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .text(t.ppv.toFixed(1) + '%');
    });

    // Title
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('PPV at Screening Prevalence: The Reality Check');

    // X-axis label
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', height - 4)
      .attr('text-anchor', 'middle').attr('fill', c.muted).attr('font-size', 13)
      .text('Positive Predictive Value');
  }

  // ============================================================
  // Shared: NPV & Negative Update Utilities
  // ============================================================

  function npv(prev, sens, spec) {
    var num = spec * (1 - prev);
    var den = num + (1 - sens) * prev;
    return den === 0 ? 1 : num / den;
  }

  function negUpdate(prev, sens, spec) {
    var num = (1 - sens) * prev;
    var den = num + spec * (1 - prev);
    return den === 0 ? 0 : num / den;
  }

  // ============================================================
  // Chart 5: MRD Dual PPV/NPV Display
  // ============================================================

  function drawMrdDual() {
    updateMrdDual();
  }

  function updateMrdDual() {
    var prev = sliderVal('mrd-prev') / 100;
    var sens = sliderVal('mrd-sens') / 100;
    var spec = sliderVal('mrd-spec') / 100;

    var ppvVal = ppv(prev, sens, spec) * 100;
    var npvVal = npv(prev, sens, spec) * 100;

    setDisplay('mrd-ppv-value', ppvVal.toFixed(1) + '%');
    setDisplay('mrd-npv-value', npvVal.toFixed(1) + '%');
    setDisplay('mrd-ppv-detail',
      'If positive, ' + ppvVal.toFixed(1) + '% chance of disease');
    setDisplay('mrd-npv-detail',
      'If negative, ' + npvVal.toFixed(1) + '% chance disease-free');
    setDisplay('mrd-prev-val', sliderVal('mrd-prev') + '%');
    setDisplay('mrd-sens-val', sliderVal('mrd-sens') + '%');
    setDisplay('mrd-spec-val', sliderVal('mrd-spec') + '%');
  }

  function wireMrdPresets() {
    var btns = document.querySelectorAll('.preset-btn');
    btns.forEach(function(btn) {
      btn.addEventListener('click', function() {
        btns.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');

        var prevEl = document.getElementById('mrd-prev');
        var sensEl = document.getElementById('mrd-sens');
        var specEl = document.getElementById('mrd-spec');
        if (prevEl) prevEl.value = btn.dataset.prev;
        if (sensEl) sensEl.value = btn.dataset.sens;
        if (specEl) specEl.value = btn.dataset.spec;
        updateMrdDual();
      });
    });
  }

  // ============================================================
  // Chart 6: MRD Test Landscape (Grouped Horizontal Bars)
  // ============================================================

  function drawMrdLandscape() {
    var c = CancerCharts.getColors();
    var container = d3.select('#mrd-landscape');
    container.selectAll('*').remove();

    var prev = sliderVal('mrd-land-prev') / 100;
    setDisplay('mrd-land-prev-val', sliderVal('mrd-land-prev') + '%');

    var tests = [
      { name: 'NeXT Personal', sens: 1.00, spec: 0.999 },
      { name: 'Signatera', sens: 0.94, spec: 0.98 },
      { name: 'clonoSEQ', sens: 0.95, spec: 0.99 },
      { name: 'Oncodetect', sens: 0.91, spec: 0.94 },
      { name: 'Reveal', sens: 0.81, spec: 0.98 }
    ];

    tests.forEach(function(t) {
      t.ppv = ppv(prev, t.sens, t.spec) * 100;
      t.npv = npv(prev, t.sens, t.spec) * 100;
    });
    tests.sort(function(a, b) { return b.npv - a.npv; });

    var margin = { top: 35, right: 55, bottom: 40, left: 120 };
    var barGroupH = 36;
    var barH = 14;
    var width = 700;
    var innerW = width - margin.left - margin.right;
    var innerH = tests.length * barGroupH;
    var height = innerH + margin.top + margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var x = d3.scaleLinear().domain([0, 100]).range([0, innerW]);

    // Grid
    g.append('g').selectAll('line')
      .data(x.ticks(5)).enter().append('line')
      .attr('x1', function(d) { return x(d); })
      .attr('x2', function(d) { return x(d); })
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    // 95% reference line
    g.append('line')
      .attr('x1', x(95)).attr('x2', x(95))
      .attr('y1', -5).attr('y2', innerH + 5)
      .attr('stroke', c.pink)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4');
    g.append('text')
      .attr('x', x(95) + 4).attr('y', -8)
      .attr('fill', c.pink).attr('font-size', 10)
      .attr('font-style', 'italic').text('95%');

    // X axis
    g.append('g').attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x).ticks(5).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) {
        g.selectAll('text').attr('fill', c.text);
        g.selectAll('line,path').attr('stroke', c.muted);
      });

    tests.forEach(function(t, i) {
      var yOff = i * barGroupH;

      // Row label
      g.append('text')
        .attr('x', -8).attr('y', yOff + barGroupH / 2 + 3)
        .attr('text-anchor', 'end').attr('fill', c.text)
        .attr('font-size', 11).text(t.name);

      // PPV bar (top)
      g.append('rect')
        .attr('x', 0).attr('y', yOff + 2)
        .attr('width', x(t.ppv)).attr('height', barH)
        .attr('fill', c.green).attr('fill-opacity', 0.8).attr('rx', 2)
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          CancerCharts.showTooltip(event,
            '<strong>' + t.name + '</strong>' +
            '<br/>Sens: ' + (t.sens * 100).toFixed(1) + '% · Spec: ' + (t.spec * 100).toFixed(1) + '%' +
            '<br/>PPV: ' + t.ppv.toFixed(1) + '% · NPV: ' + t.npv.toFixed(1) + '%');
        })
        .on('mousemove', CancerCharts.moveTooltip)
        .on('mouseout', CancerCharts.hideTooltip);

      g.append('text')
        .attr('x', x(t.ppv) + 4).attr('y', yOff + 2 + barH / 2 + 4)
        .attr('fill', c.green).attr('font-size', 10).attr('font-weight', 600)
        .text(t.ppv.toFixed(1) + '%');

      // NPV bar (bottom)
      g.append('rect')
        .attr('x', 0).attr('y', yOff + 2 + barH + 2)
        .attr('width', x(t.npv)).attr('height', barH)
        .attr('fill', c.blue).attr('fill-opacity', 0.8).attr('rx', 2)
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          CancerCharts.showTooltip(event,
            '<strong>' + t.name + '</strong>' +
            '<br/>Sens: ' + (t.sens * 100).toFixed(1) + '% · Spec: ' + (t.spec * 100).toFixed(1) + '%' +
            '<br/>PPV: ' + t.ppv.toFixed(1) + '% · NPV: ' + t.npv.toFixed(1) + '%');
        })
        .on('mousemove', CancerCharts.moveTooltip)
        .on('mouseout', CancerCharts.hideTooltip);

      g.append('text')
        .attr('x', x(t.npv) + 4).attr('y', yOff + 2 + barH + 2 + barH / 2 + 4)
        .attr('fill', c.blue).attr('font-size', 10).attr('font-weight', 600)
        .text(t.npv.toFixed(1) + '%');
    });

    // Title
    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('MRD Test Landscape: PPV & NPV');

    // Legend
    var lgG = svg.append('g')
      .attr('transform', 'translate(' + (margin.left + innerW / 2) + ',' + (height - 6) + ')');
    lgG.append('rect').attr('x', -60).attr('y', -8).attr('width', 10).attr('height', 10)
      .attr('fill', c.green).attr('fill-opacity', 0.8).attr('rx', 2);
    lgG.append('text').attr('x', -46).attr('y', 1)
      .attr('fill', c.text).attr('font-size', 11).text('PPV');
    lgG.append('rect').attr('x', 10).attr('y', -8).attr('width', 10).attr('height', 10)
      .attr('fill', c.blue).attr('fill-opacity', 0.8).attr('rx', 2);
    lgG.append('text').attr('x', 24).attr('y', 1)
      .attr('fill', c.text).attr('font-size', 11).text('NPV');
  }

  // ============================================================
  // Chart 7: Serial Negative Updating
  // ============================================================

  function drawSerialNeg() {
    var c = CancerCharts.getColors();
    var container = d3.select('#serial-neg');
    container.selectAll('*').remove();

    var margin = { top: 35, right: 60, bottom: 40, left: 120 };
    var width = 700, height = 320;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('width', '100%')
      .style('font-family', 'inherit');

    svg.append('text').attr('x', margin.left + innerW / 2).attr('y', 18)
      .attr('text-anchor', 'middle').attr('fill', c.text)
      .attr('font-size', 15).attr('font-weight', 600)
      .text('Serial Negative Tests: Residual Disease Probability');

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var labels = ['Prior', 'After Test 1 (neg)', 'After Test 2 (neg)',
                  'After Test 3 (neg)', 'After Test 4 (neg)'];
    var y = d3.scaleBand().domain(labels).range([0, innerH]).padding(0.2);
    var x = d3.scaleLinear().domain([0, 100]).range([0, innerW]);

    // Grid
    g.append('g').selectAll('line')
      .data(x.ticks(5)).enter().append('line')
      .attr('x1', function(d) { return x(d); })
      .attr('x2', function(d) { return x(d); })
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', c.grid).attr('stroke-dasharray', '3,3');

    // X axis
    g.append('g').attr('transform', 'translate(0,' + innerH + ')')
      .call(d3.axisBottom(x).ticks(5).tickFormat(function(d) { return d + '%'; }))
      .call(function(g) {
        g.selectAll('text').attr('fill', c.text);
        g.selectAll('line,path').attr('stroke', c.muted);
      });

    // Row labels
    labels.forEach(function(label) {
      g.append('text')
        .attr('x', -10)
        .attr('y', y(label) + y.bandwidth() / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', c.text)
        .attr('font-size', 12)
        .text(label);
    });

    // Bars
    var barColors = [c.pink, c.yellow, c.green, c.blue, c.purple];
    labels.forEach(function(label, i) {
      g.append('rect')
        .attr('class', 'sn-bar-' + i)
        .attr('x', 0)
        .attr('y', y(label))
        .attr('height', y.bandwidth())
        .attr('width', 0)
        .attr('fill', barColors[i])
        .attr('fill-opacity', 0.8)
        .attr('rx', 3);

      g.append('text')
        .attr('class', 'sn-label-' + i)
        .attr('x', 0)
        .attr('y', y(label) + y.bandwidth() / 2 + 5)
        .attr('fill', c.text)
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text('');
    });

    container.node().__snG = g;
    container.node().__snX = x;

    updateSerialNeg();
  }

  function updateSerialNeg() {
    var container = d3.select('#serial-neg');
    var node = container.node();
    if (!node || !node.__snG) return;

    var g = node.__snG;
    var x = node.__snX;

    var prior = sliderVal('sn-prev') / 100;
    var sens = sliderVal('sn-sens') / 100;
    var spec = sliderVal('sn-spec') / 100;

    var probs = [prior];
    for (var i = 0; i < 4; i++) {
      probs.push(negUpdate(probs[i], sens, spec));
    }

    probs.forEach(function(p, i) {
      var pPct = p * 100;
      g.select('.sn-bar-' + i)
        .transition().duration(300)
        .attr('width', x(pPct));

      g.select('.sn-label-' + i)
        .transition().duration(300)
        .attr('x', x(pPct) + 6)
        .tween('text', function() {
          var that = d3.select(this);
          var current = parseFloat(that.text()) || 0;
          var interp = d3.interpolateNumber(current, pPct);
          return function(t) {
            var val = interp(t);
            that.text(val < 0.01 ? '<0.01%' : val < 0.1 ? val.toFixed(3) + '%' : val.toFixed(1) + '%');
          };
        });
    });

    setDisplay('sn-prev-val', sliderVal('sn-prev') + '%');
    setDisplay('sn-sens-val', sliderVal('sn-sens') + '%');
    setDisplay('sn-spec-val', sliderVal('sn-spec') + '%');
  }

  // ============================================================
  // Init, Sliders, Theme Change
  // ============================================================

  function drawAll() {
    drawDotGrid();
    drawPPVCurves();
    drawBayesUpdate();
    drawRealityCheck();
    drawMrdDual();
    drawMrdLandscape();
    drawSerialNeg();
  }

  function wireSliders() {
    ['dot-prev', 'dot-sens', 'dot-spec'].forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener('input', function() { updateDotGrid(true); });
    });

    var ppvEl = document.getElementById('ppv-sens');
    if (ppvEl) ppvEl.addEventListener('input', updatePPVCurves);

    ['bu-prev', 'bu-sens', 'bu-spec'].forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener('input', updateBayesUpdate);
    });

    ['mrd-prev', 'mrd-sens', 'mrd-spec'].forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener('input', updateMrdDual);
    });
    wireMrdPresets();

    var mrdLandEl = document.getElementById('mrd-land-prev');
    if (mrdLandEl) mrdLandEl.addEventListener('input', drawMrdLandscape);

    ['sn-prev', 'sn-sens', 'sn-spec'].forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener('input', updateSerialNeg);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      drawAll();
      wireSliders();
    });
  } else {
    drawAll();
    wireSliders();
  }

  window.addEventListener('themechange', drawAll);
})();
</script>
