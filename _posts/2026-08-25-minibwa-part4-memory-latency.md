---
layout: post
title: "How minibwa Works, Part 4: Hiding Memory Latency"
date: 2026-08-25 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

The most important performance fact in minibwa is not a clever alignment trick. It is that RAM is slow.

<!--more-->

A backward-search step in the FM-index does a tiny amount of arithmetic, then jumps to a location in a multi-gigabyte index. The arithmetic is cheap. The jump is not. If the CPU has to wait for that memory load before doing the next step, the core spends most of its time idle. minibwa's main win is keeping enough independent searches in flight that those waits overlap.


<div class="minibwa-series">
<h2>The problem: the index doesn&rsquo;t fit in cache</h2>
  <p>
    Recall from Part 2 that each backward-search step lands on an
    unpredictable, far-apart row of the BWT. The human index is about 3 GB
    &mdash; hundreds of times larger than the CPU&rsquo;s last-level cache. So
    almost every step is a <b>cache miss</b>: the CPU asks RAM for a byte and
    then <em>stalls for a couple hundred cycles</em> doing nothing while the
    data crawls back.
  </p>
  <p>
    The actual work &mdash; the <code>C[c] + Occ(c,&middot;)</code> arithmetic
    &mdash; takes only a handful of cycles. So in naive seeding the CPU spends
    the overwhelming majority of its time idle, waiting on memory. Watch a
    single query crawl through its steps:
  </p>

  <div class="widget">
    <div class="wbar">Where the time goes: one query at a time vs. batched</div>
    <div class="wbody">
      <div class="controls">
        <label>batch size (queries in flight)
          <input type="range" id="batch" min="1" max="8" value="4">
          <b id="batch-v" style="font-family:var(--mono)">4</b></label>
        <button class="btn" id="run">Run ▶</button>
      </div>
      <svg id="gantt" viewBox="0 0 700 300"></svg>
      <div id="gantt-info" class="note" style="margin-top:4px"></div>
      <p class="hint">
        <span style="display:inline-block;width:11px;height:11px;background:var(--stall);border-radius:2px;vertical-align:middle"></span>
        CPU stalled, waiting on RAM &nbsp;&nbsp;
        <span style="display:inline-block;width:11px;height:11px;background:var(--bwa);border-radius:2px;vertical-align:middle"></span>
        actual compute. Simplified model: latency = 12 units, compute = 4, 4 steps per query.
      </p>
    </div>
  </div>

  <h2>The fix: batch many queries and prefetch</h2>
  <p>
    The queries are independent &mdash; read A&rsquo;s seeding has nothing to do
    with read B&rsquo;s. So instead of finishing one query before starting the
    next, minibwa keeps a <b>batch</b> of queries in flight. The trick has two
    halves working together:
  </p>
  <ul>
    <li><b>Prefetch.</b> Before the CPU will need a BWT block, it issues a
      <code>__builtin_prefetch</code> hint telling the hardware to start pulling
      that block into cache <em>now</em>.</li>
    <li><b>Interleave.</b> Then, instead of stalling, the CPU switches to a
      different query in the batch and does <em>its</em> compute. By the time it
      cycles back, the prefetched data has arrived. The wait is hidden behind
      useful work.</li>
  </ul>
  <p>
    Drag the batch slider above past 3 and the gray stall bars vanish &mdash;
    the compute blocks tile densely because each query&rsquo;s memory wait is
    filled by its neighbors&rsquo; arithmetic. This is the whole game. The
    real <code>bwt.c</code> is littered with it:
  </p>

  <div class="note" style="font-family:var(--mono); font-size:13px; white-space:pre-wrap; line-height:1.4">
<span style="color:var(--ink-soft)">// bwt.c — prefetch the BWT blocks the NEXT iteration will touch</span>
mb_bwt_block_prefetch(bwt, s-&gt;p.x[1]);
mb_bwt_block_prefetch(bwt, s-&gt;p.x[1] + s-&gt;p.size);
<span style="color:var(--ink-soft)">// ...then move on to another query in the batch instead of stalling</span>
  </div>

  <p>
    minibwa applies this everywhere a random memory access hides: SMEM finding
    (<code>mb_bwt_smem_batch</code>, ~<b class="is-new">2.5&times;</b> faster),
    suffix-array lookups (<code>mb_bwt_sa_batch</code>, over
    <b class="is-new">4&times;</b> faster), even across <em>both</em> seeding
    passes from Part 3 (<code>mb_seed_intv_batch</code>). bwa-mem
    couldn&rsquo;t do the two-pass-with-prefetch trick because its older SMEM
    algorithm wasn&rsquo;t structured for it; minibwa reimplemented SMEM finding
    (borrowing Travis Gagie&rsquo;s formulation from ropebwt3) specifically
    <em>because</em> the simpler structure makes batching possible.
  </p>

  <div class="note key">
    <span class="label">The reframe that matters</span>
    The algorithm got <em>simpler</em>, not smarter, and that&rsquo;s the point.
    A simpler loop with no data-dependent control flow is one you can software-
    pipeline: issue the loads early, interleave independent work, never stall.
    On modern hardware, &ldquo;don&rsquo;t wait for RAM&rdquo; beats &ldquo;do
    fewer operations&rdquo; almost every time.
  </div>

  <h2>One more trick: the 10-mer cache</h2>
  <p>
    The first few backward-search steps of <em>every</em> seed are the same kind
    of work on the same tiny set of short prefixes. So minibwa precomputes the
    SA interval for all 4<sup>10</sup> possible 10-mers once
    (<code>mb_bwt_cache</code>) and stores them in a flat table. A seed then
    skips its first 10 random-memory backward steps and starts from a single
    cache lookup. Ten unpredictable RAM jumps collapse into one:
  </p>

  <div class="widget">
    <div class="wbar">10-mer cache: skip the first ten random jumps</div>
    <div class="wbody">
      <svg id="cacheviz" viewBox="0 0 700 150"></svg>
      <button class="btn ghost" id="cache-toggle">Toggle cache on/off</button>
      <div id="cache-info" class="note" style="margin-top:10px"></div>
    </div>
  </div>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;The index is too big for cache, so every seeding step would stall the
    CPU waiting on RAM. minibwa keeps many reads in flight at once and prefetches
    &mdash; so while one read waits for memory, the CPU does another read&rsquo;s
    arithmetic. The waiting disappears, and that&rsquo;s where the speed comes
    from.&rdquo;
  </div>

<p>This is the core idea I wanted to understand when I started reading minibwa: the algorithm did not magically stop needing memory. It changed the shape of the loop so memory waits could be hidden behind other useful work. Once you see that pattern, the rest of the aligner starts to look like variations on the same move.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
const NS="http://www.w3.org/2000/svg";
const L=12, C=4, STEPS=4; // latency, compute, steps per query

/* ---- gantt ---- */
(function(){
  const svg=document.getElementById("gantt");
  const info=document.getElementById("gantt-info");
  const batch=document.getElementById("batch"), batchv=document.getElementById("batch-v");
  const SCALE = 5;  // px per time unit
  function draw(animate){
    const B=+batch.value; batchv.textContent=B;
    svg.innerHTML="";
    // --- serial timeline (top): show B queries done one after another, single lane ---
    const serialTime = B*STEPS*(L+C);
    // --- batched timeline (bottom): round-robin, compute-bound when B*C>=L ---
    const hidden = B*C >= L;
    let g="";
    g += `<text x="0" y="14" font-family="var(--sans)" font-size="13" font-weight="700" fill="var(--ink)">One at a time (naive)</text>`;
    // serial: lay out per query, wrapping not needed since we scale to fit
    let serScale = 690 / serialTime;
    let t=0;
    for (let q=0;q<B;q++){
      for (let s=0;s<STEPS;s++){
        g += `<rect x="${t*serScale}" y="22" width="${L*serScale}" height="20" fill="var(--stall)"/>`;
        t+=L;
        g += `<rect x="${t*serScale}" y="22" width="${Math.max(1,C*serScale)}" height="20" fill="var(--bwa)"/>`;
        t+=C;
      }
    }
    g += `<text x="0" y="60" font-family="var(--mono)" font-size="11" fill="var(--ink-soft)">${serialTime} time units — ${Math.round(100*B*STEPS*C/serialTime)}% useful work</text>`;

    // batched: B lanes, compute tiles in round-robin order k=s*B+i at start k*C
    g += `<text x="0" y="110" font-family="var(--sans)" font-size="13" font-weight="700" fill="var(--ink)">Batched + prefetch (minibwa)</text>`;
    const batTime = STEPS*B*C + (hidden?L:0); // fill latency once
    let batScale = 690 / Math.max(batTime, serialTime*0.001);
    // draw faint "in-flight memory" underlay per lane
    for (let i=0;i<B;i++){
      const ly = 120 + i* (Math.min(20, 150/B));
      const lh = Math.min(16, 150/B - 2);
      for (let s=0;s<STEPS;s++){
        const k = s*B+i;
        const x = (L + k*C)*batScale;       // +L: first fill
        g += `<rect x="${x}" y="${ly}" width="${Math.max(1.5,C*batScale)}" height="${lh}" fill="var(--bwa)" opacity="0.9"/>`;
      }
      g += `<text x="-2" y="${ly+lh-2}" font-family="var(--mono)" font-size="9" fill="var(--ink-soft)" text-anchor="end"></text>`;
    }
    // initial fill region (the one unavoidable wait)
    g += `<rect x="0" y="120" width="${L*batScale}" height="${Math.min(20,150/B)*B}" fill="var(--stall-soft)"/>`;
    g += `<text x="${L*batScale+4}" y="132" font-family="var(--sans)" font-size="10" fill="var(--ink-soft)">one fill, then no stalls</text>`;
    const speedup = serialTime / batTime;
    g += `<text x="0" y="295" font-family="var(--mono)" font-size="11" fill="var(--ink-soft)">${batTime} time units — ${hidden?"latency fully hidden":"latency partly hidden"} — <tspan font-weight="700" fill="var(--new)">${speedup.toFixed(1)}× faster</tspan></text>`;
    svg.innerHTML=g;
    info.innerHTML = `<span class="label">batch = ${B}</span>` + (hidden
      ? `With ${B} queries in flight, the other ${B-1} queries cover each ${L}-unit memory wait. After the first fill, the CPU stays busy doing arithmetic. <b>${speedup.toFixed(1)}× faster</b> than one-at-a-time here.`
      : `With only ${B} in flight, there isn't enough independent compute to cover a ${L}-unit wait, so some stalling remains. Push the batch higher (you need batch·compute ≥ latency, i.e. ≥ ${Math.ceil(L/C)}) to hide it completely.`);
  }
  batch.oninput=()=>draw(false);
  document.getElementById("run").onclick=()=>draw(true);
  draw(false);
})();

/* ---- 10-mer cache ---- */
(function(){
  const svg=document.getElementById("cacheviz");
  const info=document.getElementById("cache-info");
  let on=true;
  function draw(){
    let g=`<text x="0" y="16" font-family="var(--sans)" font-size="13" font-weight="700" fill="var(--ink)">Backward search, first 10 letters of a seed</text>`;
    const y=60, x0=10, dx=66;
    for (let i=0;i<10;i++){
      const skipped = on && i<10;
      const col = skipped? "var(--stall)" : "var(--accent)";
      const op = skipped? 0.35 : 1;
      g += `<circle cx="${x0+i*dx}" cy="${y}" r="10" fill="${col}" opacity="${op}"/>`;
      if (i>0) g += `<line x1="${x0+(i-1)*dx+10}" y1="${y}" x2="${x0+i*dx-10}" y2="${y}" stroke="${skipped?'var(--stall)':'var(--accent)'}" stroke-width="2" opacity="${op}" stroke-dasharray="${skipped?'3 3':'0'}"/>`;
    }
    if (on){
      g += `<rect x="${x0-12}" y="${y+24}" width="${9*dx+24}" height="34" rx="8" fill="var(--new)" opacity="0.15" stroke="var(--new)"/>`;
      g += `<text x="${x0+4.5*dx}" y="${y+45}" text-anchor="middle" font-family="var(--sans)" font-size="13" font-weight="700" fill="var(--new)">1 cache lookup replaces all 10 random RAM jumps</text>`;
    } else {
      g += `<text x="${x0+4.5*dx}" y="${y+45}" text-anchor="middle" font-family="var(--sans)" font-size="13" font-weight="700" fill="var(--accent)">10 separate random RAM accesses</text>`;
    }
    svg.innerHTML=g;
    info.innerHTML = on
      ? `<span class="label">cache ON</span>The 10-mer's interval is read from a precomputed flat table — one access, and likely already hot in cache. The remaining steps of the seed proceed from there. This is why the simpler SMEM algorithm matters: the old one couldn't start mid-search from a cached interval.`
      : `<span class="label">cache OFF</span>Without the cache, the seed pays ten consecutive cache-missing memory accesses just to get going — ten stalls before it even reaches the interesting part of the read.`;
  }
  document.getElementById("cache-toggle").onclick=()=>{on=!on;draw();};
  draw();
})();
</script>
