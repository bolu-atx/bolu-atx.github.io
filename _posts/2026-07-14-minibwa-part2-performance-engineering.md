---
layout: post
title: "How minibwa Works, Part 2: Performance Engineering"
date: 2026-07-14 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

The most important performance fact in minibwa is not a clever alignment trick. It is that RAM is slow, and the FM-index makes you jump around RAM constantly.

<!--more-->

Once you see that, the engineering choices line up: batch independent searches, prefetch what the next step will need, cache the common prefixes, and run cheap tests before expensive dynamic programming.

<div class="minibwa-series">
<h2>The problem: the index doesn&rsquo;t fit in cache</h2>
  <p>
    Recall from the FM-index discussion that each backward-search step lands on an
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
          <input type="range" id="perf-lat-batch" min="1" max="8" value="4">
          <b id="perf-lat-batch-v" style="font-family:var(--mono)">4</b></label>
        <button class="btn" id="perf-lat-run">Run ▶</button>
      </div>
      <svg id="perf-lat-gantt" viewBox="0 0 700 300"></svg>
      <div id="perf-lat-gantt-info" class="note" style="margin-top:4px"></div>
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
    passes (<code>mb_seed_intv_batch</code>). bwa-mem
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
      <svg id="perf-cache-viz" viewBox="0 0 700 150"></svg>
      <button class="btn ghost" id="perf-cache-toggle">Toggle cache on/off</button>
      <div id="perf-cache-info" class="note" style="margin-top:10px"></div>
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

<h2>DP only in the gaps &mdash; and only when needed</h2>
  <p>
    The chain already pins down most of the read with exact seeds. Real DP runs
    only in the short stretches <em>between</em> chained seeds and at the read
    ends, using minimap2&rsquo;s <span class="from-mm2">ksw2</span> routines
    &mdash; vectorized Smith&ndash;Waterman with a dual affine-gap model,
    hand-written in SSE4.1 on x86 and NEON on ARM. (The authors tried AVX2 and
    saw no clear improvement, so they didn&rsquo;t bother.) Even so, this DP is
    about <b>35% of total CPU time</b> on whole-genome short reads &mdash; the
    single biggest cost after seeding.
  </p>
  <p>
    So minibwa adds a fast path: it first attempts an
    <span class="is-new">ungapped</span> alignment of the gap. If that comes out
    clean &mdash; few enough mismatches &mdash; it accepts it and skips the DP
    entirely. Only when the ungapped attempt looks bad (suggesting a real indel)
    does it fall back to full SIMD Smith&ndash;Waterman. Most gaps in most reads
    have no indel, so the fast path fires constantly.
  </p>

  <div class="note">
    <span class="label">The shape of the work</span>
    Exact seeds (free, from the index) cover most of the read &rarr; ungapped
    check (cheap) handles the clean gaps &rarr; SIMD DP (expensive) runs only on
    the few gaps that actually contain an indel. Each tier is an order of
    magnitude rarer and costlier than the last. That layering is the whole
    performance philosophy of the aligner in miniature.
  </div>

  <h2>Pairing, and the mate-rescue trap</h2>
  <p>
    Short reads usually come in pairs from the two ends of a fragment, so the
    two mates should land close together in the correct orientation. minibwa
    pairs them with <span class="from-bwa">bwa-mem&rsquo;s logic</span>. The
    expensive part is <b>mate rescue</b>: when one mate maps confidently but the
    other doesn&rsquo;t, you search the region near the mapped mate for the
    missing one &mdash; running Smith&ndash;Waterman over that whole window.
  </p>
  <p>
    The trap: most of the time the missing mate genuinely isn&rsquo;t there (it
    was junk, or off in a repeat), and you&rsquo;ve burned a full DP for
    nothing. minibwa adds a <span class="is-new">pre-alignment filter</span>
    (<code>mb_ungap</code>) to predict, cheaply, whether a real alignment even
    exists before paying for DP.
  </p>

  <h2>The filter: voting on a diagonal, Hough-style</h2>
  <p>
    The idea is borrowed from the Hough transform for finding lines. If the
    missing mate really sits in this window, then many short <code>q</code>-mers
    (minibwa uses <code>q&nbsp;=&nbsp;7</code>) of the read will match the
    reference at the <em>same</em> diagonal offset &mdash; they all &ldquo;vote&rdquo;
    for the same line. If the read isn&rsquo;t really there, its q-mer matches
    scatter randomly across offsets and no offset gets many votes.
  </p>
  <p>
    So: slide the read across the window and tally how many q-mers match at
    each offset. If the best offset gets at least 10 votes, run the full
    Smith&ndash;Waterman. Otherwise, skip it &mdash; there&rsquo;s no line to
    find. Add mismatches to the read below and watch the matching q-mers on the
    true diagonal disappear:
  </p>

  <div class="widget">
    <div class="wbar">Mate-rescue pre-filter: do the votes clear the bar?</div>
    <div class="wbody">
      <div class="controls">
        <label>mismatches in read
          <input type="range" id="perf-rescue-mm" min="0" max="8" value="2">
          <b id="perf-rescue-mm-v" style="font-family:var(--mono)">2</b></label>
        <label>q-mer length
          <input type="range" id="perf-rescue-q" min="3" max="6" value="4">
          <b id="perf-rescue-q-v" style="font-family:var(--mono)">4</b></label>
        <span class="hint">toy threshold: 6 votes</span>
      </div>
      <svg id="perf-rescue-votes" viewBox="0 0 700 310"></svg>
      <div id="perf-rescue-verdict" class="note" style="margin-top:4px"></div>
    </div>
  </div>

  <p>
    The real threshold is 10 votes with 7-mers; the widget uses 6 votes with
    shorter q-mers so the idea fits in a toy sequence. The payoff is real: the
    paper notes this filter lets minibwa <em>skip unsuccessful alignments</em>,
    and disabling it is exactly what narrows minibwa&rsquo;s lead over bwa-mem2
    on Hi-C and other data where mate rescue is heavy. A cheap vote replaces an
    expensive DP that was going to fail anyway.
  </p>

  <div class="note key">
    <span class="label">The recurring move</span>
    Notice the pattern for the third time: a cheap test that predicts whether
    the expensive operation will succeed, run <em>before</em> the expensive
    operation. Prefetch hid latency; the ungapped fast path skipped needless DP;
    the Hough filter skips needless mate rescue. Same instinct, three places.
  </div>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;Real alignment runs only in the gaps between seeds, and only when an
    ungapped check says there&rsquo;s an indel worth the DP. For paired reads, a
    cheap 7-mer vote decides whether the missing mate is even findable before we
    spend a full Smith&ndash;Waterman looking for it.&rdquo;
  </div>

<p>This is the performance story in one sentence: minibwa does not magically remove the hard work, but it refuses to pay for it while the CPU is idle or when a cheap test already says the expensive step will fail.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>
<script>
/* minibwa p4 */
(function(){
const L=12, C=4, STEPS=4; // latency, compute, steps per query

/* ---- gantt ---- */
(function(){
  const svg=document.getElementById("perf-lat-gantt");
  const info=document.getElementById("perf-lat-gantt-info");
  const batch=document.getElementById("perf-lat-batch"), batchv=document.getElementById("perf-lat-batch-v");
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
  document.getElementById("perf-lat-run").onclick=()=>draw(true);
  draw(false);
})();

/* ---- 10-mer cache ---- */
(function(){
  const svg=document.getElementById("perf-cache-viz");
  const info=document.getElementById("perf-cache-info");
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
  document.getElementById("perf-cache-toggle").onclick=()=>{on=!on;draw();};
  draw();
})();
})();
</script>

<script>
/* minibwa p6 */
(function(){
const REFW = "TGACCAGTCATTGCAGAACGTTGACCTAGGTCAACTGAC";
let readSeq = REFW.slice(6, 6+20); // true placement at offset 6
let mmPos = [];
const TRUE_OFF = 6, RLEN = 20, THR = 6;

function applyMM(){
  const k=+document.getElementById("perf-rescue-mm").value;
  // deterministic mismatch positions spread across read
  mmPos=[]; for (let i=0;i<k;i++) mmPos.push(Math.floor((i+0.5)*RLEN/k));
  const orig=REFW.slice(TRUE_OFF,TRUE_OFF+RLEN).split("");
  readSeq=orig.map((b,i)=> mmPos.includes(i)? ({A:"T",T:"A",C:"G",G:"C"}[b]) : b).join("");
  render();
}
function render(){
  const q=+document.getElementById("perf-rescue-q").value;
  document.getElementById("perf-rescue-mm-v").textContent=document.getElementById("perf-rescue-mm").value;
  document.getElementById("perf-rescue-q-v").textContent=q;
  // votes: for each offset t (ref start), count q-mer matches at that diagonal
  const maxOff=REFW.length-readSeq.length;
  const votes=[];
  const matchingStarts=[];
  for (let t=0;t<=maxOff;t++){
    let v=0;
    for (let i=0;i+q<=readSeq.length;i++){
      if (readSeq.substr(i,q)===REFW.substr(t+i,q)){
        v++;
        if (t===TRUE_OFF) matchingStarts.push(i);
      }
    }
    votes.push(v);
  }
  const peak=Math.max(...votes), peakAt=votes.indexOf(peak);
  const svg=document.getElementById("perf-rescue-votes");
  const X0=48, cell=14, refY=58, readY=108;
  const histX=36, histY=278, histW=628, histH=90;
  const maxV=Math.max(peak,THR,1);
  const baseColor=b=>({A:"var(--mm2)", C:"var(--bwa)", G:"var(--new)", T:"var(--accent)"}[b] || "var(--ink)");
  let g=`<text x="${X0}" y="18" font-family="var(--sans)" font-size="12" font-weight="700" fill="var(--ink)">Candidate mate placed at the best offset</text>`;
  g+=`<text x="0" y="${refY}" font-family="var(--sans)" font-size="11" fill="var(--ink-soft)">reference</text>`;
  g+=`<text x="0" y="${readY}" font-family="var(--sans)" font-size="11" fill="var(--ink-soft)">read</text>`;

  for (let i=0;i<REFW.length;i++){
    const x=X0+i*cell;
    const inWindow=i>=TRUE_OFF && i<TRUE_OFF+RLEN;
    if (inWindow) g+=`<rect x="${x-2}" y="${refY-14}" width="${cell}" height="22" rx="3" fill="var(--paper-2)" stroke="var(--rule-2)" opacity="0.75"/>`;
    g+=`<text x="${x+cell/2-2}" y="${refY}" text-anchor="middle" font-family="var(--mono)" font-size="13" fill="${baseColor(REFW[i])}">${REFW[i]}</text>`;
  }
  g+=`<line x1="${X0+TRUE_OFF*cell-4}" y1="${refY+11}" x2="${X0+(TRUE_OFF+RLEN)*cell-4}" y2="${refY+11}" stroke="var(--rule-2)" stroke-width="2"/>`;

  for (let i=0;i<readSeq.length;i++){
    const x=X0+(TRUE_OFF+i)*cell;
    const mismatch=mmPos.includes(i);
    const col=mismatch ? "var(--accent)" : baseColor(readSeq[i]);
    g+=`<rect x="${x-2}" y="${readY-15}" width="${cell}" height="23" rx="3" fill="${mismatch?'var(--accent-bg)':'transparent'}" stroke="${mismatch?'var(--accent)':'var(--rule-2)'}" opacity="${mismatch?1:0.55}"/>`;
    g+=`<text x="${x+cell/2-2}" y="${readY}" text-anchor="middle" font-family="var(--mono)" font-size="13" fill="${col}" font-weight="${mismatch?700:400}">${readSeq[i]}</text>`;
  }

  matchingStarts.forEach((i,idx)=>{
    const x=X0+(TRUE_OFF+i)*cell-2, w=q*cell;
    const y=136+(idx%3)*8;
    g+=`<rect x="${x}" y="${y}" width="${w}" height="5" rx="2" fill="var(--mm2)" opacity="0.72"/>`;
  });
  mmPos.forEach(i=>{
    const x=X0+(TRUE_OFF+i)*cell+cell/2-2;
    g+=`<line x1="${x}" y1="${refY+13}" x2="${x}" y2="${readY-18}" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  });
  g+=`<text x="${X0}" y="174" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">green bars = q-mers that still match exactly on this diagonal; red bases break every q-mer crossing them</text>`;

  g+=`<text x="${histX}" y="206" font-family="var(--sans)" font-size="12" font-weight="700" fill="var(--ink)">Votes by possible offset</text>`;
  g+=`<line x1="${histX}" y1="${histY}" x2="${histX+histW}" y2="${histY}" stroke="var(--rule-2)"/>`;
  const bw=histW/votes.length;
  votes.forEach((v,t)=>{
    const h=v/maxV*histH, x=histX+t*bw, y=histY-h;
    const isPeak=t===peakAt;
    const col = isPeak ? "var(--mm2)" : "var(--bwa)";
    g+=`<rect x="${x}" y="${y}" width="${Math.max(2,bw-2)}" height="${h}" fill="${col}" opacity="${isPeak?0.95:0.42}"/>`;
  });
  const ty=histY-THR/maxV*histH;
  g+=`<line x1="${histX}" y1="${ty}" x2="${histX+histW}" y2="${ty}" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="5 3"/>`;
  g+=`<text x="${histX+histW}" y="${ty-5}" text-anchor="end" font-family="var(--sans)" font-size="11" fill="var(--accent)">threshold = ${THR}</text>`;
  g+=`<text x="${histX+peakAt*bw+bw/2}" y="${histY-peak/maxV*histH-6}" text-anchor="middle" font-family="var(--mono)" font-size="12" fill="var(--mm2)" font-weight="700">${peak}</text>`;
  g+=`<text x="${histX+peakAt*bw+bw/2}" y="${histY+17}" text-anchor="middle" font-family="var(--sans)" font-size="10" fill="var(--ink-soft)">best offset ${peakAt}</text>`;
  svg.innerHTML=g;
  const pass=peak>=THR;
  document.getElementById("perf-rescue-verdict").innerHTML = pass
    ? `<span class="label" style="color:var(--mm2)">run Smith-Waterman</span>The best diagonal collected <b>${peak}</b> votes, clearing the toy threshold of ${THR}. Enough q-mers still line up that a real alignment is plausible.`
    : `<span class="label" style="color:var(--accent)">skip Smith-Waterman</span>The best diagonal collected only <b>${peak}</b> votes, below the toy threshold of ${THR}. The exact q-mer evidence has fallen apart, so minibwa skips the expensive DP.`;
}
[
  ["perf-rescue-mm", applyMM],
  ["perf-rescue-q", render]
].forEach(([id, handler])=>document.getElementById(id).addEventListener("input", handler));
applyMM();
})();
</script>
