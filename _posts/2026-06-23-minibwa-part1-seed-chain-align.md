---
layout: post
title: "How minibwa Works, Part 1: Seed, Chain, Align"
date: 2026-06-23 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

A 150-base read shows up from the sequencer. Somewhere in the 3.1-billion-base human reference, there is probably a place it came from. Maybe it has a sequencing error. Maybe the sample has a real SNP or small indel. Maybe it came from a repeat where half the genome looks suspiciously similar.

<!--more-->

The dumb solution is to align that read against the whole genome and pick the best spot. That sounds honest, and it is completely impossible. This first post is about the trick every fast short-read aligner uses instead: find cheap exact matches, chain the ones that make geometric sense, and only then pay for real dynamic programming.


<div class="minibwa-series">
<h2>The job</h2>
  <p>
    You have a few hundred million reads, each 100&ndash;250 bp, sequenced with
    errors and from a genome that differs from your reference by SNPs and
    indels. For each read you want the position in the reference it most likely
    came from, an alignment (the CIGAR), and a confidence (the mapping quality).
    That&rsquo;s mapping.
  </p>

  <h2>Why you can&rsquo;t just align everything</h2>
  <p>
    The textbook answer to &ldquo;where does this string best fit in that
    string&rdquo; is Smith&ndash;Waterman: a dynamic-programming table of size
    <code>read length &times; reference length</code>. For one human read that
    is one row of 100&ndash;odd cells across all 3.1 billion reference bases.
    Multiply by hundreds of millions of reads and the arithmetic is absurd. Turn
    the knobs and watch:
  </p>

  <div class="widget">
    <div class="wbar">The cost of brute force vs. seed-and-extend</div>
    <div class="wbody">
      <div class="controls">
        <label>Reads (millions)
          <input type="range" id="nreads" min="1" max="800" value="400">
          <b id="nreads-v" style="font-family:var(--mono)">400</b></label>
        <label>Read length
          <input type="range" id="rlen" min="50" max="300" value="150" step="10">
          <b id="rlen-v" style="font-family:var(--mono)">150</b></label>
      </div>
      <svg id="costviz" viewBox="0 0 680 150"></svg>
      <p class="hint" id="cost-text"></p>
    </div>
  </div>

  <p>
    The bar for brute force runs off the screen because it <em>is</em> off the
    screen &mdash; it&rsquo;s thousands of CPU-years. The trick that makes
    mapping possible is to never run DP against the whole genome. Instead:
  </p>

  <h2>The skeleton: seed, chain, align</h2>
  <p>
    Almost every modern aligner is the same three moves. Step through them:
  </p>

  <div class="widget">
    <div class="wbar">Seed &rarr; chain &rarr; align &mdash; step through it</div>
    <div class="wbody">
      <svg id="scaviz" viewBox="0 0 680 230"></svg>
      <div class="controls" style="margin-top:12px">
        <button class="btn" id="sca-step">Next step &rarr;</button>
        <button class="btn ghost" id="sca-reset">Reset</button>
      </div>
      <div id="sca-cap" class="note" style="margin-top:4px"></div>
    </div>
  </div>

  <p>
    <span class="from-bwa">Seeding</span> finds short stretches that match the
    reference <em>exactly</em>. Exact matching is cheap because we can look it up
    in an index instead of computing DP &mdash; that&rsquo;s what the next
    post is about. Each seed says &ldquo;this little piece of the read is
    here, here, and here in the genome.&rdquo;
  </p>
  <p>
    <span class="from-mm2">Chaining</span> takes the scattered seeds and finds a
    set of them that line up &mdash; same diagonal, increasing in both read and
    reference coordinates. A good colinear chain is a strong hypothesis for
    where the read belongs. Most of the genome gets no chain at all, so it never
    costs us anything.
  </p>
  <p>
    <span class="from-mm2">Alignment</span> only now runs Smith&ndash;Waterman
    &mdash; and only in the small gaps <em>between</em> chained seeds, over a few
    hundred bases, not three billion. This is where the CIGAR and the exact
    base-level differences come from.
  </p>

  <div class="note">
    <span class="label">The whole point</span>
    Exact matching is something an index can do in microseconds. Approximate
    matching (DP) is expensive. So we use cheap exact matching to throw away
    99.999% of the genome, and pay for expensive DP only on the tiny survivors.
  </div>

  <h2>Where the three tools differ</h2>
  <p>
    Given that everyone uses the same skeleton, what distinguishes the tools is
    <em>which engine</em> drives each step and <em>how fast</em> those engines
    run. minibwa&rsquo;s thesis is that bwa-mem had the best seeding idea and
    minimap2 had the best chaining and alignment code &mdash; so take both, and
    then make the seeding engine scream. The rest of this series is that
    sentence, unpacked.
  </p>

  <table class="data">
    <tr><th>Step</th><th>bwa-mem</th><th>minimap2</th><th>minibwa</th></tr>
    <tr><td>Index</td><td>FM-index (BWT)</td><td>minimizer hash</td><td class="num"><span class="tag bwa">FM-index</span></td></tr>
    <tr><td>Seeds</td><td>SMEMs (exact, variable length)</td><td>minimizers (fixed k-mers)</td><td class="num"><span class="tag bwa">SMEMs</span></td></tr>
    <tr><td>Chaining</td><td>tree-based, heuristic</td><td>colinear DP</td><td class="num"><span class="tag mm2">colinear DP</span></td></tr>
    <tr><td>Alignment</td><td>own extension code</td><td>ksw2 SIMD</td><td class="num"><span class="tag mm2">ksw2 SIMD</span></td></tr>
  </table>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;Mapping is impossible by brute force, so we seed with cheap exact
    matches, chain the ones that line up, and run real alignment only in the
    gaps. minibwa keeps bwa-mem&rsquo;s seeds and minimap2&rsquo;s
    chain-and-align.&rdquo;
  </div>

<p>That seed-chain-align shape is the backbone for the whole series. The next question is the obvious one: how do we find exact matches in a 3 GB reference quickly enough to do this hundreds of millions of times?</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
const NS = "http://www.w3.org/2000/svg";

/* ---- cost widget ---- */
(function(){
  const svg = document.getElementById("costviz");
  const nr = document.getElementById("nreads"), rl = document.getElementById("rlen");
  const nrv = document.getElementById("nreads-v"), rlv = document.getElementById("rlen-v");
  const txt = document.getElementById("cost-text");
  const G = 3.1e9, CELLS_PER_NS = 1e9; // ~1 cell/ns optimistic
  function fmt(s){
    if (s < 60) return s.toFixed(1)+" s";
    if (s < 3600) return (s/60).toFixed(1)+" min";
    if (s < 86400) return (s/3600).toFixed(1)+" h";
    if (s < 3.15e7) return (s/86400).toFixed(1)+" days";
    return (s/3.15e7).toFixed(0)+" years";
  }
  function draw(){
    const n = +nr.value * 1e6, L = +rl.value;
    nrv.textContent = nr.value; rlv.textContent = rl.value;
    // brute: n reads * L * G cells; seed: assume ~1000 cells of DP per read
    const brute = n * L * G / CELLS_PER_NS;
    const seed  = n * 1000 / CELLS_PER_NS;
    svg.innerHTML = "";
    const rows = [["Brute-force DP (read × genome)", brute, "var(--accent)"],
                  ["Seed-and-extend (DP only in gaps)", seed, "var(--mm2)"]];
    // log scale bar
    const maxlog = Math.log10(brute), minref = Math.log10(seed);
    rows.forEach((r,i)=>{
      const y = 20 + i*64;
      const frac = Math.max(.02, Math.log10(r[1]) / maxlog);
      svg.innerHTML += `
        <text x="0" y="${y}" font-family="var(--sans)" font-size="13" font-weight="700">${r[0]}</text>
        <rect x="0" y="${y+8}" width="${frac*680}" height="22" rx="4" fill="${r[2]}" opacity="0.85"/>
        <text x="${frac*680+8}" y="${y+24}" font-family="var(--mono)" font-size="13" fill="var(--ink-soft)">${fmt(r[1])}</text>`;
    });
    txt.innerHTML = `One human-scale dataset. Brute force: <b>${fmt(brute)}</b> of pure DP. Seed-and-extend pays DP on only the survivors: <b>${fmt(seed)}</b>. (bars are log-scaled, or the top one would be kilometers long.)`;
  }
  [nr,rl].forEach(e=>e.addEventListener("input",draw)); draw();
})();

/* ---- seed-chain-align stepper ---- */
(function(){
  const svg = document.getElementById("scaviz");
  const cap = document.getElementById("sca-cap");
  const REF_Y = 150, READ_Y = 40, X0 = 40, REFW = 600;
  // seed positions: [readFrac, refFrac, len]
  const seeds = [[.05,.10,.12],[.30,.34,.10],[.55,.58,.13],[.80,.83,.09],
                 [.40,.05,.06],[.62,.90,.05]]; // last two are decoys off-diagonal
  const steps = [
    "We start with a read (top) and the reference (bottom). We don't know yet where the read goes.",
    "Seeding: exact-match seeds light up. Some are real (on one diagonal), some are spurious matches elsewhere.",
    "Chaining: we keep the seeds that line up colinearly — same diagonal, increasing in both axes. The decoys are dropped.",
    "Alignment: DP fills only the short gaps between chained seeds (hatched). That's the only expensive work, and it's tiny."
  ];
  let step = 0;
  function draw(){
    svg.innerHTML = "";
    // read + ref bars
    svg.innerHTML += `<rect x="${X0}" y="${READ_Y}" width="${REFW}" height="16" rx="4" fill="var(--paper-2)" stroke="var(--rule-2)"/>
      <text x="${X0}" y="${READ_Y-8}" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">read</text>
      <rect x="${X0}" y="${REF_Y}" width="${REFW}" height="16" rx="4" fill="var(--paper-2)" stroke="var(--rule-2)"/>
      <text x="${X0}" y="${REF_Y-8}" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">reference</text>`;
    seeds.forEach((s,i)=>{
      const onDiag = i < 4;
      const rx = X0 + s[0]*REFW, rxw = s[2]*REFW;
      const fx = X0 + s[1]*REFW, fxw = s[2]*REFW;
      if (step === 0) return;
      const dropped = step >= 2 && !onDiag;
      const col = dropped ? "var(--rule-2)" : (onDiag ? "var(--bwa)" : "var(--new)");
      const op = dropped ? 0.3 : 0.85;
      svg.innerHTML += `<rect x="${rx}" y="${READ_Y}" width="${rxw}" height="16" rx="3" fill="${col}" opacity="${op}"/>
        <rect x="${fx}" y="${REF_Y}" width="${fxw}" height="16" rx="3" fill="${col}" opacity="${op}"/>`;
      if (step >= 2 && !dropped)
        svg.innerHTML += `<line x1="${rx+rxw/2}" y1="${READ_Y+16}" x2="${fx+fxw/2}" y2="${REF_Y}" stroke="${col}" stroke-width="1.5" opacity="0.5"/>`;
    });
    if (step >= 3){
      // hatched gaps between consecutive on-diagonal seeds (read axis)
      const real = seeds.slice(0,4).sort((a,b)=>a[0]-b[0]);
      for (let i=0;i<real.length-1;i++){
        const x = X0 + (real[i][0]+real[i][2])*REFW;
        const w = (real[i+1][0] - real[i][0]-real[i][2])*REFW;
        if (w>1) svg.innerHTML += `<rect x="${x}" y="${READ_Y-3}" width="${w}" height="22" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="3 2"/>`;
      }
    }
    cap.innerHTML = `<span class="label">Step ${step+1} of ${steps.length}</span>${steps[step]}`;
  }
  document.getElementById("sca-step").onclick = ()=>{ step=(step+1)%steps.length; draw(); };
  document.getElementById("sca-reset").onclick = ()=>{ step=0; draw(); };
  draw();
})();
</script>
