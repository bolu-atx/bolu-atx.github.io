---
layout: post
title: "How minibwa Works, Part 3: Finding Seeds with SMEMs"
date: 2026-08-04 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

Imagine a read that matches the reference perfectly except for two bases. Those two mismatches chop the read into three exact-match islands.

<!--more-->

Which islands should we keep as seeds? Too short, and they occur everywhere by chance. Too long, and a repeat can hide useful smaller anchors inside it. bwa-mem's answer, which minibwa keeps, is the SMEM: a supermaximal exact match, meaning the longest exact match you cannot extend left or right.


<div class="minibwa-series">
<h2>What makes a seed &ldquo;super-maximal&rdquo;</h2>
  <p>
    A read that truly came from some genomic locus matches it perfectly
    <em>except</em> at the handful of positions where there&rsquo;s a SNP or a
    sequencing error. Those mismatch positions chop the read into runs of exact
    agreement. The longest such runs &mdash; the ones you cannot extend left or
    right without hitting a mismatch, and which aren&rsquo;t contained inside a
    longer match &mdash; are the <b>supermaximal exact matches (SMEMs)</b>.
  </p>
  <p>
    minibwa hunts for <span class="from-bwa">(19,&nbsp;1)-SMEMs</span>: matches
    at least 19 bp long that occur at least once. The length floor of 19 is no
    accident &mdash; shorter than that and a match occurs so often by chance
    that it&rsquo;s useless. Click a base in the read below to introduce a
    mismatch (a SNP or sequencing error) and watch the seeds split:
  </p>

  <div class="widget">
    <div class="wbar">SMEMs: maximal exact runs between mismatches</div>
    <div class="wbody">
      <div class="controls">
        <label>min seed length
          <input type="range" id="minlen" min="3" max="12" value="6">
          <b id="minlen-v" style="font-family:var(--mono)">6</b></label>
        <button class="btn ghost" id="clearmm">Clear mismatches</button>
      </div>
      <p class="hint" style="margin-top:0">reference region the read came from:</p>
      <div id="ref" class="track"></div>
      <p class="hint">read (click any base to toggle a mismatch):</p>
      <div id="read" class="track"></div>
      <svg id="seedbars" viewBox="0 0 700 70"></svg>
      <div id="seedinfo" class="note" style="margin-top:4px"></div>
    </div>
  </div>

  <h2>The repeat problem, and the second pass</h2>
  <p>
    Here&rsquo;s the catch that makes seeding subtle. A long SMEM is great when
    it occurs in just one place. But genomes are repetitive: a long match can
    still occur in dozens of places, and worse, a long SMEM can <em>swallow</em>
    a more informative shorter seed that sits inside it. If you only ever take
    the single longest match, you can miss the seed that would have placed the
    read correctly.
  </p>
  <p>
    bwa-mem&rsquo;s fix, kept by minibwa, is a
    <span class="is-new">second seeding pass</span>. After collecting the
    standard SMEMs, minibwa re-seeds <em>inside</em> any SMEM that is both long
    and repetitive &mdash; in the code, when its span is at least twice the
    minimum and its occurrence count is modest. It looks for
    <span class="from-bwa">(mid,&nbsp;occ+1)-SMEMs</span>: shorter matches,
    starting from the middle, that occur <em>more</em> often than the parent
    &mdash; the buried, more-specific anchors. You can see both passes in
    <code>mb_seed_intv()</code>:
  </p>

  <div class="note" style="font-family:var(--mono); font-size:13px; white-space:pre-wrap; line-height:1.4">
do { <span style="color:var(--bwa)">// pass 1: standard SMEMs</span>
    x = mb_bwt_smem(bwt, len, seq, x, min_len, <b>1</b>, &amp;p);
    ...
} while (x &lt; len);
for (i = 0; i &lt; n_a0; ++i) { <span style="color:var(--new)">// pass 2: sub-SMEMs inside long, repetitive ones</span>
    if (en - st &lt; min_len * 2 || v-&gt;a[i].size &gt; max_sub_occ) continue;
    ...
    x = mb_bwt_smem(bwt, en, seq, x, sub_min_len, <b>v-&gt;a[i].size + 1</b>, &amp;p);
}
  </div>

  <p>
    bwa-mem could only afford one cheap second pass with its old engine.
    minibwa&rsquo;s reimplemented SMEM finder (next post) makes the second
    pass nearly free, so it can lean on re-seeding harder for better accuracy in
    repeats. The toy widget above shows occurrence counts so you can see which
    seeds <em>would</em> trigger a second pass &mdash; the long ones that still
    match in more than one place.
  </p>

  <div class="note key">
    <span class="label">Why SMEMs and not minimizers</span>
    minimap2 seeds with fixed-length minimizers &mdash; great for long, noisy
    reads. bwa-mem&rsquo;s variable-length SMEMs are more sensitive for short,
    accurate reads, because a single SMEM can span almost the whole read when
    there are no errors, giving an unambiguous anchor. minibwa keeps SMEMs
    precisely because its target is short, accurate reads.
  </div>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;Seeds are the read&rsquo;s exact-match runs between mismatches. I take
    the longest ones &mdash; SMEMs &mdash; and then make a second pass inside the
    long repetitive ones to dig out the shorter, more-specific seeds I&rsquo;d
    otherwise miss.&rdquo;
  </div>

<p>SMEMs give minibwa high-quality anchors for short accurate reads. The catch is that finding them means walking the FM-index again and again, which brings back the random-memory problem from Part 2. Part 4 is where minibwa earns most of its speedup.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
const REF  = "ACGTGACCTGACATTGCAGTCGGATTGACCAGTCATTGCAGAACGTTGACCTA";
//             read maps to ref[8..40); note "TTGCAG" repeat at ~14 and ~34
const ORIGIN = 8, RLEN = 30;
let readBases = REF.slice(ORIGIN, ORIGIN+RLEN).split("");
const refRegion = REF.slice(ORIGIN, ORIGIN+RLEN);
let mism = new Set();

function occ(sub){ // count occurrences in full REF
  if (!sub) return 0;
  let n=0, i=0;
  while ((i = REF.indexOf(sub, i)) !== -1){ n++; i++; }
  return n;
}
function renderTracks(){
  const minlen = +document.getElementById("minlen").value;
  document.getElementById("minlen-v").textContent = minlen;
  // ref track
  document.getElementById("ref").innerHTML = refRegion.split("")
    .map(b=>`<span class="cell ref base ${b}">${b}</span>`).join("");
  // read track
  document.getElementById("read").innerHTML = readBases
    .map((b,i)=>`<span class="cell base ${b} ${mism.has(i)?'mm':''}" data-i="${i}">${b}</span>`).join("");
  document.querySelectorAll("#read .cell").forEach(c=>{
    c.onclick = ()=>{ const i=+c.dataset.i; if(mism.has(i)){mism.delete(i);} else {mism.add(i);} apply(); };
  });
  computeSeeds(minlen);
}
function apply(){
  // toggling a mismatch substitutes the base so the run actually breaks
  readBases = refRegion.split("").map((b,i)=> mism.has(i) ? ({A:"T",T:"A",C:"G",G:"C"}[b]) : b);
  renderTracks();
}
function computeSeeds(minlen){
  // SMEMs = maximal exact runs of read vs origin (runs between mismatches)
  const runs = [];
  let s=0;
  const isMatch = i => readBases[i] === refRegion[i];
  for (let i=0;i<=RLEN;i++){
    if (i===RLEN || !isMatch(i)){
      if (i - s >= 1) runs.push([s,i]);
      s = i+1;
    }
  }
  const seeds = runs.filter(r => r[1]-r[0] >= minlen);
  // draw bars
  const svg = document.getElementById("seedbars");
  const X0=0, W=700, cellw = W/RLEN;
  let html = `<text x="0" y="12" font-family="var(--sans)" font-size="11" fill="var(--ink-soft)">seeds (length ≥ ${minlen}):</text>`;
  seeds.forEach((r,i)=>{
    const x = r[0]*cellw, w=(r[1]-r[0])*cellw;
    const sub = refRegion.slice(r[0], r[1]);
    const o = occ(sub);
    const reseeded = (r[1]-r[0] >= minlen*2) && o>1;
    const col = reseeded ? "var(--new)" : "var(--bwa)";
    html += `<rect x="${x}" y="22" width="${w-2}" height="22" rx="4" fill="${col}" opacity="0.85"/>
      <text x="${x+w/2}" y="37" text-anchor="middle" font-size="11" fill="#fff" font-weight="700">${r[1]-r[0]}bp ×${o}</text>`;
    if (reseeded)
      html += `<text x="${x+w/2}" y="58" text-anchor="middle" font-size="10" fill="var(--new)" font-family="var(--sans)">↳ re-seed</text>`;
  });
  if (!seeds.length) html += `<text x="10" y="36" font-size="13" fill="var(--accent)" font-family="var(--sans)">No seed survives the length filter — read would be hard to place.</text>`;
  svg.innerHTML = html;
  const reseedCount = seeds.filter(r => (r[1]-r[0]>=minlen*2) && occ(refRegion.slice(r[0],r[1]))>1).length;
  document.getElementById("seedinfo").innerHTML =
    `<span class="label">${seeds.length} seed${seeds.length!==1?'s':''}, ${mism.size} mismatch${mism.size!==1?'es':''}</span>`+
    `Each bar shows length and how many times that exact string occurs in the full reference. `+
    (reseedCount? `<b style="color:var(--new)">${reseedCount}</b> long seed${reseedCount!==1?'s':''} occur more than once — those are the ones pass&nbsp;2 would dig into for shorter, unique sub-seeds.`
                : `None are both long and repetitive right now, so pass&nbsp;2 finds nothing extra. Try removing mismatches to grow one long seed across the <code>TTGCAG</code> repeat.`);
}
document.getElementById("minlen").oninput = renderTracks;
document.getElementById("clearmm").onclick = ()=>{ mism.clear(); apply(); };
apply();
</script>
