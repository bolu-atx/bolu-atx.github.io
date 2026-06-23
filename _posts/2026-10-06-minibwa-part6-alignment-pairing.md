---
layout: post
title: "How minibwa Works, Part 6: Alignment, Pairing, and Mate Rescue"
date: 2026-10-06 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

Chaining tells us where the read probably belongs. Alignment tells us exactly how it differs from the reference.

<!--more-->

This is where Smith-Waterman dynamic programming comes back, but in a much smaller box. minibwa uses minimap2's SIMD alignment code for the gaps between seeds, then adds cheap filters so it can avoid running full dynamic programming when the answer is almost certainly boring or hopeless.


<div class="minibwa-series">
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
    So: tally, for each offset, how many q-mer matches land on it. If the best
    offset gets at least 10 votes, run the full Smith&ndash;Waterman. Otherwise,
    skip it &mdash; there&rsquo;s no line to find. Add mismatches to the read
    below and watch the peak collapse past the threshold:
  </p>

  <div class="widget">
    <div class="wbar">Mate-rescue pre-filter: do the votes clear the bar?</div>
    <div class="wbody">
      <div class="controls">
        <label>mismatches in read
          <input type="range" id="mm" min="0" max="12" value="2">
          <b id="mm-v" style="font-family:var(--mono)">2</b></label>
        <label>q-mer length
          <input type="range" id="q" min="3" max="6" value="4">
          <b id="q-v" style="font-family:var(--mono)">4</b></label>
        <label>vote threshold
          <input type="range" id="thr" min="2" max="12" value="6">
          <b id="thr-v" style="font-family:var(--mono)">6</b></label>
      </div>
      <p class="hint" style="margin:2px 0">reference window near the mapped mate:</p>
      <div id="ref" class="seqline"></div>
      <p class="hint">candidate (missing) mate:</p>
      <div id="read" class="seqline"></div>
      <svg id="votes" viewBox="0 0 700 150"></svg>
      <div id="verdict" class="note" style="margin-top:4px"></div>
    </div>
  </div>

  <p>
    The real threshold is 10 votes with 7-mers; the widget uses smaller numbers
    so a toy sequence can illustrate it. The payoff is real: the paper notes
    this filter lets minibwa <em>skip unsuccessful alignments</em>, and disabling
    it is exactly what narrows minibwa&rsquo;s lead over bwa-mem2 on Hi-C and
    other data where mate rescue is heavy. A cheap vote replaces an expensive
    DP that was going to fail anyway.
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

<p>The pattern is now pretty hard to miss: do a cheap test first, and only run the expensive thing when the cheap test says it is worth it. Part 7 is about the less glamorous but very real production details: retuning parameters, giving up in repeats, and handling bisulfite sequencing without a wrapper.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
const REFW = "TGACCAGTCATTGCAGAACGTTGACCTAGGTCAACTGAC";
let readSeq = REFW.slice(6, 6+20); // true placement at offset 6
let mmPos = [];

function applyMM(){
  const k=+document.getElementById("mm").value;
  // deterministic mismatch positions spread across read
  mmPos=[]; for (let i=0;i<k;i++) mmPos.push(Math.floor((i+0.5)*20/k));
  const orig=REFW.slice(6,6+20).split("");
  readSeq=orig.map((b,i)=> mmPos.includes(i)? ({A:"T",T:"A",C:"G",G:"C"}[b]) : b).join("");
  render();
}
function render(){
  const q=+document.getElementById("q").value;
  const thr=+document.getElementById("thr").value;
  document.getElementById("mm-v").textContent=document.getElementById("mm").value;
  document.getElementById("q-v").textContent=q;
  document.getElementById("thr-v").textContent=thr;
  document.getElementById("ref").innerHTML=REFW.split("").map(b=>`<span class="cell base ${b}">${b}</span>`).join("");
  document.getElementById("read").innerHTML=readSeq.split("").map((b,i)=>`<span class="cell base ${b} ${mmPos.includes(i)?'mm':''}">${b}</span>`).join("");
  // votes: for each offset t (ref start), count q-mer matches at that diagonal
  const maxOff=REFW.length-readSeq.length;
  const votes=[];
  for (let t=0;t<=maxOff;t++){
    let v=0;
    for (let i=0;i+q<=readSeq.length;i++){
      if (readSeq.substr(i,q)===REFW.substr(t+i,q)) v++;
    }
    votes.push(v);
  }
  const peak=Math.max(...votes), peakAt=votes.indexOf(peak);
  // draw histogram
  const svg=document.getElementById("votes");
  const X0=10, W=680, bw=W/votes.length, maxV=Math.max(peak,thr,1);
  let g=`<text x="0" y="12" font-family="var(--sans)" font-size="11" fill="var(--ink-soft)">votes per diagonal offset</text>`;
  votes.forEach((v,t)=>{
    const h=v/maxV*100, x=X0+t*bw, y=120-h;
    const col = t===peakAt? "var(--mm2)":"var(--bwa)";
    g+=`<rect x="${x}" y="${y}" width="${Math.max(2,bw-2)}" height="${h}" fill="${col}" opacity="${t===peakAt?0.9:0.5}"/>`;
  });
  const ty=120-thr/maxV*100;
  g+=`<line x1="${X0}" y1="${ty}" x2="${X0+W}" y2="${ty}" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="5 3"/>`;
  g+=`<text x="${X0+W}" y="${ty-4}" text-anchor="end" font-family="var(--sans)" font-size="11" fill="var(--accent)">threshold = ${thr}</text>`;
  g+=`<text x="${X0+peakAt*bw}" y="${120-peak/maxV*100-4}" font-family="var(--mono)" font-size="11" fill="var(--mm2)" font-weight="700">${peak}</text>`;
  svg.innerHTML=g;
  const pass=peak>=thr;
  document.getElementById("verdict").innerHTML = pass
    ? `<span class="label" style="color:var(--mm2)">✓ run Smith–Waterman</span>The best diagonal collected <b>${peak}</b> votes (≥ ${thr}). A real alignment almost certainly exists here, so it's worth the full DP. The voting peak even tells us roughly where to start.`
    : `<span class="label" style="color:var(--accent)">✗ skip — no DP</span>The best diagonal got only <b>${peak}</b> votes (< ${thr}). The q-mer matches are scattered, not lined up — there's no alignment to find. minibwa skips the Smith–Waterman entirely and saves the work.`;
}
["mm","q","thr"].forEach(id=>document.getElementById(id).addEventListener("input", id==="mm"?applyMM:render));
applyMM();
</script>
