---
layout: post
title: "How minibwa Works, Part 5: Chaining Seeds into Alignments"
date: 2026-09-15 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

After seeding, we do not have an alignment. We have a pile of exact-match anchors: some from the true location, some from repeats, some just unlucky noise.

<!--more-->

The useful picture is a dotplot. Put read position on one axis and reference position on the other. Real anchors form a diagonal. Spurious anchors scatter. Chaining is the dynamic program that finds the best diagonal-ish run of seeds and turns it into a candidate alignment.


<div class="minibwa-series">
<h2>The dotplot picture</h2>
  <p>
    Put read position on one axis and reference position on the other. Every
    seed match is a point (really a short diagonal segment, since a seed has
    length). A read that came from one contiguous locus shows up as points
    strung along a single diagonal &mdash; as you move right along the read, you
    move right along the reference in lockstep. Spurious seeds scatter off that
    diagonal.
  </p>
  <p>
    Chaining&rsquo;s job: find the highest-scoring <b>colinear</b> set &mdash;
    anchors increasing in both coordinates, close to one diagonal. Gaps between
    consecutive anchors are allowed (that&rsquo;s an indel or an error-induced
    break), but they cost score, and the cost grows with how far apart and how
    off-diagonal they are. Hit <b>Run chaining</b>:
  </p>

  <div class="widget">
    <div class="wbar">Colinear chaining on a dotplot</div>
    <div class="wbody">
      <div class="controls">
        <label>spurious seeds
          <input type="range" id="noise" min="0" max="40" value="14">
          <b id="noise-v" style="font-family:var(--mono)">14</b></label>
        <label>indel in read
          <input type="checkbox" id="indel"></label>
        <button class="btn" id="run">Run chaining</button>
        <button class="btn ghost" id="reroll">New seeds</button>
      </div>
      <svg id="plot" viewBox="0 0 420 420" style="margin:0 auto"></svg>
      <div id="chain-info" class="note" style="margin-top:4px"></div>
    </div>
  </div>

  <h2>How the chaining decides &mdash; one DP recurrence</h2>
  <p>
    It&rsquo;s dynamic programming over anchors sorted by position. For anchor
    <code>i</code>, the best chain ending at <code>i</code> is its own weight
    plus the best predecessor <code>j</code> that sits below-left of it, minus
    the cost of the jump from <code>j</code> to <code>i</code>:
  </p>
  <div class="note" style="font-family:var(--mono); font-size:14px; text-align:center; line-height:1.6">
    f(i) = max<sub>j</sub> [ f(j) + overlap-adjusted weight(i) &minus; gap_cost(j&rarr;i) ]
  </div>
  <p>
    The <b>gap cost</b> is the crux. A jump where read-distance and
    reference-distance match (same diagonal) is nearly free &mdash; that&rsquo;s
    just the read continuing. A jump that&rsquo;s longer on one axis than the
    other implies an insertion or deletion, and it&rsquo;s penalized by how big
    the discrepancy is. Anchors too far apart aren&rsquo;t even considered. Toggle
    the <b>indel</b> checkbox above: the chain survives a moderate diagonal jump
    because the gap cost stays below the reward of extending the chain.
  </p>

  <h2>The wrinkle minibwa had to fix: variable-length seeds</h2>
  <p>
    minimap2 chains <em>minimizers</em> &mdash; fixed-length, regularly spaced
    anchors. minibwa feeds it <span class="from-bwa">SMEMs</span>, which vary
    wildly in length and can <em>overlap</em> each other (recall the two-pass
    re-seeding from Part 3 deliberately produces overlapping seeds). If you
    naively scored each seed by its full length, overlapping seeds would
    double-count the bases they share and inflate the chain score.
  </p>
  <p>
    So minibwa <span class="is-new">adapts the chaining</span> to count only the
    <em>new</em> bases each anchor contributes beyond the previous one &mdash;
    the overlap-adjusted weight in the recurrence above. This is the &ldquo;adapted
    for variable-length seeds&rdquo; line from the design notes, and it&rsquo;s
    the one place minimap2&rsquo;s chaining couldn&rsquo;t be used unmodified.
  </p>

  <div class="note key">
    <span class="label">Why chaining, not bwa-mem&rsquo;s extension</span>
    bwa-mem extended each seed independently and glued results together with
    heuristics &mdash; fine for 100&nbsp;bp reads, shaky for anything longer or
    with structural differences. Colinear chaining reasons about <em>all</em>
    seeds jointly, so it handles larger indels and longer reads gracefully. This
    is why minibwa calls >10&nbsp;bp indels slightly better than bwa-mem in the
    variant-calling benchmark, and why it works on long reads at all.
  </div>

  <p>
    A read can produce more than one good chain &mdash; think segmental
    duplications or a read spanning a structural variant. minibwa keeps up to
    the <span class="is-new">50 best chains</span> for a short read and sends
    each to the alignment stage, which is the subject of the next post.
  </p>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;On a read-vs-reference dotplot, the true alignment is a string of
    seeds on one diagonal. Chaining is the dynamic program that finds that
    string, paying a penalty for off-diagonal jumps. minibwa uses
    minimap2&rsquo;s chaining, tweaked so overlapping variable-length seeds
    don&rsquo;t double-count.&rdquo;
  </div>

<p>At this point the read has a plausible location and a chain of anchors explaining most of it. What we still do not have is the base-level answer: the exact mismatches, insertions, deletions, and CIGAR string. That is the alignment stage.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
const NS="http://www.w3.org/2000/svg";
const svg=document.getElementById("plot");
const SZ=420, PAD=34, N=100; // coordinate space 0..N
let anchors=[], chosen=[];

function rng(seed){ let s=seed; return ()=> (s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff; }
let seed0=7;

function gen(){
  const r=rng(seed0);
  anchors=[];
  const indel=document.getElementById("indel").checked;
  // true diagonal anchors: y = x + offset, with optional indel jump partway
  const off=18, nTrue=8;
  for (let i=0;i<nTrue;i++){
    let x = 8 + i*10 + r()*3;
    let y = x + off + (indel && i>=4 ? 12 : 0); // a deletion: ref jumps ahead
    const w = 4 + r()*5;
    anchors.push({x,y,w,true:true});
  }
  const noise=+document.getElementById("noise").value;
  for (let i=0;i<noise;i++){
    anchors.push({x: r()*N, y: r()*N, w: 3+r()*4, true:false});
  }
  chosen=[];
  draw();
}

function chain(){
  // sort by x then y
  const a=[...anchors].map((p,i)=>({...p,i})).sort((p,q)=> p.x-q.x || p.y-q.y);
  const n=a.length;
  const f=new Array(n), pre=new Array(n).fill(-1);
  const MAXGAP=35;
  for (let i=0;i<n;i++){
    f[i]=a[i].w;
    for (let j=0;j<i;j++){
      const dx=a[i].x-a[j].x, dy=a[i].y-a[j].y;
      if (dx<=0 || dy<=0) continue;
      if (dx>MAXGAP || dy>MAXGAP) continue;
      const gap=Math.abs(dx-dy);
      const dist=Math.max(dx,dy);
      // overlap-adjusted weight: only new bases beyond overlap with j
      const newbases=Math.min(a[i].w, dx, dy);
      const cost = 0.6*gap + 0.15*dist;     // off-diagonal + distance penalty
      const sc = f[j] + newbases - cost;
      if (sc>f[i]){ f[i]=sc; pre[i]=j; }
    }
  }
  // best end
  let best=0; for (let i=1;i<n;i++) if (f[i]>f[best]) best=i;
  const path=[]; let k=best;
  while (k>=0){ path.push(a[k].i); k=pre[k]; }
  chosen=path; draw(f[best]);
}

function draw(score){
  const sc = v => PAD + v/N*(SZ-2*PAD);
  let g=`<rect x="${PAD}" y="${PAD}" width="${SZ-2*PAD}" height="${SZ-2*PAD}" fill="none" stroke="var(--rule-2)"/>`;
  g+=`<text x="${SZ/2}" y="${SZ-6}" text-anchor="middle" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">read position →</text>`;
  g+=`<text x="14" y="${SZ/2}" text-anchor="middle" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)" transform="rotate(-90 14 ${SZ/2})">reference position →</text>`;
  // chain line
  if (chosen.length>1){
    const pts=chosen.map(i=>`${sc(anchors[i].x)},${SZ-sc(anchors[i].y)}`).join(" ");
    g+=`<polyline points="${pts}" fill="none" stroke="var(--mm2)" stroke-width="2.5" opacity="0.6"/>`;
  }
  // anchors as short diagonal segments
  const chosenSet=new Set(chosen);
  anchors.forEach((p,i)=>{
    const inchain=chosenSet.has(i);
    const col = inchain ? "var(--mm2)" : (p.true? "var(--bwa)":"var(--rule-2)");
    const x1=sc(p.x), y1=SZ-sc(p.y), x2=sc(p.x+p.w), y2=SZ-sc(p.y+p.w);
    g+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${inchain?4:3}" stroke-linecap="round" opacity="${inchain?1:(p.true?0.7:0.5)}"/>`;
  });
  svg.innerHTML=g;
  const info=document.getElementById("chain-info");
  if (score===undefined){
    info.innerHTML=`<span class="label">${anchors.length} seeds</span>Blue = seeds on the true diagonal, gray = spurious. Press <b>Run chaining</b> to find the best colinear chain (drawn in green).`;
  } else {
    const trueIn = chosen.filter(i=>anchors[i].true).length;
    info.innerHTML=`<span class="label">chain score ${score.toFixed(0)}</span>The DP picked ${chosen.length} anchors (${trueIn} of them true), ignoring the off-diagonal scatter. `+
      (document.getElementById("indel").checked? `Notice it spans the diagonal jump in the middle — that's the deletion, paid for once as a gap cost rather than breaking the chain.` : `Every chosen anchor sits on one diagonal: a clean, contiguous placement.`);
  }
}
document.getElementById("noise").oninput=e=>{document.getElementById("noise-v").textContent=e.target.value; gen();};
document.getElementById("indel").onchange=gen;
document.getElementById("reroll").onclick=()=>{seed0=(seed0*7+3)&0x7fff; gen();};
document.getElementById("run").onclick=chain;
gen();
</script>
