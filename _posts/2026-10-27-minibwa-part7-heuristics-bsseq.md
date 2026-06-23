---
layout: post
title: "How minibwa Works, Part 7: Heuristics, Repeats, and BS-seq"
date: 2026-10-27 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

Toy examples are kind. Real sequencing data is not.

<!--more-->

Reads have different lengths. Some land in repeats where the reference is a bad representative of the sample. Bisulfite sequencing deliberately turns some Cs into Ts, which breaks the assumptions of a normal aligner. This part is about the pragmatic layer around the core seed-chain-align machine.


<div class="minibwa-series">
<h2>One formula that slides from short reads to long</h2>
  <p>
    Short reads and long reads want different settings &mdash; band widths, gap
    limits, how many seeds to keep. Most aligners make you pick a preset
    (<code>-x sr</code> vs <code>-x lr</code>). minibwa&rsquo;s default instead
    sets each such parameter <span class="is-new">per read</span>, interpolating
    smoothly between a short-read value and a long-read value with one formula:
  </p>
  <div class="note" style="font-family:var(--mono); font-size:14px; text-align:center; line-height:1.7">
    &theta;(&#8467;) = &theta;<sub>L</sub> &minus; (&theta;<sub>L</sub> &minus; &theta;<sub>S</sub>) &middot; 2<sup>&minus;max(&#8467;&minus;&#8467;<sub>min</sub>, 0) / (&#8467;<sub>mid</sub> &minus; &#8467;<sub>min</sub>)</sup>
  </div>
  <p>
    With <code>&#8467;<sub>min</sub> = 100</code> and
    <code>&#8467;<sub>mid</sub> = 2000</code> bp hardcoded: at or below 100 bp
    you get the pure short-read value &theta;<sub>S</sub>; as reads lengthen the
    exponential decay pulls the parameter toward the long-read value
    &theta;<sub>L</sub>, reaching halfway at ~2000 bp. A mixed-length library
    just works, with no flag. Drag the read length:
  </p>

  <div class="widget">
    <div class="wbar">The adaptive parameter curve &theta;(&#8467;)</div>
    <div class="wbody">
      <div class="controls">
        <label>read length &#8467;
          <input type="range" id="len" min="50" max="3000" value="150" step="10">
          <b id="len-v" style="font-family:var(--mono)">150</b> bp</label>
      </div>
      <svg id="curve" viewBox="0 0 680 240"></svg>
      <div id="curve-info" class="note" style="margin-top:4px"></div>
      <p class="hint">
        Short-read regime (&#8467; ≤ 100) &mdash;&mdash; transition (~2 kb half-way) &mdash;&mdash; long-read regime.
        Setting <code>-x sr</code>, <code>-x lr</code>, or <code>--adap=no</code> turns this off and pins one regime.
      </p>
    </div>
  </div>

  <h2>Knowing when to give up</h2>
  <p>
    The most counterintuitive heuristic: minibwa
    <span class="is-new">deliberately does less work in highly repetitive
    regions</span>. Centromeric and other repeat-dense sequence evolves fast and
    differs structurally between any individual and the reference, so a short
    read landing there usually <em>cannot</em> be placed correctly no matter how
    much DP you throw at it. bwa-mem grinds away anyway. minibwa caps the effort
    &mdash; fewer extension attempts, fewer rescued candidates &mdash; and on
    simulated data accuracy is essentially unchanged, because the bases it gives
    up on were unmappable to begin with.
  </p>
  <div class="note">
    <span class="label">An honest tradeoff, not a free lunch</span>
    This is the one place the benchmarks show minibwa slightly behind bwa-mem on
    raw accuracy &mdash; and the gap is concentrated entirely in centromeric and
    acrocentric regions. Exclude those, and the two are level. The bet is that
    nobody trusts variant calls in the centromere anyway, so spending less time
    there is the right call. You can see it directly in the paper&rsquo;s
    accuracy panels: the difference vanishes once those regions are removed.
  </div>

  <h2>Bisulfite sequencing, built in</h2>
  <p>
    Bisulfite treatment converts unmethylated <b>C</b> to <b>T</b>, so methylated
    cytosines can be read out &mdash; but it wrecks normal alignment, because now
    a genomic C legitimately matches a read T. The standard workaround
    (BWA-Meth, Bismark) is a wrapper that converts everything to a 3-letter
    alphabet and shells out to a regular aligner. minibwa does it
    <span class="is-new">natively</span>. Toggle the steps:
  </p>

  <div class="widget">
    <div class="wbar">How minibwa indexes and maps bisulfite reads</div>
    <div class="wbody">
      <div class="controls">
        <button class="btn" id="bs-step">Next step &rarr;</button>
        <button class="btn ghost" id="bs-reset">Reset</button>
      </div>
      <svg id="bsviz" viewBox="0 0 680 180"></svg>
      <div id="bs-info" class="note" style="margin-top:4px"></div>
    </div>
  </div>

  <p>
    The subtle bits that make it accurate, all handled inside minibwa: the
    converted index actually contains <em>four</em> copies of the genome
    (forward C&rarr;T, forward G&rarr;A, and both reverses) so either strand and
    either read of a pair can find its seeds; seed matches are re-checked against
    the <em>original</em> sequences and split at any T&rarr;C mismatch (discarding
    pieces under 19 bp) to avoid false matches created by the conversion; and base
    alignment uses an <span class="is-new">asymmetric scoring matrix</span> that
    forgives C&rarr;T (the expected conversion) but penalizes T&rarr;C (a real
    difference). The result beats both BWA-Meth and BISCUIT on accuracy while
    running several times faster &mdash; and over 10&times; faster than Bismark.
  </p>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;One formula retunes every length-sensitive parameter per read, so I
    never pick a preset. minibwa spends less effort in repeats it can&rsquo;t map
    anyway, trading a sliver of centromeric accuracy for speed. And it aligns
    bisulfite reads natively with a four-copy converted index and an asymmetric
    score that forgives C-to-T but not T-to-C.&rdquo;
  </div>

<p>None of these details changes the basic architecture, but they decide whether the tool is pleasant and trustworthy on real datasets. The final post pulls the pieces together: what minibwa buys, where it fits, and when I would actually reach for it.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
const NS="http://www.w3.org/2000/svg";

/* ---- adaptive curve ---- */
(function(){
  const svg=document.getElementById("curve");
  const len=document.getElementById("len"), lenv=document.getElementById("len-v");
  const info=document.getElementById("curve-info");
  const LMIN=100, LMID=2000, thetaS=20, thetaL=200; // illustrative: e.g. DP band width
  const X0=46, Y0=20, W=600, H=170, LMAX=3000;
  function theta(l){ return thetaL - (thetaL-thetaS)*Math.pow(2, -Math.max(l-LMIN,0)/(LMID-LMIN)); }
  function px(l){ return X0 + l/LMAX*W; }
  function py(t){ return Y0 + (1 - (t-thetaS)/(thetaL-thetaS))*H; }
  function draw(){
    const l=+len.value; lenv.textContent=l;
    let g=`<line x1="${X0}" y1="${Y0}" x2="${X0}" y2="${Y0+H}" stroke="var(--rule-2)"/>
           <line x1="${X0}" y1="${Y0+H}" x2="${X0+W}" y2="${Y0+H}" stroke="var(--rule-2)"/>
           <text x="${X0-6}" y="${py(thetaS)+4}" text-anchor="end" font-size="11" fill="var(--ink-soft)" font-family="var(--sans)">short</text>
           <text x="${X0-6}" y="${py(thetaL)+4}" text-anchor="end" font-size="11" fill="var(--ink-soft)" font-family="var(--sans)">long</text>
           <text x="${X0+W/2}" y="${Y0+H+28}" text-anchor="middle" font-size="12" fill="var(--ink-soft)" font-family="var(--sans)">read length (bp) →</text>`;
    // regime shading
    g+=`<rect x="${X0}" y="${Y0}" width="${px(LMIN)-X0}" height="${H}" fill="var(--bwa)" opacity="0.06"/>`;
    // curve
    let pts="";
    for (let l=0;l<=LMAX;l+=20) pts+=`${px(l)},${py(theta(l))} `;
    g+=`<polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>`;
    // halfway marker at LMID
    g+=`<line x1="${px(LMID)}" y1="${Y0}" x2="${px(LMID)}" y2="${Y0+H}" stroke="var(--rule-2)" stroke-dasharray="3 3"/>
        <text x="${px(LMID)}" y="${Y0+H+14}" text-anchor="middle" font-size="10" fill="var(--ink-soft)" font-family="var(--sans)">2 kb</text>`;
    // current point
    const t=theta(l);
    g+=`<circle cx="${px(l)}" cy="${py(t)}" r="6" fill="var(--new)" stroke="#fff" stroke-width="2"/>
        <line x1="${px(l)}" y1="${py(t)}" x2="${px(l)}" y2="${Y0+H}" stroke="var(--new)" stroke-width="1" stroke-dasharray="2 2"/>`;
    svg.innerHTML=g;
    const frac=(t-thetaS)/(thetaL-thetaS);
    info.innerHTML=`<span class="label">&#8467; = ${l} bp</span>`+
      (l<=LMIN? `At ${l} bp you're in the pure short-read regime — the parameter sits at its short value, exactly what a dedicated short-read aligner would use.`
       : l>=LMID*1.4? `At ${l} bp the parameter has decayed ${Math.round(frac*100)}% of the way to the long-read value — minibwa is treating this essentially as a long read.`
       : `At ${l} bp the parameter is ${Math.round(frac*100)}% of the way from short to long — a smooth blend, chosen per read with no preset.`);
  }
  len.oninput=draw; draw();
})();

/* ---- BS-seq stepper ---- */
(function(){
  const svg=document.getElementById("bsviz");
  const info=document.getElementById("bs-info");
  const steps=[
    {t:"Bisulfite treatment converts unmethylated C→T in the read. A genomic C can now legitimately read as T — normal alignment breaks.",
     draw:()=>`
       <text x="0" y="20" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">genomic:</text>
       <text x="90" y="20" font-family="var(--mono)" font-size="16">A C G <tspan fill="var(--accent)">C</tspan> T A <tspan fill="var(--accent)">C</tspan> G</text>
       <text x="0" y="60" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">read (BS):</text>
       <text x="90" y="60" font-family="var(--mono)" font-size="16">A C G <tspan fill="var(--accent)">T</tspan> T A <tspan fill="var(--accent)">T</tspan> G</text>
       <text x="90" y="92" font-family="var(--sans)" font-size="12" fill="var(--accent)">unmethylated C's became T's ↑</text>`},
    {t:"minibwa converts the index too: it builds an FM-index over FOUR copies of the reference — forward C→T, forward G→A, and both reverse complements — so any strand/mate can seed.",
     draw:()=>{
       const labels=["forward C→T","forward G→A","reverse G→A","reverse C→T"];
       return labels.map((l,i)=>`<rect x="${10+i*168}" y="60" width="150" height="40" rx="8" fill="var(--new)" opacity="0.14" stroke="var(--new)"/>
         <text x="${85+i*168}" y="84" text-anchor="middle" font-family="var(--sans)" font-size="12" font-weight="700" fill="var(--new)">${l}</text>`).join("")
         + `<text x="340" y="30" text-anchor="middle" font-family="var(--sans)" font-size="13" font-weight="700">one FM-index, four converted genomes</text>`;
     }},
    {t:"After seeding, minibwa re-fetches the ORIGINAL read and reference, splits each seed at any T→C mismatch, and drops fragments under 19 bp — so conversion artifacts can't create false seeds.",
     draw:()=>`
       <text x="0" y="40" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">seed (converted): </text>
       <rect x="160" y="28" width="360" height="18" rx="4" fill="var(--bwa)" opacity="0.6"/>
       <text x="0" y="80" font-family="var(--sans)" font-size="12" fill="var(--ink-soft)">original check: </text>
       <rect x="160" y="68" width="150" height="18" rx="4" fill="var(--mm2)" opacity="0.7"/>
       <line x1="320" y1="62" x2="320" y2="92" stroke="var(--accent)" stroke-width="2"/>
       <text x="324" y="80" font-family="var(--sans)" font-size="11" fill="var(--accent)">T→C mismatch: split here</text>
       <rect x="330" y="68" width="190" height="18" rx="4" fill="var(--mm2)" opacity="0.7"/>`},
    {t:"Base alignment uses an ASYMMETRIC scoring matrix: a read T against a reference C costs nothing (expected conversion), but a read C against a reference T is penalized (a real mismatch). Methylation calls fall out of the original bases.",
     draw:()=>{
       const M=[["", "ref A","ref C","ref G","ref T"],
                ["read C","–","+","–","ok"],
                ["read T","–","ok→","–","+"]];
       let s="";
       M.forEach((row,r)=> row.forEach((c,cc)=>{
         const x=120+cc*100, y=30+r*42;
         const hi = (r===2&&cc===2)||(r===1&&cc===2);
         s+=`<text x="${x}" y="${y}" text-anchor="middle" font-family="var(--mono)" font-size="13" fill="${hi?'var(--new)':'var(--ink)'}" font-weight="${hi?700:400}">${c}</text>`;
       }));
       s+=`<text x="340" y="170" text-anchor="middle" font-family="var(--sans)" font-size="11" fill="var(--new)">read T vs ref C = forgiven; read C vs ref T = penalized</text>`;
       return s;
     }}
  ];
  let i=0;
  function draw(){
    svg.innerHTML=steps[i].draw();
    info.innerHTML=`<span class="label">Step ${i+1} of ${steps.length}</span>${steps[i].t}`;
  }
  document.getElementById("bs-step").onclick=()=>{i=(i+1)%steps.length;draw();};
  document.getElementById("bs-reset").onclick=()=>{i=0;draw();};
  draw();
})();
</script>
