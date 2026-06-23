---
layout: post
title: "How minibwa Works, Part 8: The Scorecard"
date: 2026-11-17 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

Now we can put the machine back together.

<!--more-->

minibwa keeps bwa-mem's FM-index and SMEM seeding, swaps in minimap2-style chaining and SIMD alignment, then reorganizes the hot loops so the CPU spends less time waiting on memory and less time running doomed dynamic programming. The question for this last post is practical: what does that buy, and when should you use it?


<div class="minibwa-series">
<h2>Speed</h2>
  <p>
    Same accuracy class as bwa-mem on whole-genome short reads, at a fraction of
    the wall-clock. Relative throughput (higher is faster), from the paper&rsquo;s
    short-read benchmark:
  </p>

  <div class="widget">
    <div class="wbar">Short-read throughput, relative to bwa-mem</div>
    <div class="wbody">
      <svg id="speed" viewBox="0 0 680 200"></svg>
      <p class="hint">Approximate, from the paper&rsquo;s headline figures. strobealign is ~10% faster than minibwa but, per the paper, &ldquo;does not compete on small-variant calling.&rdquo;</p>
    </div>
  </div>

  <h2>The full picture</h2>
  <table class="data">
    <tr><th>Dimension</th><th>minibwa vs. the field</th></tr>
    <tr><td>Speed vs bwa-mem</td><td>~4× faster, comparable accuracy</td></tr>
    <tr><td>Speed vs bwa-mem2</td><td>~2× faster (lead narrows on Hi-C/heavy mate-rescue data)</td></tr>
    <tr><td>Accuracy (WGS sim.)</td><td>bwa-mem slightly ahead; gap is <em>entirely</em> in centromeric/acrocentric regions — level once excluded</td></tr>
    <tr><td>Variant calling (HG002, DeepVariant)</td><td>Slightly fewer false negatives — 528 fewer SNP FNs, 1,104 fewer indel FNs than bwa-mem, from better chaining of >10 bp indels</td></tr>
    <tr><td>Peak memory</td><td>&lt; 20 GB — lower than most, higher than Bowtie2</td></tr>
    <tr><td>Accurate long reads</td><td>A touch faster than minimap2 at similar accuracy; both ~10× faster than Winnowmap2</td></tr>
    <tr><td>Bisulfite (BS-seq)</td><td>Native; beats BWA-Meth & BISCUIT on accuracy, several× faster, &gt;10× faster than Bismark</td></tr>
  </table>

  <h2>Which tool should you actually run?</h2>
  <p>
    My short version: use minibwa for short accurate reads and directional
    bisulfite data. Use minimap2 when the reads are noisy, spliced, or when you
    want one boring default for long-read workflows.
  </p>
  <table class="data">
    <tr><th>Data</th><th>Run</th><th>Why</th></tr>
    <tr><td>Short reads (Illumina, 100&ndash;250 bp)</td><td><code>minibwa</code></td><td>Its home turf: fast, accurate, and variant-calling quality on par with or slightly better than bwa-mem.</td></tr>
    <tr><td>Accurate long reads (HiFi)</td><td><code>minibwa</code> or <code>minimap2</code></td><td>minibwa is a little faster at similar accuracy; minimap2 is still the safer default if the same pipeline also handles other long-read modes.</td></tr>
    <tr><td>Noisy long reads (ONT, old PacBio)</td><td><code>minimap2</code></td><td>minibwa is not built for high error rates. minimap2&rsquo;s minimizer seeding is much more robust here.</td></tr>
    <tr><td>Spliced RNA-seq</td><td><code>minimap2 -x splice</code> or a splice-aware aligner</td><td>minibwa has no spliced-alignment mode.</td></tr>
    <tr><td>Directional bisulfite (methylation)</td><td><code>minibwa --meth</code></td><td>Native support; more accurate than BWA-Meth/BISCUIT and much faster than Bismark. Undirectional BS-seq is not supported.</td></tr>
    <tr><td>Hi-C</td><td><code>minibwa --hic</code></td><td>Supported and fast, though the speed lead over bwa-mem2 narrows because Hi-C leans heavily on mate rescue.</td></tr>
  </table>

  <h2>The one idea, three times</h2>
  <p>
    If you remember nothing else: minibwa&rsquo;s speed is not a faster
    algorithm, it&rsquo;s the refusal to waste cycles. The same instinct shows
    up at every scale of the pipeline &mdash; do a cheap thing first to avoid an
    expensive thing later:
  </p>
  <table class="data">
    <tr><th>Cheap test</th><th>Expensive thing it avoids</th><th>Part</th></tr>
    <tr><td>Prefetch + batch the next memory access</td><td>A 200-cycle CPU stall on a cache miss</td><td class="num">04</td></tr>
    <tr><td>10-mer interval cache</td><td>Ten random-memory backward-search steps</td><td class="num">04</td></tr>
    <tr><td>Ungapped alignment attempt</td><td>Full SIMD Smith–Waterman in the gap</td><td class="num">06</td></tr>
    <tr><td>7-mer Hough vote</td><td>A doomed mate-rescue Smith–Waterman</td><td class="num">06</td></tr>
    <tr><td>Cap effort in repeats</td><td>Grinding DP on unmappable bases</td><td class="num">07</td></tr>
  </table>

  <h2>The eight sentences</h2>
  <p>If you can say all of these, you understand minibwa:</p>
  <ol>
    <li>Mapping is impossible by brute force, so we seed, chain, then align only in the gaps.</li>
    <li>The BWT plus a count table finds exact matches by shrinking an interval backward — but each step is a random memory jump.</li>
    <li>Seeds are SMEMs — the read&rsquo;s longest exact-match runs — plus a second pass inside long repetitive ones.</li>
    <li>minibwa is fast because it batches reads and prefetches, hiding memory latency instead of doing less math.</li>
    <li>Chaining finds the colinear string of seeds on a dotplot, tweaked so overlapping seeds don&rsquo;t double-count.</li>
    <li>Real DP runs only in gaps, only when an ungapped check fails; a 7-mer vote gates mate rescue.</li>
    <li>One formula retunes parameters per read length; effort is capped in unmappable repeats.</li>
    <li>Bisulfite reads map natively via a four-copy converted index and an asymmetric C→T-forgiving score.</li>
  </ol>

  <div class="note feynman">
    <span class="label">The whole thing, in one breath</span>
    minibwa is bwa-mem&rsquo;s seeding and minimap2&rsquo;s chain-and-align,
    stitched together and then engineered so the CPU never waits on memory and
    never runs an expensive step it can cheaply predict will fail.
  </div>

  <p style="margin-top:40px">
    Source and paper:
    <a href="https://github.com/lh3/minibwa">github.com/lh3/minibwa</a> &middot;
    <a href="https://arxiv.org/abs/2606.15357">arXiv:2606.15357</a>.
    The design notes in <code>dev.md</code> map each component to its origin.
  </p>

<p>The version I carry around now is simple: minibwa is not a new theory of read alignment. It is a careful recombination of proven pieces, with the hottest parts reshaped around modern hardware. That is less glamorous than inventing a brand-new aligner, but it is exactly the kind of engineering that makes a tool matter.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
const NS="http://www.w3.org/2000/svg";

/* ---- speed bars ---- */
(function(){
  const svg=document.getElementById("speed");
  const rows=[ // [label, relative throughput vs bwa-mem=1, color]
    ["minibwa", 4.0, "var(--new)"],
    ["bwa-mem2", 2.0, "var(--bwa)"],
    ["bwa-mem", 1.0, "var(--rule-2)"],
  ];
  const X0=110, W=480, maxv=4.4;
  let g="";
  rows.forEach((r,i)=>{
    const y=20+i*56, w=r[1]/maxv*W;
    g+=`<text x="${X0-10}" y="${y+22}" text-anchor="end" font-family="var(--sans)" font-size="14" font-weight="700">${r[0]}</text>
        <rect x="${X0}" y="${y}" width="${w}" height="32" rx="5" fill="${r[2]}" opacity="0.9"/>
        <text x="${X0+w+8}" y="${y+22}" font-family="var(--mono)" font-size="14" fill="var(--ink-soft)">${r[1].toFixed(1)}×</text>`;
  });
  svg.innerHTML=g;
})();

</script>
