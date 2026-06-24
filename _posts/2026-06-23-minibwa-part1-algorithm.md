---
layout: post
title: "How minibwa Works, Part 1: The Algorithm"
date: 2026-06-23 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

A 150-base read shows up from the sequencer. Somewhere in the 3.1-billion-base human reference, there is probably a place it came from. The whole algorithm is a way to avoid comparing that read against all 3.1 billion bases directly.

<!--more-->

The trick is seed, chain, align: find cheap exact matches, connect the ones that make geometric sense, and only then pay for dynamic programming in the tiny gaps that remain. minibwa keeps bwa-mem's FM-index and SMEM seeding, then borrows minimap2-style chaining and alignment for the back half.

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
          <input type="range" id="alg-cost-nreads" min="1" max="800" value="400">
          <b id="alg-cost-nreads-v" style="font-family:var(--mono)">400</b></label>
        <label>Read length
          <input type="range" id="alg-cost-rlen" min="50" max="300" value="150" step="10">
          <b id="alg-cost-rlen-v" style="font-family:var(--mono)">150</b></label>
      </div>
      <svg id="alg-cost-viz" viewBox="0 0 680 150"></svg>
      <p class="hint" id="alg-cost-text"></p>
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
      <svg id="alg-sca-viz" viewBox="0 0 680 230"></svg>
      <div class="controls" style="margin-top:12px">
        <button class="btn" id="alg-sca-step">Next step &rarr;</button>
        <button class="btn ghost" id="alg-sca-reset">Reset</button>
      </div>
      <div id="alg-sca-cap" class="note" style="margin-top:4px"></div>
    </div>
  </div>

  <p>
    <span class="from-bwa">Seeding</span> finds short stretches that match the
    reference <em>exactly</em>. Exact matching is cheap because we can look it up
    in an index instead of computing DP &mdash; that&rsquo;s what the FM-index
    section below is about. Each seed says &ldquo;this little piece of the read is
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
    then make the seeding engine scream. The rest of this post is that sentence,
    unpacked.
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

<p>
    You know the BWT already. What you may not have built is the realization
    that the BWT plus two small lookup tables <em>is</em> a search engine for
    exact substrings. minibwa&rsquo;s index is
    <span class="from-bwa">almost identical to bwa-mem&rsquo;s</span>, so this
    section is really about understanding the tool both share.
  </p>

  <h2>The Burrows&ndash;Wheeler matrix, and why sorting helps</h2>
  <p>
    Take the reference, append a sentinel <code>$</code> that sorts before
    everything, and list all rotations sorted alphabetically. The first column
    (call it <b>F</b>) is just the sorted letters of the genome. The last column
    (<b>L</b>) is the BWT. The magic is the <b>LF-mapping</b>: the <i>i</i>-th
    occurrence of a letter in <b>L</b> is the same physical genome position as
    the <i>i</i>-th occurrence of that letter in <b>F</b>. That correspondence
    lets us walk the index.
  </p>

  <h2>Backward search</h2>
  <p>
    To find a pattern, we feed its letters in <em>reverse</em>. We keep a
    half-open interval <code>[lo, hi)</code> of rows in the matrix &mdash; the
    rows whose sorted suffix starts with the part of the pattern matched so far.
    Each new letter <code>c</code> shrinks the interval with two table lookups:
  </p>
  <div class="note" style="font-family:var(--mono); font-size:15px; text-align:center">
    lo &larr; C[c] + Occ(c, lo) &nbsp;&nbsp;&nbsp; hi &larr; C[c] + Occ(c, hi)
  </div>
  <p>
    <code>C[c]</code> is how many genome letters sort before <code>c</code>;
    <code>Occ(c, i)</code> is how many <code>c</code>&rsquo;s appear in the BWT
    above row <code>i</code>. The width <code>hi &minus; lo</code> is the number
    of occurrences &mdash; the moment it hits zero, the pattern isn&rsquo;t in
    the genome. Try it:
  </p>

  <div class="widget">
    <div class="wbar">Backward search on a toy genome</div>
    <div class="wbody">
      <div class="controls ipt-row">
        <label>reference
          <input type="text" id="alg-fm-ref" value="GATTACAGATTACA" size="18" maxlength="20"></label>
        <label>query
          <input type="text" id="alg-fm-qry" value="ATTACA" size="10" maxlength="14"></label>
        <button class="btn ghost" id="alg-fm-rebuild">Build index</button>
      </div>
      <div class="controls">
        <button class="btn" id="alg-fm-step">Feed next letter (backward) &rarr;</button>
        <button class="btn ghost" id="alg-fm-reset">Reset search</button>
        <span id="alg-fm-status" style="font-family:var(--mono); font-size:14px"></span>
      </div>
      <div id="alg-fm-qview" class="seq" style="text-align:center; margin:6px 0 14px"></div>
      <div style="overflow-x:auto"><table class="bwm" id="alg-fm-bwm"></table></div>
      <div id="alg-fm-explain" class="note" style="margin-top:6px"></div>
    </div>
  </div>

  <h2>From interval to genome positions</h2>
  <p>
    The interval gives a <em>count</em>, but seeding also needs the actual
    coordinates. Each row in the interval corresponds to one suffix-array entry
    &mdash; one genome position. To save memory, only a fraction of the suffix
    array is stored (minibwa samples
    <span class="is-new">1 in 16</span>, up from bwa-mem&rsquo;s 1 in 32), and
    the rest are recovered by LF-walking until you hit a sampled row. More
    samples means fewer walk steps &mdash; a small memory cost for a speed win
    that matters once you&rsquo;re doing this billions of times.
  </p>

  <div class="note key">
    <span class="label">The detail that drives the rest of the tutorial</span>
    Look at the table during a search: the active rows jump around
    <em>unpredictably</em>. Each backward step lands on a different,
    far-apart region of the BWT in memory. On a real 3 GB index that means a
    cache miss almost every step &mdash; the CPU stalls hundreds of cycles
    waiting for RAM. Hold that thought. It is the entire reason the
    performance-engineering post exists.
  </div>

  <h2>How minibwa&rsquo;s index differs in practice</h2>
  <p>
    The data structure is bwa-mem&rsquo;s, but the file format and build path
    are not. minibwa builds the BWT with
    <span class="is-new">libsais</span> by default &mdash; multi-threaded and
    much faster than bwa-mem&rsquo;s old construction, at the cost of more RAM
    (the classic bwa-mem builder is still available for tight-memory machines).
    It also stores the plain 2-bit genome in a new
    <span class="is-new"><code>l2bit</code></span> format (like UCSC&rsquo;s
    2-bit, but able to address contigs longer than 2&sup3;&sup2; bases), which
    alignment reads from directly.
  </p>
  <p>
    At the API level minibwa adopts ropebwt3&rsquo;s
    <span class="is-new">half-open intervals</span> instead of bwa-mem&rsquo;s
    closed ones &mdash; a small bookkeeping change that makes the batched
    seeding logic cleaner.
  </p>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;The BWT plus a count table lets me find every exact occurrence of a
    string by feeding its letters backward and shrinking an interval. It&rsquo;s
    fast in operations &mdash; but each step is a random jump in memory, and
    that&rsquo;s what we&rsquo;ll have to fight.&rdquo;
  </div>

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
          <input type="range" id="alg-smem-minlen" min="3" max="12" value="6">
          <b id="alg-smem-minlen-v" style="font-family:var(--mono)">6</b></label>
        <button class="btn ghost" id="alg-smem-clearmm">Clear mismatches</button>
      </div>
      <p class="hint" style="margin-top:0">reference region the read came from:</p>
      <div id="alg-smem-ref" class="track"></div>
      <p class="hint">read (click any base to toggle a mismatch):</p>
      <div id="alg-smem-read" class="track"></div>
      <svg id="alg-smem-seedbars" viewBox="0 0 700 70"></svg>
      <div id="alg-smem-seedinfo" class="note" style="margin-top:4px"></div>
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
    minibwa&rsquo;s reimplemented SMEM finder makes the second
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
          <input type="range" id="alg-chain-noise" min="0" max="40" value="14">
          <b id="alg-chain-noise-v" style="font-family:var(--mono)">14</b></label>
        <label>indel in read
          <input type="checkbox" id="alg-chain-indel"></label>
        <button class="btn" id="alg-chain-run">Run chaining</button>
        <button class="btn ghost" id="alg-chain-reroll">New seeds</button>
      </div>
      <svg id="alg-chain-plot" viewBox="0 0 420 420" style="margin:0 auto"></svg>
      <div id="alg-chain-info" class="note" style="margin-top:4px"></div>
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
    re-seeding above deliberately produces overlapping seeds). If you
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
    each to the alignment stage, where base-level alignment begins.
  </p>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;On a read-vs-reference dotplot, the true alignment is a string of
    seeds on one diagonal. Chaining is the dynamic program that finds that
    string, paying a penalty for off-diagonal jumps. minibwa uses
    minimap2&rsquo;s chaining, tweaked so overlapping variable-length seeds
    don&rsquo;t double-count.&rdquo;
  </div>

<h2>Base alignment: only pay DP in small boxes</h2>
<p>
  Chaining gives a plausible genomic location and a run of anchors, but it does
  not tell you the exact mismatches, insertions, deletions, or CIGAR string.
  That is where Smith&ndash;Waterman dynamic programming comes back, just in a much
  smaller box: the short gaps between chained seeds and the read ends.
</p>
<p>
  minibwa uses minimap2&rsquo;s <span class="from-mm2">ksw2</span> routines for that
  base-level work: vectorized Smith&ndash;Waterman with affine gap costs. The
  algorithmic shape stays the same: exact seeds explain most of the read, the
  chain chooses the candidate locus, and DP cleans up only the places exact
  matching could not explain.
</p>
<div class="note feynman">
  <span class="label">Say it out loud</span>
  &ldquo;minibwa maps by finding SMEM seeds with a BWT/FM-index, chaining the seeds
  that sit on one diagonal, and running SIMD alignment only in the small gaps
  left between anchors.&rdquo;
</div>
<p>That is the algorithm. The next layer is why this same algorithm gets much faster when you reshape the hot loops around memory latency and cheap filters.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>
<script>
/* minibwa p1 */
(function(){

/* ---- cost widget ---- */
(function(){
  const svg = document.getElementById("alg-cost-viz");
  const nr = document.getElementById("alg-cost-nreads"), rl = document.getElementById("alg-cost-rlen");
  const nrv = document.getElementById("alg-cost-nreads-v"), rlv = document.getElementById("alg-cost-rlen-v");
  const txt = document.getElementById("alg-cost-text");
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
  const svg = document.getElementById("alg-sca-viz");
  const cap = document.getElementById("alg-sca-cap");
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
  document.getElementById("alg-sca-step").onclick = ()=>{ step=(step+1)%steps.length; draw(); };
  document.getElementById("alg-sca-reset").onclick = ()=>{ step=0; draw(); };
  draw();
})();
})();
</script>

<script>
/* minibwa p2 */
(function(){
/* ---- toy FM-index ---- */
let FM = null;
function buildFM(text){
  const T = text.toUpperCase().replace(/[^ACGT]/g,"") + "$";
  const n = T.length;
  const sa = [...Array(n).keys()].sort((a,b)=> T.slice(a) < T.slice(b) ? -1 : 1);
  const bwt = sa.map(i => T[(i + n - 1) % n]);
  const rows = sa.map(i => T.slice(i) + T.slice(0,i));   // full rotations for display
  // C[]
  const alpha = ["$","A","C","G","T"];
  const cnt = {}; alpha.forEach(c=>cnt[c]=0);
  for (const ch of T) cnt[ch]++;
  const C = {}; let acc=0;
  for (const c of alpha){ C[c]=acc; acc+=cnt[c]; }
  // Occ prefix: occ[c][i] = #c in bwt[0..i)
  const occ = {}; alpha.forEach(c=>occ[c]=new Array(n+1).fill(0));
  for (let i=0;i<n;i++) for (const c of alpha) occ[c][i+1] = occ[c][i] + (bwt[i]===c?1:0);
  return { T, n, sa, bwt, rows, C, occ };
}

let search = null; // {q, k:index into q from right, lo, hi, done, fail}
function startSearch(q){
  q = q.toUpperCase().replace(/[^ACGT]/g,"");
  search = { q, pos: q.length, lo: 0, hi: FM.n, fail:false };
}
function stepSearch(){
  if (!search || search.pos<=0 || search.fail) return;
  search.pos--;
  const c = search.q[search.pos];
  const lo = FM.C[c] + FM.occ[c][search.lo];
  const hi = FM.C[c] + FM.occ[c][search.hi];
  search.lo = lo; search.hi = hi;
  if (hi <= lo) search.fail = true;
}

function render(){
  // query view: matched suffix highlighted
  const q = search.q;
  const matchedFrom = search.pos;
  let qv = "";
  for (let i=0;i<q.length;i++){
    const cls = i>=matchedFrom ? "" : "";
    const active = (i>=matchedFrom);
    qv += `<span class="base ${q[i]}" style="${active?'':'opacity:.3'}">${q[i]}</span>`;
  }
  document.getElementById("alg-fm-qview").innerHTML = qv || "<span style='color:var(--ink-soft)'>(empty query)</span>";

  // table
  let html = "<tr><td class='idx'></td><td class='idx'>F</td>";
  const colw = FM.rows[0].length;
  for (let j=1;j<colw-1;j++) html += "<td class='idx'></td>";
  html += "<td class='idx'>L=BWT</td></tr>";
  for (let i=0;i<FM.n;i++){
    const active = i>=search.lo && i<search.hi;
    html += `<tr class="${active?'active':''}"><td class="idx">${i}</td>`;
    const r = FM.rows[i];
    for (let j=0;j<r.length;j++){
      const cls = j===0 ? "f" : (j===r.length-1 ? "l" : "");
      html += `<td class="${cls}">${r[j]}</td>`;
    }
    html += "</tr>";
  }
  document.getElementById("alg-fm-bwm").innerHTML = html;

  // status + explain
  const w = search.hi - search.lo;
  const st = document.getElementById("alg-fm-status");
  const ex = document.getElementById("alg-fm-explain");
  if (search.fail){
    st.innerHTML = `<b style="color:var(--accent)">not found</b> — interval collapsed`;
    ex.innerHTML = `<span class="label">Dead end</span>The interval width hit 0: the suffix <code>${search.q.slice(search.pos)}</code> does not occur in this reference. In a real run, the seed would simply stop extending here.`;
  } else if (search.pos === 0){
    const pos = FM.sa.slice(search.lo, search.hi).sort((a,b)=>a-b);
    st.innerHTML = `interval [${search.lo}, ${search.hi}) &nbsp; width=<b>${w}</b>`;
    ex.innerHTML = `<span class="label">Done — ${w} occurrence${w!==1?'s':''}</span>The full query <code>${search.q}</code> occupies ${w} row${w!==1?'s':''}. Those rows' suffix-array values are genome positions <b>${pos.join(", ")}</b>. That's the seed's hit list.`;
  } else {
    const next = search.q[search.pos-1] || "·";
    st.innerHTML = `interval [${search.lo}, ${search.hi}) &nbsp; width=<b>${w}</b> &nbsp; matched <code>${search.q.slice(search.pos)||"(none yet)"}</code>`;
    ex.innerHTML = `<span class="label">${w} rows still match</span>So far we've matched the suffix <code>${search.q.slice(search.pos)||"(nothing)"}</code> backward. Next we'll prepend <code>${next}</code> and apply <code>C[${next}]+Occ(${next}, lo/hi)</code> to shrink the highlighted band.`;
  }
}

function rebuild(){
  FM = buildFM(document.getElementById("alg-fm-ref").value);
  startSearch(document.getElementById("alg-fm-qry").value);
  render();
}
document.getElementById("alg-fm-rebuild").onclick = rebuild;
document.getElementById("alg-fm-reset").onclick = ()=>{ startSearch(document.getElementById("alg-fm-qry").value); render(); };
document.getElementById("alg-fm-step").onclick = ()=>{ stepSearch(); render(); };
document.getElementById("alg-fm-qry").addEventListener("change", ()=>{ startSearch(document.getElementById("alg-fm-qry").value); render(); });
rebuild();
})();
</script>

<script>
/* minibwa p3 */
(function(){
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
  const minlen = +document.getElementById("alg-smem-minlen").value;
  document.getElementById("alg-smem-minlen-v").textContent = minlen;
  // ref track
  document.getElementById("alg-smem-ref").innerHTML = refRegion.split("")
    .map(b=>`<span class="cell ref base ${b}">${b}</span>`).join("");
  // read track
  document.getElementById("alg-smem-read").innerHTML = readBases
    .map((b,i)=>`<span class="cell base ${b} ${mism.has(i)?'mm':''}" data-i="${i}">${b}</span>`).join("");
  document.querySelectorAll("#alg-smem-read .cell").forEach(c=>{
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
  const svg = document.getElementById("alg-smem-seedbars");
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
  document.getElementById("alg-smem-seedinfo").innerHTML =
    `<span class="label">${seeds.length} seed${seeds.length!==1?'s':''}, ${mism.size} mismatch${mism.size!==1?'es':''}</span>`+
    `Each bar shows length and how many times that exact string occurs in the full reference. `+
    (reseedCount? `<b style="color:var(--new)">${reseedCount}</b> long seed${reseedCount!==1?'s':''} occur more than once — those are the ones pass&nbsp;2 would dig into for shorter, unique sub-seeds.`
                : `None are both long and repetitive right now, so pass&nbsp;2 finds nothing extra. Try removing mismatches to grow one long seed across the <code>TTGCAG</code> repeat.`);
}
document.getElementById("alg-smem-minlen").oninput = renderTracks;
document.getElementById("alg-smem-clearmm").onclick = ()=>{ mism.clear(); apply(); };
apply();
})();
</script>

<script>
/* minibwa p5 */
(function(){
const svg=document.getElementById("alg-chain-plot");
const SZ=420, PAD=34, N=100; // coordinate space 0..N
let anchors=[], chosen=[];

function rng(seed){ let s=seed; return ()=> (s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff; }
let seed0=7;

function gen(){
  const r=rng(seed0);
  anchors=[];
  const indel=document.getElementById("alg-chain-indel").checked;
  // true diagonal anchors: y = x + offset, with optional indel jump partway
  const off=18, nTrue=8;
  for (let i=0;i<nTrue;i++){
    let x = 8 + i*10 + r()*3;
    let y = x + off + (indel && i>=4 ? 12 : 0); // a deletion: ref jumps ahead
    const w = 4 + r()*5;
    anchors.push({x,y,w,true:true});
  }
  const noise=+document.getElementById("alg-chain-noise").value;
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
  const info=document.getElementById("alg-chain-info");
  if (score===undefined){
    info.innerHTML=`<span class="label">${anchors.length} seeds</span>Blue = seeds on the true diagonal, gray = spurious. Press <b>Run chaining</b> to find the best colinear chain (drawn in green).`;
  } else {
    const trueIn = chosen.filter(i=>anchors[i].true).length;
    info.innerHTML=`<span class="label">chain score ${score.toFixed(0)}</span>The DP picked ${chosen.length} anchors (${trueIn} of them true), ignoring the off-diagonal scatter. `+
      (document.getElementById("alg-chain-indel").checked? `Notice it spans the diagonal jump in the middle — that's the deletion, paid for once as a gap cost rather than breaking the chain.` : `Every chosen anchor sits on one diagonal: a clean, contiguous placement.`);
  }
}
document.getElementById("alg-chain-noise").oninput=e=>{document.getElementById("alg-chain-noise-v").textContent=e.target.value; gen();};
document.getElementById("alg-chain-indel").onchange=gen;
document.getElementById("alg-chain-reroll").onclick=()=>{seed0=(seed0*7+3)&0x7fff; gen();};
document.getElementById("alg-chain-run").onclick=chain;
gen();
})();
</script>
