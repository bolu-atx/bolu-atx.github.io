---
layout: post
title: "How minibwa Works, Part 2: The Genome as a Searchable Index"
date: 2026-07-14 09:00:00 -0700
tags: bioinformatics algorithms sequencing minibwa programming
author: bolu-atx
categories: programming
---

Suppose I hand you the string `ATTACA` and ask: where does this exact sequence occur in the genome?

<!--more-->

Doing that once is easy enough. Doing it billions of times, while mapping hundreds of millions of reads, is the real job. minibwa inherits bwa-mem's answer: turn the reference into an FM-index, which is a Burrows-Wheeler-transform based data structure that lets us search backward one character at a time without scanning the genome.


<div class="minibwa-series">
<p>
    You know the BWT already. What you may not have built is the realization
    that the BWT plus two small lookup tables <em>is</em> a search engine for
    exact substrings. minibwa&rsquo;s index is
    <span class="from-bwa">almost identical to bwa-mem&rsquo;s</span>, so this
    chapter is really about understanding the tool both share.
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
          <input type="text" id="ref" value="GATTACAGATTACA" size="18" maxlength="20"></label>
        <label>query
          <input type="text" id="qry" value="ATTACA" size="10" maxlength="14"></label>
        <button class="btn ghost" id="rebuild">Build index</button>
      </div>
      <div class="controls">
        <button class="btn" id="step">Feed next letter (backward) &rarr;</button>
        <button class="btn ghost" id="reset">Reset search</button>
        <span id="status" style="font-family:var(--mono); font-size:14px"></span>
      </div>
      <div id="qview" class="seq" style="text-align:center; margin:6px 0 14px"></div>
      <div style="overflow-x:auto"><table class="bwm" id="bwm"></table></div>
      <div id="explain" class="note" style="margin-top:6px"></div>
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
    waiting for RAM. Hold that thought. It is the entire reason Part 4
    exists.
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
    seeding logic in the next two chapters cleaner.
  </p>

  <div class="note feynman">
    <span class="label">Say it out loud</span>
    &ldquo;The BWT plus a count table lets me find every exact occurrence of a
    string by feeding its letters backward and shrinking an interval. It&rsquo;s
    fast in operations &mdash; but each step is a random jump in memory, and
    that&rsquo;s what we&rsquo;ll have to fight.&rdquo;
  </div>

<p>So the FM-index gives us cheap exact matching in terms of operations. But the rows we touch jump around memory in a way the CPU cannot predict. That tension, cheap arithmetic wrapped around expensive memory misses, is the performance problem minibwa is really built to solve.</p>
<p class="minibwa-credit"><em>Sources and credit: this explanation is based on Heng Li and Nils Homer's <a href="https://arxiv.org/abs/2606.15357">minibwa paper</a>, and on the design lineage from <a href="https://github.com/lh3/bwa">BWA / bwa-mem</a> and <a href="https://github.com/lh3/minimap2">minimap2</a>. The interactive tutorial and widgets were drafted with Opus 4.8.</em></p>
</div>

<script>
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
  document.getElementById("qview").innerHTML = qv || "<span style='color:var(--ink-soft)'>(empty query)</span>";

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
  document.getElementById("bwm").innerHTML = html;

  // status + explain
  const w = search.hi - search.lo;
  const st = document.getElementById("status");
  const ex = document.getElementById("explain");
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
  FM = buildFM(document.getElementById("ref").value);
  startSearch(document.getElementById("qry").value);
  render();
}
document.getElementById("rebuild").onclick = rebuild;
document.getElementById("reset").onclick = ()=>{ startSearch(document.getElementById("qry").value); render(); };
document.getElementById("step").onclick = ()=>{ stepSearch(); render(); };
document.getElementById("qry").addEventListener("change", ()=>{ startSearch(document.getElementById("qry").value); render(); });
rebuild();
</script>
