---
layout: post
title: "The AI End Game, Revisited"
date: 2026-03-20 10:00:00 -0700
tags: ai machine-learning economics
author: bolu-atx
categories: machine-learning
---

Three weeks ago I wrote about [the AI end game](/machine-learning/2026/03/01/the-end-game/) — the feedback loop between AI-driven layoffs, shrinking consumer demand, and what Citrini Research calls "ghost GDP." I sketched out four phases for how this might play out, with Phase I (the Efficiency Divergence) as the present moment and the rest as informed speculation.

Three weeks later, Phase I is accelerating faster than I expected.

<!--more-->

## The Structural Break

When I wrote the original post, Block's 4,000-person cut was the clearest signal. Now [Meta is planning layoffs affecting up to 20% of its workforce](https://www.cnbc.com/2026/03/14/meta-planning-sweeping-layoffs-as-ai-costs-mount-reuters.html) — potentially 15,000+ people from a headcount of nearly 79,000. Org flattening, middle-management cuts, reduced hiring especially for junior roles, all explicitly tied to AI efficiency. It's the same pattern Jack Dorsey described: smaller teams, more AI leverage, fewer people. In the first 74 days of 2026, over 55,000 roles were eliminated across 166 tech companies. This doesn't look isolated anymore.

But the more interesting thing isn't the headline layoffs. It's what's happening underneath them. In a normal downturn, the cycle looks like this: layoffs lead to less consumer spending, which leads to less corporate investment, which leads to more layoffs. It's a deflationary spiral, and we have decades of policy tools meant to interrupt it.

What we're seeing now is different. Employment is going down, but investment is going *up*. Meta is planning [up to $135 billion in AI capital expenditure in 2026](https://www.ibtimes.com/meta-faces-potential-20-layoffs-ai-spending-tops-135-billion-2026-3799498) — while cutting 20% of its people. The old destruction loop assumed that when companies cut, they also pulled back on spending. Instead, they're moving that spending from labor into capital. Headcount shrinks; AI capex grows.
That's the structural break. A company can grow output, ramp AI spend, and still shrink its workforce at the same time. The policy tools we built for the old loop — stimulus spending, interest rate cuts, retraining programs — don't obviously fit this new shape.

What makes this self-reinforcing is the prisoner's dilemma baked into every level of the system. If your competitor adopts AI and you don't, you lose. If the engineer next to you uses agents and you refuse, you're the one who looks slow. This isn't really something anyone gets to opt out of. The cost of non-adoption is falling behind, and the fear of falling behind is what keeps the whole thing moving.

And the same logic keeps showing up at every level. The individual contributor who doesn't pick up AI tools gets outperformed by the one who does. The team that resists agent workflows gets outshipped by the team that embraces them. The company that delays restructuring watches its margins erode against leaner competitors. The country that regulates too cautiously watches capital and talent flow to the ones that don't. At every scale — individual, team, org, company, nation — the logic is the same: adopt or get left behind. Nobody wants to be the one who cooperated while everyone else defected.[^1]

[^1]: The path isn't smooth. Amazon recently [held an engineering meeting after AI-related outages](https://www.ft.com/content/7cab4ec7-4712-4137-b602-119a44f771de) forced them to pump the brakes on AI-assisted development. This is inevitable — organizations will overshoot, break things, and pull back before figuring out the right integration points. And we should be honest that we're still in the hype cycle — the trough of disillusionment is coming, probably when the first wave of AI-heavy rewrites starts producing outages and the ROI on $135B capex quarters gets scrutinized. But the trough doesn't mean the technology was wrong. It means expectations overshot reality for a while. The internet went through the same thing. The underlying structural shift survived the dot-com bust just fine.

## Coordination Collapse

The reason this is moving faster than most people expected comes down to what AI agents actually replace. It's not just individual workers — it's the *organizational structure that needed those workers in the first place*.

Think about what a typical mid-size company spends most of its coordination budget on: breaking work across teams, passing context between roles, draft-review-refine cycles, cross-functional alignment meetings, the whole PM layer that exists mostly to keep humans synchronized. Agents don't just do the tasks faster — they cut way into the need for that coordination layer in the first place.

This is why Meta isn't just cutting costs. They're rebuilding the org around much higher AI leverage per employee. When agents handle decomposition, iteration, and synthesis, you need fewer glue roles. The junior analyst who compiled data for the senior analyst. The project manager who kept three teams in sync. The associate who drafted documents for a partner to review. A lot of those roles existed because humans are expensive to coordinate. Agents make that coordination much cheaper.

Put differently: 3-5 people with agent workflows can sometimes match the output of 20-50 people in a legacy org. Not in theory — in some workflows, right now.

What it *doesn't* mean, at least not yet, is that we've cleanly moved into Phase II. In the original framework, Phase I was about efficiency gains and Phase II was about commoditization — when everyone has agents, the market value of AI-assisted output trends toward zero. I still think we're firmly in Phase I territory. The important thing is that Phase I is moving faster than I expected, and some of the ingredients that could eventually push us toward Phase II are showing up earlier than I thought they would.

## The Toolchain Is Collapsing

One thing I didn't really see coming in the original post: the frontier AI companies are absorbing the developer toolchain.

[OpenAI acquired Astral](https://openai.com/index/openai-to-acquire-astral/) — the company behind Ruff, uv, and ty, tools that have become near-foundational to modern Python development — to integrate directly into Codex, which now has over 2 million weekly active users. [Anthropic acquired Bun](https://bun.com/blog/bun-joins-anthropic), the JavaScript runtime, bundler, and package manager, as infrastructure for Claude Code (which hit [$1 billion in run-rate revenue](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone) in November). These don't feel random. They look like a straightforward land grab for end-to-end agent capability. When an AI company owns the runtime, the package manager, and the linter, the agent doesn't just write code — it can build, test, and deploy it with a lot less friction.

Where this seems to be heading is pretty clear: AI shifts from being a *developer tool* to being *the developer*. Not for everything, not yet. But the gap between "AI helps a human developer" and "AI handles most of the workflow with human oversight" is closing faster than the tooling ecosystem expected.

And it's not just the dev toolchain. On March 18, [Stripe launched the Machine Payments Protocol](https://stripe.com/blog/machine-payments-protocol) — an open standard for agents to pay other agents and services directly, handling microtransactions, recurring payments, and fiat-to-stablecoin flows. Agents can already spin up headless browsers, send physical mail, and order sandwiches — paying per action without a human in the loop. In the original post, I put machine-to-machine commerce in Phase III (2030-2035). I don't think that means we've somehow skipped ahead to Phase III. It just means some of the infrastructure I expected later is arriving earlier, even while the broader economy still looks very Phase I to me.

For existing companies, this changes the moat question. If AI can increasingly handle end-to-end development, then technical implementation becomes less of a differentiator. What's left? Data moats — proprietary datasets that agents can't replicate. Network effects — user bases that compound in value. Platform lock-in — ecosystems that are expensive to leave. Distribution — the ability to reach customers. Those things were always important. Now they start mattering a lot more. The companies that come out of this in good shape will be the ones that spent this period building those moats, not just cutting headcount.

## The Arbitrage Window

Here's the part that matters if you're a builder.

Right now, there's a gap between what's possible with agent-native workflows and what most organizations are actually doing. The incumbents are mid-reorganization. Their AI strategy is a slide deck, a task force, and a quarterly review cycle. They're mostly cutting costs, not building new things. The efficiency gap between an agent-native team and a legacy org is enormous — and I don't think the market has really priced that in yet.

This creates a real arbitrage window:

- **Middle-layer collapse isn't priced.** Markets still assume coordination costs are fixed and that headcount scales with output. Both assumptions are now false. Consulting firms, SaaS companies with heavy customer success layers, enterprise analytics teams — all sitting on cost structures that agents make obsolete.
- **Junior talent is oversupplied.** Hiring has slowed but the talent pipeline hasn't. Smart people are available and relatively cheap. Combine that with AI leverage and you get massive output per dollar.
- **Incumbents are slow.** The competitive logic is straightforward: if you can ship in weeks what a legacy org takes quarters to plan, you win the market before they've finished their reorg.

But the window closes. Once agent adoption saturates — and it probably will faster than most automation waves because the deployment cost is so low — the advantage shifts back to whoever has data, distribution, and capital. The arbitrage exists precisely because we're in the transition. Transitions don't last forever.

## Both Things Are True

I called this the most exciting time to be in the industry, and I meant it. The cost of building something useful has never been lower. The tools have never been more powerful. The incumbents have never been more vulnerable to small, fast, opinionated teams. If you've been sitting on an idea, the conditions for going after it are about as good as they've ever been.

It's also true that a lot of people are going to get hurt. The coordination collapse I described above doesn't just eliminate middle management and junior glue roles — it also wipes out the *apprenticeship pathways* through which people develop expertise. If there are no junior analyst roles, where do future senior analysts come from? If agents handle the draft-review-refine cycle, how do new lawyers learn judgment? We're quietly dismantling the senior talent pipeline, and nobody seems to have much of a plan for rebuilding it.

And the macro picture hasn't gotten any less concerning. Employment destruction plus investment expansion is good for productivity metrics and corporate margins. It's not obviously good for the 40% of the workforce whose roles are heavy on coordination and synthesis — exactly the kind of work agents are best at absorbing. The structural demand gap from the original post doesn't feel theoretical anymore. It feels like the logical endpoint of what Meta, Block, and the next wave of companies are doing right now.

The honest way to frame this is that these aren't competing narratives — they're the same story, just seen from different places. If you're positioned to build, this is a huge opportunity. If you're sitting in the coordination layer that's being dissolved, this is a real threat. And the transition between those two positions is going to be fast, messy, and mostly unsupported by institutions that move at policy speed, not AI speed.

The game is the same one I described three weeks ago. It's just moving faster than I thought.

---

TLDR infographic by nanobanana/Gemini:

![Structural Shifts in AI — an XKCD-style infographic](/assets/posts-media/structural-shifts-in-ai.jpg)

*This is a follow-up to [The AI End Game](/machine-learning/2026/03/01/the-end-game/). Sources: [Meta planning sweeping layoffs as AI costs mount](https://www.cnbc.com/2026/03/14/meta-planning-sweeping-layoffs-as-ai-costs-mount-reuters.html) (CNBC, March 2026). [Meta AI spending tops $135B](https://www.ibtimes.com/meta-faces-potential-20-layoffs-ai-spending-tops-135-billion-2026-3799498) (IBTimes). [OpenAI to acquire Astral](https://openai.com/index/openai-to-acquire-astral/) (March 2026). [Bun is joining Anthropic](https://bun.com/blog/bun-joins-anthropic) (December 2025). [Stripe Machine Payments Protocol](https://stripe.com/blog/machine-payments-protocol) (March 2026). Citrini Research, ["The 2028 Global Intelligence Crisis"](https://www.citriniresearch.com/p/2028gic) (Feb 2026).*
