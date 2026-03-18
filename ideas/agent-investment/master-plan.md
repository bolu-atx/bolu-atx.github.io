# Blog Series: Building an Agent-Native Investment System

Series working title: **"Agents Don't Read Dashboards"**

Premise: I'm building a personal investment decision system where deterministic pipelines gather and compute everything, an AI agent does the judgment calls, and I make the final decisions. No auto-trading. No black boxes. This series documents the architecture, the ideas behind it, and the build itself.

Each post should stand alone but build on the previous. The research docs (auto-research-inspiration.md, designing-software-for-agents.md) provide the theoretical backbone — the posts translate those ideas into concrete, applied decisions made while building a real system.

---

## Post 1+2 (combined): Why I'm Building a Financial Copilot — and Designing It for Agents

**Core argument:** The tools available for personal investors exist at two extremes — spreadsheets you maintain by hand, or black-box robo-advisors that hide all reasoning. The interesting design point is in the middle: a system that does the tedious data gathering and computation automatically, surfaces structured context to an AI agent for judgment, and leaves final decisions to you.

### Content outline

**Opening hook — the weekly ritual problem**
- Walk through what a manual weekly portfolio review actually looks like: open 4 tabs, check positions, read earnings transcripts, scan macro indicators, check allocation drift, write notes. It takes 2-3 hours and the output is inconsistent week to week.
- The core frustration: it's not that the analysis is hard, it's that the *context assembly* is hard. By the time you've gathered everything, you've burned your attention budget on logistics.

**The two failure modes of existing tools**
- *Manual extreme:* spreadsheets, Notion boards, note-taking apps. You own the reasoning but the data gathering is unsustainable. Context decays between sessions — you forget what you concluded last week about NVDA's earnings.
- *Automated extreme:* robo-advisors, algorithmic trading. The reasoning is hidden. You can't inspect why it recommended trimming a position. When market conditions change, you can't adjust the model's beliefs — you can only turn it off.
- Why neither extreme works for someone who actually wants to *think* about their investments but doesn't want to spend 3 hours on data entry.

**The three-layer model**
- Introduce the architecture: Layer 1 (automated) → Layer 2 (agent) → Layer 3 (human).
- Layer 1: deterministic. Fetches portfolio data, market quotes, news, SEC filings, macro indicators. Computes risk metrics (beta, Sharpe, drawdown, concentration). Detects allocation drift. Stores everything with timestamps and provenance. No judgment, no LLM calls — just structured data gathering and math.
- Layer 2: agent. Reads the structured context Layer 1 produced. Evaluates thesis conditions against evidence. Assigns rubric scores. Writes narrative analysis. Produces recommendations (BUY_MORE / HOLD / TRIM / EXIT). This is where reasoning happens, but it's constrained by the rubric and the data.
- Layer 3: human. Reads the report, reviews the evidence, makes the actual decision. The system is advisory-only — it never executes trades.
- Connect to the autoresearch paradigm: Karpathy's autoresearch uses the same pattern. The "plant" (train.py) is the thing being optimized. The agent reasons about it. The human sets the program.md constraints. My system applies this to investment decisions instead of neural network architectures.

**Why "no LLM SDK in the codebase" matters**
- The system doesn't call OpenAI or Anthropic APIs directly. All AI reasoning happens in the calling agent session.
- Why this is a feature: any agent can drive the system. Claude Code today, a custom Agent SDK app tomorrow, or a human filling in scores by hand.
- The system is a structured data tool. The intelligence is pluggable.
- Fallback is natural: without an agent, you still get a complete data-gathering report with empty scores. A human can fill them in.

**The data flow diagram**
- Include the system data flow from the spec (adapted as a Mermaid diagram).
- Walk through it top to bottom: sources of truth → fetch + save → derived watchlist → context assembly → score + judge → decisions + report.

**What this series covers**
- Brief preview of the remaining posts in the series.

### Key diagrams
- Three-layer model (simple vertical stack)
- System data flow (adapted from spec, Mermaid)

### Tone/style notes
- Start with the personal frustration — make the reader feel the pain of manual weekly reviews before introducing the architecture.
- Keep it grounded: "I built this because I was tired of forgetting what I thought about NVDA two weeks ago."
- Avoid grand claims about AI replacing analysts. This is a personal tool.

---

## ~~Post 2~~ (merged into Post 1+2 above)

Content from the original Post 2 (CLI HATEOAS, truncation, credit assignment, tool-calling trap, no LLM SDK) was merged into the combined Post 1+2. See the drafted post: `_posts/2026-03-18-agents-dont-read-dashboards-part1.md`.

---

## Post 3: Thesis-Driven Investing as a Data Structure

**Core argument:** Most investment tools organize around tickers. But you don't invest in tickers — you invest in stories. "AI infrastructure spending will accelerate" is a thesis that implicates NVDA, AMD, AVGO, and TSM in different roles. Encoding this structure in YAML, with per-symbol conditions and provenance tracking, transforms vague conviction into auditable, agent-readable context.

### Content outline

**Opening hook — the ticker list problem**
- Show a typical watchlist: AAPL, NVDA, TSLA, AMZN, GOOG. Flat list, no context.
- Ask: why is NVDA on this list? Is it a core holding or a speculative bet? What would make you sell it? What evidence would confirm or disconfirm your thesis?
- A month later, you've forgotten. The ticker is still on the list but the reasoning is gone. This is the fundamental failure of ticker-centric tools.

**Theses are stories, not tickers**
- Introduce the thesis concept: a narrative about a structural change that creates an investment opportunity. Multiple tickers play different roles in the same story.
- Example thesis: "AI infrastructure spending extends" → NVDA (primary, full conviction), AMD (secondary, half conviction), AVGO (secondary, custom silicon exposure), SMH (hedge, broad semiconductor basket).
- Each symbol has: role (primary/secondary/hedge/adhoc), conviction (full/half/starter), conditions (add triggers, trim triggers, exit triggers), and price targets.
- The thesis also has: catalysts (earnings dates, regulatory decisions), qualitative signals (confirming/disconfirming evidence checklist), and status (active/archived/draft).

**The YAML schema**
- Walk through a complete thesis YAML file. Show the actual structure.
- Why YAML and not a database: human-editable, version-controlled, diffable. The agent and the human edit the same file. Git history shows exactly when and why a thesis changed.
- The dual persistence model: YAML is authoritative for reading. DB is append-only audit log. `save_thesis()` writes YAML to disk AND inserts a `thesis_snapshots` row.
- Why both: YAML gives you current state. DB gives you history. "What did this thesis look like 3 months ago?" "When did we add AMD?" "How did conviction change over time?"

**Derived watchlists and provenance**
- There is no manually-maintained watchlist file. The watchlist is computed: all symbols from active theses ∪ portfolio holdings.
- Every symbol has provenance: `{symbol: [source_labels]}`. NVDA might appear because it's in the "AI infrastructure" thesis AND it's a current holding. AAPL might appear only because you hold it — there's no active thesis for it.
- Why provenance matters: a symbol without a thesis is a flag. You hold it, but why? The system surfaces this as an action item: "AAPL has no active thesis — consider creating one or reviewing the position."

**How thesis structure feeds the agent**
- Each symbol's scoring packet includes: the thesis narrative, the symbol's role and conviction, the per-symbol conditions, and all the data Layer 1 gathered (price action, earnings, news, SEC filings, insider transactions).
- The agent evaluates the evidence against the conditions. "Add trigger: data center revenue growth > 20% YoY. Evidence: Q4 earnings showed 28% growth. Signal: confirming."
- This is radically different from "here's everything about NVDA, tell me what to do." The thesis provides the evaluation framework. The agent applies it.

**A worked example**
- Walk through a complete cycle: thesis definition → data gathering → scoring packet → agent evaluation → recommendation.
- Show how the same evidence (e.g., a competitor gaining market share) might trigger different recommendations depending on the symbol's role in the thesis (trim a secondary position, but hold a primary with higher conviction).

### Key diagrams
- Thesis → symbols relationship (Mermaid: one thesis node connected to multiple symbol nodes with role labels)
- Data flow: thesis + market data → scoring packet → agent evaluation → recommendation

### Code examples
- A complete thesis YAML file
- The derived watchlist output with provenance
- A scoring packet (the context the agent actually sees)

### Tone/style notes
- This is the most "investment thinking" post. The reader should come away understanding why thesis-driven structure makes investment reasoning more consistent, not just more automated.
- Show that this structure benefits humans too — even without an agent, having explicit conditions and provenance makes your own thinking sharper.

---

## Post 4: Rubric Scoring — Making Agent Judgment Consistent

**Core argument:** The hardest problem in agent-assisted decision-making isn't getting the agent to reason — it's getting it to reason *consistently*. Without structure, the same agent given the same data on two different days will produce different recommendations. Scoring rubrics with versioned dimensions, anchor descriptions, and deterministic composite math solve this.

### Content outline

**Opening hook — the inconsistency problem**
- Show two hypothetical agent evaluations of the same stock with the same data, producing different recommendations. Monday: "strong buy, momentum is positive." Wednesday: "hold, valuation is stretched." Both are plausible narratives. Neither is grounded.
- The root cause: when you ask an agent to make a holistic judgment, it's doing vibes-based reasoning. The output depends on which aspects of the data it happens to attend to.

**Rubric definitions as evaluation contracts**
- A rubric is a versioned set of evaluation dimensions. Each dimension has: a name, a scale (1-5), anchor descriptions for each scale point, and an evaluation prompt template.
- Example dimensions for a stock evaluation rubric: thesis alignment (0.4 weight), technical momentum (0.15), fundamental quality (0.2), risk/reward (0.15), macro alignment (0.1).
- The anchor descriptions are critical. "Thesis alignment: 5 = all conditions met, catalysts firing, confirming evidence strong. 3 = mixed signals, some conditions met. 1 = disconfirming evidence dominates, exit conditions approaching."
- The agent doesn't decide what to evaluate or how to weight it. The rubric defines the dimensions and the anchors. The agent's job is narrow: read the context, evaluate each dimension against the anchors, assign a score, cite evidence.

**The scoring packet**
- Walk through what the agent actually receives for a single symbol: thesis narrative, role, conviction, conditions, price targets, plus all the gathered data (quotes, earnings, news, SEC filings, insider transactions, macro indicators).
- The rubric template is rendered with this context into an evaluation prompt. The agent fills in scores and evidence citations.
- Show a completed scorecard: dimension → score → evidence → brief reasoning.

**Deterministic composite math**
- The composite score is a weighted average of dimension scores. This is deterministic — no LLM involved.
- Action thresholds map composite scores to recommendations: ≥4.0 → BUY_MORE, 3.0-3.9 → HOLD, 2.0-2.9 → TRIM, <2.0 → EXIT.
- Why this split matters: the agent provides the judgment (individual dimension scores). The system provides the decision framework (weights, thresholds, actions). If a recommendation seems wrong, you can inspect exactly which dimension drove it and whether the evidence supports the score.

**Versioning and drift detection**
- Rubrics are versioned. When you change the dimensions or weights, it's a new version. Historical scores are always evaluated against the rubric version that was active at the time.
- This enables longitudinal analysis: "has the agent's thesis alignment scoring drifted over time?" "do the composite scores correlate with actual returns?"
- Connect to autoresearch: Karpathy's val_bpb is an unambiguous scalar objective. Rubric composites serve the same function — they give the agent a clear target and make the output measurable.

**Portfolio-level risk as a Layer 1 concern**
- Some metrics don't need agent judgment at all: position concentration, sector drift, portfolio beta, correlation flags, max drawdown, Sharpe ratio.
- These are pure computation on position data. Layer 1 calculates them and flags threshold breaches.
- The agent incorporates risk flags into its reasoning ("concentration in semiconductors is above policy threshold, which increases the bar for adding more NVDA"), but the detection is deterministic.

**Income and tax awareness**
- Similarly deterministic: dividend yields, ex-dates, payout history, realized gains/losses, tax-loss harvesting candidates.
- The system computes these and surfaces them. The agent factors them into recommendations ("NVDA has a short-term loss that could be harvested, but the thesis is strong — flag for human review").

### Key diagrams
- Rubric evaluation flow: scoring packet → rubric template → agent evaluation → dimension scores → deterministic composite → recommendation
- Portfolio risk dashboard: show the quantitative metrics as a structured output

### Code examples
- A rubric definition (dimensions, scales, anchors, weights)
- A rendered evaluation prompt (rubric template + context)
- A completed scorecard with composite calculation

### Tone/style notes
- This post should make the reader think about consistency in any AI-assisted judgment task, not just investing. The rubric pattern is generalizable.
- Emphasize that rubrics don't replace agent reasoning — they channel it. The agent is still doing the hard work of evaluating evidence against criteria. The rubric just ensures it evaluates the same criteria every time.

---

## Post 5: From CLI to Telegram — Three Interfaces, One Core

**Core argument:** The investment system needs to work in three modes: interactive CLI for deep analysis, a service daemon for automated weekly runs, and a Telegram bot for reactive queries and proactive alerts. All three share the same core — the interfaces are thin adapters over application services, not separate systems.

### Content outline

**Opening hook — the notification that matters**
- "It's Tuesday morning. Your phone buzzes. A Telegram message: 'Allocation drift alert: semiconductor concentration at 34% (policy max: 30%). NVDA up 8% since last review. Thesis: AI infrastructure. Suggested action: review trim triggers.' You glance at it, decide to look at the full report tonight, and reply 'show me the full thesis scorecard for NVDA.' Thirty seconds later, the bot responds with a structured summary."
- This is the end state. Let me show you how we get there.

**The CLI as the foundation**
- Everything starts as a CLI command. `bof review run`, `bof thesis list`, `bof portfolio snapshot`, `bof watchlist show`.
- The CLI is the primary interface for development and deep analysis. It's where you run ad-hoc queries, edit theses, inspect raw data.
- All CLI commands share the same application services. The CLI is a thin layer that parses arguments, calls the service, and formats the output (JSON by default, with optional human-readable rendering).

**Service mode — the autonomous loop**
- A long-running harness that orchestrates agent sessions without manual invocation.
- Trigger types: cron (weekly review every Sunday night), alerts (price crosses threshold, allocation drifts past policy), events (earnings approaching for a held symbol).
- When a trigger fires: assemble context (trigger payload + investment policy + relevant long-term memory + current portfolio state) → spawn headless agent session → agent executes the workflow using CLI commands as tools → capture output → route results (report to disk, summary to Telegram, events to audit log).
- The agent in service mode is autonomous but constrained: it can only invoke the system's own CLI commands. It doesn't have raw internet access or filesystem access beyond the project directory.

**Agent orchestration mechanics**
- How does a headless agent session actually work? The service assembles a context payload (similar to a program.md in autoresearch), spawns an agent process (Claude Code, Agent SDK app, etc.), and provides the CLI as the tool interface.
- The agent reads the context, decides which commands to run, gathers data, evaluates rubrics, and writes back structured results.
- Connect to autoresearch: this is the same pattern as Karpathy's overnight loop. The "plant" is the portfolio state. The agent modifies nothing — it only reads and evaluates. The constraints come from the investment policy and rubric definitions.

**Memory tiers**
- Short-term: per-session, ephemeral. The current review context, in-progress conversation state. Passed as prompt context, discarded after session ends.
- Long-term: across sessions, durable. Past review outcomes, thesis evolution, cross-week observations ("AAPL has beaten earnings 4 quarters in a row"), what worked and what didn't. Agent reads relevant entries at session start, writes new observations at session end.
- Investment policy: rarely changes, human-edited. Investment philosophy, risk tolerance, time horizon, sector convictions, behavioral guardrails. Equivalent to an Investment Policy Statement (IPS). Loaded into every agent session as foundational context.
- Connect to research: this maps directly to the agentic memory tiers in the designing-software-for-agents doc. Short-term = context window. Long-term = structural note-taking / persistent memory. Policy = system prompt / guardrails.

**The Telegram interface**
- Two modes: proactive (system-initiated) and reactive (human-initiated).
- Proactive: weekly review summary, alert notifications, earnings reminders. The service sends these after an agent session completes or when an alert fires.
- Reactive: human asks a question ("how is NVDA doing?", "what's my allocation drift?", "run a review now"). The service spawns an agent session to gather data and compose a reply.
- Design constraint: Telegram messages have a character limit and the reader (me) is on a phone. The response must be concise — executive summary level, with an option to request the full report.
- Why Telegram and not Slack/email/SMS: personal tool, personal messaging. Telegram has a good bot API, supports Markdown formatting, and I already use it.

**How alerts work**
- Portfolio-level event predicates wired into an event service: price alerts, allocation drift alerts, earnings proximity alerts, dividend ex-date alerts.
- Alerts are evaluated on each data refresh (at minimum, daily). When an alert fires, it creates an event with the trigger details and routes a notification to Telegram.
- Alerts are not agent-driven — they're deterministic predicates on portfolio state. The agent only gets involved if the human requests analysis in response to an alert.

### Key diagrams
- Three interfaces sharing one core (CLI / Service / Telegram all connecting to the same application services layer)
- Service mode trigger → context assembly → agent session → output routing flow
- Memory tier diagram (short-term / long-term / policy with arrows showing read/write patterns)

### Tone/style notes
- This is the most "engineering" post. Walk through the architecture practically.
- The Telegram angle makes it tangible — readers can imagine getting these notifications on their own phone.
- Emphasize that the three interfaces are thin. The core is the application services. Adding a fourth interface (web dashboard, MCP server) is straightforward because the hard work is in the core.

---

## Post 6: Build Diary — What Broke and What Surprised Me

**Core argument:** This is the living post. Updated as the build progresses. Captures the gap between the design docs and reality — what worked as planned, what required rethinking, and what I didn't anticipate.

### Content outline (seeds — expanded as the build progresses)

**Data source reliability**
- Which APIs are solid and which are flaky? Rate limits, data quality issues, coverage gaps.
- SEC EDGAR vs. commercial data providers. Free tier limitations.
- The practical challenge of getting clean, timely brokerage data.

**Agent consistency in practice**
- How well do rubric scores hold up across sessions? Is the agent anchoring on the same evidence?
- Cases where the rubric structure helped catch reasoning errors.
- Cases where the agent's reasoning was better than the rubric — signals that the dimensions need updating.

**Context window management in production**
- How much context does a full weekly review actually consume?
- Where truncation helped and where it hid important details.
- The tension between comprehensive context and focused evaluation.

**The thesis maintenance problem**
- Theses decay. Markets change. How often do theses need updating?
- Can the agent help identify when a thesis is stale? (e.g., "this thesis hasn't been modified in 3 months but the market conditions have changed significantly")

**Service mode operational lessons**
- Reliability of the overnight agent loop.
- Error recovery: what happens when a data source is down during the weekly review?
- Memory accumulation: does long-term memory actually improve agent performance over time, or does it just add noise?

**Surprising wins and failures**
- Things that worked better than expected.
- Things that required complete rethinking.
- Features I thought were critical that turned out to be unnecessary.
- Features I didn't plan for that became essential.

### Tone/style notes
- This post is more informal than the others. Stream of consciousness is fine.
- Date-stamped entries. Each entry is a short observation with context.
- This is the post that makes the series feel real — it's not just architecture diagrams, it's a person building a thing and learning from it.

---

## Series-level notes

**Publishing cadence:** Posts 1-3 can be written before the system is fully built — they cover the problem, the design philosophy, and the data model. Posts 4-5 benefit from having working code to show. Post 6 is ongoing.

**Cross-references:** Each post should link to the others in the series. Use a consistent series header (like the pipeline-thinking series).

**Code examples:** Use real code from the bo-finance repo where possible. Sanitize any personal financial data but keep the structure authentic.

**Research integration:** The auto-research and designing-software-for-agents docs provide the theoretical foundation. Each post should reference specific ideas from the research but translate them into the applied context. Don't dump theory — show how it influenced a concrete design decision.

**Audience:** Software engineers who invest personally and are curious about applying agent patterns to their own workflows. Not financial advisors. Not AI researchers. Smart generalists who want to see how these ideas work in practice.
