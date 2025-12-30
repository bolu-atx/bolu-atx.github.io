---
layout: post
title: "What I Learned Using Claude to Rewrite a Legacy C/C++ Codebase"
date: 2025-12-30 10:00:00 -0700
tags: claude ai llm cpp modernization
author: bolu-atx
categories: programming
---

I recently completed a two-month project rewriting a large cross-platform (Windows/macOS) legacy C/C++ codebase to modern C++17 using Claude Opus 4.5. The codebase had all the hallmarks of decades-old code: an arcane build system (Perl scripts + Visual Studio projects + Xcode projects), raw pointers everywhere, `malloc` and `new` intermixed, custom container libraries, and global state scattered throughout.

This post shares the key learnings from that experience.

<!--more-->

## Pre-Development: Build System First

Before any meaningful AI-assisted development, I had to address the build system. The original setup was:
- Perl scripts generating platform-specific project files
- Visual Studio `.vcxproj` for Windows
- Xcode projects for macOS
- Different compilers with different warning levels and standards compliance

This made iteration slow and error-prone. **My first task for Claude was rewriting everything in Meson + Ninja**, switching to a consistent GNU/Clang toolchain across all platforms.

The result:

| Platform | Toolchain | Build Time |
|----------|-----------|------------|
| macOS | Clang/LLVM (Xcode CLI) | ~45s |
| Windows | MinGW-w64 UCRT64 (MSYS2) | ~60s |
| Linux | GCC or Clang | ~40s |

I supplemented Meson with Python's `invoke` library for Makefile-like automation. Now I can run `inv build` or `inv release` on any platform with identical behavior. The uniform build system meant Claude could suggest build commands that actually worked, and the fast compile times (under a minute) kept the feedback loop tight.

**Lesson learned**: Don't try to AI-modernize code if you can't quickly compile and test changes. Fix the build system first.

## Testing and Verifiability

After the build system, I tackled testing. The application has a GUI, making headless testing challenging. Claude helped set up a three-tier test strategy:

### 1. Unit Tests (180+ test cases, 7000+ assertions)

Component-level tests using Catch2. These covered:
- ODE integration against analytical solutions
- Expression parsing and bytecode compilation
- Parameter management and fitting infrastructure
- Data structure round-trip serialization

### 2. Parity Tests (15 test cases, 195 assertions)

These compare the modern rewritten code against the legacy implementation to ensure numerical equivalence:

```cpp
// 1. Load and run legacy simulation
runLegacySimulation(legacySystem);

// 2. Convert to modern types
auto modern = convertToModern(legacySystem);

// 3. Run modern simulation
modernSimulator.run(modern);

// 4. Compare within tolerance
for (size_t i = 0; i < traces.size(); i++) {
    CHECK(modernTrace[i] == Approx(legacyTrace[i]).epsilon(1e-6));
}
```

Parity tests were **critical** for the modernization strategy. They let us replace legacy code incrementally while maintaining confidence that behavior remained identical.

### 3. Smoke Tests (14 test cases, 176 assertions)

End-to-end tests that load real data files and exercise full workflows without a GUI. These required building mock interaction harnesses that sent "fake" event messages to simulate user behavior.

**Testing difficulties uncovered architectural problems:**

- Global state dependencies: Core functions depended on globals normally set by the UI
- Tight coupling: Fitting code crashed when called without full UI state initialization
- Constructor semantics: The old `ZTLVec(n)` pre-allocated AND zeroed n elements; the new version only reserved capacity. Code relying on implicit zero-initialization broke silently.

These issues were documented and became targets for architectural improvements.

## Managing Context on a Large Codebase

The codebase was complex enough that naive prompting didn't work. Claude would make changes that broke invariants or conflicted with patterns established elsewhere. We developed a documentation-driven workflow:

### Building "Specialized Context"

We asked Claude to explore the codebase and generate documentation on specific topics:
- **Architecture**: Layer boundaries, startup sequence, plugin system
- **Data modeling**: How structures flow through the system, ownership semantics
- **Message systems**: Event dispatch, threading model, global state lifecycle

Each exploration produced a markdown file that became part of the project's `docs/` directory. Over time, we accumulated:
- `architecture.md` (350+ lines covering three architectural layers)
- `data-structures.md` (1000+ lines on the core hash table system)
- `testing.md` (400+ lines on test infrastructure)
- `kinetics-interface-boundary.md` (documenting legacy/modern type boundaries)

These docs became "specialized context" that Claude could reference when planning work.

### Distilling Coding Principles

We asked Claude to analyze the codebase patterns and distill them into a coding style guide. The resulting `CLAUDE.md` file included principles like:

```
## Modernization
- Valid-by-construction: Full init in ctors, std::optional for fallible
- Modern idioms: Templates over macros, std::variant over unions
- Functional > stateful: Pure functions, explicit params, no globals
- No adapters: Parity test → new code → deprecate legacy
```

This file was referenced explicitly when planning major refactors. It helped Claude stay consistent with established patterns rather than inventing new approaches for each task.

## Developing Features with Epics

For each major feature refactor, we followed a structured planning process:

### 1. Create an Epic Document

Using Claude's extended thinking mode, we'd pass relevant context (architecture docs, affected subsystems) and ask for a phased plan with:
- Clear scope and rationale
- Migration phases with specific files and changes
- Test success criteria
- API migration references

For example, the "ZTL Sunset Epic" for replacing custom containers with STL had:

```markdown
## Inventory
| Container | Member Fields | Local Variables | Function Params | Total |
|-----------|---------------|-----------------|-----------------|-------|
| ZTLVec<T> | 18 | ~20 | ~25 | ~63 |
| ZTLPVec<T> | 14 | ~10 | ~10 | ~34 |

## Migration Phases
### Phase 1: Runtime-Only Containers
Migrate all non-serialized containers. No file format concerns.

### Phase 2: Serialized Containers
Migrate containers written to .kin files. Binary-compatible.

### Phase 3: Cleanup
Delete ztl.h, remove deprecated tests, update docs.
```

### 2. Separate Planning from Implementation

The planning agent didn't implement. Implementation happened in separate Claude sessions. But the planning session was kept running to verify completion against the documented criteria.

This separation prevented the "scope creep" problem where Claude would start implementing mid-plan and lose sight of the bigger picture.

### 3. Multi-Model Strategy

I delegated certain epics to Gemini and GPT-4.5 as experiments. They performed adequately, but Claude/Opus remained superior at:
- **Tool use**: Navigating files, running tests, making targeted edits
- **Maintaining context**: Remembering constraints from earlier in the conversation

However, when Claude got stuck on complex debugging, passing the problem to another model sometimes provided fresh perspectives.

## What Still Didn't Work Well

Even with all this infrastructure, Opus 4.5 had limitations:

### Long-Term Context Weighting

Claude struggled to prioritize among multiple documentation files. Given docs on architecture, testing, and coding style, it wouldn't always know which constraints mattered most for a specific task. Sometimes it violated a critical pattern documented in one file while perfectly following another.

### Over-Conservative Refactoring

Given a modernization task, Claude defaulted to incremental changes that sometimes added complexity. For example, it would add adapter layers rather than making clean breaks:

```cpp
// Claude's incremental approach
class ModernSystem {
    LegacySystem& legacy_;  // Adapter to old code
    // ... modern fields with sync calls to legacy
};

// What we actually wanted
class ModernSystem {
    // ... modern fields only, legacy deleted
};
```

Prompts like "be bold, we have tests, make the clean break" seemed to help, though this could be confirmation bias.

### Off-the-Beaten-Path Struggles

The codebase was esoteric - scientific computing with domain-specific data structures, custom container libraries, and unusual threading patterns. Claude had likely seen limited similar code in training. It handled this surprisingly well given the constraints, better than other LLMs, but still made domain-specific errors that a human expert wouldn't.

## Conclusion

This project couldn't have been completed by a single developer. Pre-Claude, it would have taken a team of 5-10 engineers at least a quarter. The combination of:

1. **Fast build system** enabling rapid iteration
2. **Three-tier testing** ensuring correctness through changes
3. **Documentation-driven context** keeping Claude aligned
4. **Structured epics** breaking work into verifiable phases

...turned an intimidating legacy modernization into a series of tractable tasks.

The experience left me impressed by what's possible with current AI tooling, but also concerned. I could evaluate Claude's suggestions because I understood the domain, the patterns, and what "good code" looks like for this problem space. New developers relying heavily on AI tools might not develop that discernment.

For now, the combination of human expertise setting direction and AI assistance accelerating execution feels like the right balance. Ask me again in a year.
