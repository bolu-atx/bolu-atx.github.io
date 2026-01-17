#  Working with CLAUDE on a large legacy codebase C/C++

- Recently re-wrote a HUGE cross platform (win/mac) codebase of legacy C/C++ to modern C++ with claude opus, tons of issues with legacy code base, outdated build system (perl + msvc + xcodeprojects), pointers, malloc/new everywhere (intermixed too)

The project took over 2 months, key learnings along the way

# pre-dev setup
- Build system is essential, make it fast, speed up the compile/test/verification loop, we did this first with claude, rewrote everything in meson and switched to gnu/clang toolchain for all platforms
- I supplemented cross platform meson with `invoke` which helps with Makefile like automation, it's cross-platform too so i can just do `inv build` or `inv release` on all systems


# testing / verifiability is critical
- Testing is next, added catch2 tests and integrated it with 3 layers of testing, unit tests, smoke tests, parity tests
- The application has a UI so it's really difficult to test without refactoring, had claude setup test harness and mock interactions with the UI so we can test in headless mode, we called this smoke test since it attempted to mimic user behavior by sending 'fake' event messages


# asking claude to manage large codebase with proper context
- Then we needed to rewrite the high risk code, the codebase is hugely complex so it was really difficult to prompt it to do the right thing.
- Asked claude to generate documentation / explore the code on various topics
- architecture / data modeling / data flow / testing coverage / go into a rabbit hole and trace how a certain file is loaded and what are the downstream data structs and things that changed as a result of that event
- along the way, we would ask claude to look at the docs, optimize its CLAUDE.md and reference updated docs
- the docs basically become 'specialized context' for the project that claude can use
- we also ask claude to distill the key principles (i.e. functional over object, stateless over stateful, certain patterns - RAII, explicit interfaces over implicit, immutability preferred etc) and put them into a coding-style guide docs, and references that explicitly when planning epics


# developing features
- For each major feature refactor of re-write, we leveraged claude ultrathink, pass it the relevant docs context, and then ask it to plan out an epic with well defined phases, completion, and test success criterias
The planning agent doesn't do implementation, implementation is done in a separate claude session, but the planning agent is kept running and asked to verify the completion
- In this process, I tried to delegate certain epics to gemini / gpt5.2 models too, they do well enough, but claude/opus definitely the best in using tools, so i think that's the current advantage of claude over the other models. But for compelx troubleshooting / debugging, sometimes if claude gets stuck i'll pass it to the other agent.


# What still sucked even with opus 4.5:

- long term context is still not great even with all the supplementation we provided it in terms of docs (it seems to really struggle in weighing them, which one is most important, which one can be violated)
- opus 4.5 is fairly conservative, given a refactoring task it'll try to do it incrementally, which sometimes actually adds complexity to a project, in these situations, words like "you can do it, be bold & aggressive, you are super-human, we've already setup tests, you can go all out" seemed to help it taking bolder steps - could also just be my hallucination tho.
- going off the beaten path will hurt you - this project is so esoteric and closed source, claude probably had limited training of the style / concerns of this project - even with this limitation, it did pretty great, much better than other LLMs or even sonnet, but still, it struggled quite a bit, i can see the gap closing tho.


# overall conclusion

- this project couldn't have been done with a single dev, it would've taken 10 people at least a quarter to do before claude opus 4.5 is available, I'm truly impressed and scared of what's coming next. It is great to extract myself from the actual hands-on coding but also scary at the same time - I worry about new generation of devs or software engineers lacking the hands on exercise to be able to tell good code from bad code the minute we start doing something off the beaten path









