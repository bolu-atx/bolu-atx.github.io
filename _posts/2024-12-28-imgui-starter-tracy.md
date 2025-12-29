---
layout: post
title:  "Building High-Performance UIs with ImGui: Lessons from Tracy Profiler"
date:   2024-12-28 21:00:00 -0700
tags: cpp imgui graphics opengl wasm
author: bolu-atx
categories: programming
---

Tracy is a real-time, nanosecond resolution profiler used by game developers and performance engineers worldwide. What makes it remarkable isn't just its profiling capabilities - it's the fact that the entire UI, handling millions of data points with buttery-smooth 60fps rendering, is built with Dear ImGui. I recently extracted the UI boilerplate from Tracy into a standalone starter project, and the patterns I found are worth sharing.

<!--more-->

## Immediate Mode vs Retained Mode: A Paradigm Shift

Before diving into the code, let's understand why ImGui's "immediate mode" approach is fundamentally different from traditional UI frameworks.

**Retained Mode** (Qt, React, WPF) maintains a persistent object graph:

```cpp
// Retained mode - create widget once, manage its lifetime
Button* btn = new Button("Click me");
btn->setOnClick([] { doSomething(); });
layout->addWidget(btn);
// Later: update state, handle events, manage memory...
```

**Immediate Mode** (ImGui) rebuilds the UI every frame:

```cpp
// Immediate mode - declare UI intent each frame
if (ImGui::Button("Click me")) {
    doSomething();  // Runs when button is clicked
}
// No cleanup, no callbacks, no state management
```

This seems wasteful - redrawing everything 60 times per second? But modern GPUs are designed for exactly this workload. The trade-off is brilliant: you eliminate an entire category of bugs (stale state, dangling callbacks, synchronization issues) in exchange for CPU cycles that are essentially free on modern hardware.

For applications like Tracy that visualize constantly-changing data, immediate mode is perfect. There's no "dirty checking" or "reconciliation" - you just draw what you have right now.

## The Backend Abstraction: One Codebase, Two Platforms

The imgui-starter project demonstrates a clean approach to supporting both native desktop (GLFW + OpenGL) and web (Emscripten + WebGL) from the same codebase.

### The Backend Interface

```cpp
class Backend {
public:
    virtual bool Init(const Config& config) = 0;
    virtual void Run(const std::function<void()>& drawCallback) = 0;
    virtual void BeginFrame() = 0;
    virtual void EndFrame() = 0;
    virtual float GetDpiScale() const = 0;
    virtual bool ShouldClose() const = 0;
    // ...
};

// Factory creates the right backend at compile time
std::unique_ptr<Backend> Backend::Create() {
#ifdef __EMSCRIPTEN__
    return std::make_unique<BackendEmscripten>();
#else
    return std::make_unique<BackendGlfw>();
#endif
}
```

This isn't dependency injection for its own sake - it's a practical solution to genuinely different platform requirements.

### GLFW Backend: Native Desktop

The native backend uses GLFW for windowing and OpenGL 3.2 Core for rendering:

```cpp
bool BackendGlfw::Init(const Config& config) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(config.width, config.height,
                                 config.title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);  // VSync

    // ImGui setup
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    return true;
}

void BackendGlfw::Run(const std::function<void()>& drawCallback) {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        BeginFrame();
        drawCallback();
        EndFrame();
    }
}
```

The main loop is traditional: poll events, render frame, swap buffers, repeat.

### Emscripten Backend: WebAssembly

The WebAssembly backend is more involved because browsers control the main loop:

```cpp
void BackendEmscripten::Run(const std::function<void()>& drawCallback) {
    m_drawCallback = drawCallback;

    // Register input handlers with the browser
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT,
                                     this, true, KeyCallback);
    emscripten_set_mousedown_callback("#canvas", this, true, MouseCallback);
    // ... more event handlers

    // Hand control to the browser - our frame function gets called
    emscripten_set_main_loop_arg(
        [](void* arg) { static_cast<BackendEmscripten*>(arg)->Frame(); },
        this, 0, 1);
}

void BackendEmscripten::Frame() {
    UpdateCanvasSize();
    BeginFrame();
    m_drawCallback();
    EndFrame();
}
```

Instead of `while (running)`, we register a callback that the browser invokes at its preferred frame rate. This is fundamental - in a browser, blocking the main thread freezes the entire tab.

The input handling requires translating JavaScript key codes to ImGui's format:

```cpp
EM_BOOL BackendEmscripten::KeyCallback(int eventType,
    const EmscriptenKeyboardEvent* e, void* userData) {

    ImGuiIO& io = ImGui::GetIO();

    // Map browser key codes to ImGui keys
    static const std::unordered_map<std::string, ImGuiKey> keyMap = {
        {"Tab", ImGuiKey_Tab}, {"ArrowLeft", ImGuiKey_LeftArrow},
        {"ArrowRight", ImGuiKey_RightArrow}, {"ArrowUp", ImGuiKey_UpArrow},
        // ... 100+ mappings
    };

    auto it = keyMap.find(e->key);
    if (it != keyMap.end()) {
        io.AddKeyEvent(it->second, eventType == EMSCRIPTEN_EVENT_KEYDOWN);
    }
    return EM_TRUE;
}
```

### Build System: Conditional Compilation

CMake handles the platform differences:

```cmake
if(EMSCRIPTEN)
    target_compile_options(app PRIVATE
        -sUSE_WEBGL2=1
        -sFULL_ES2=1)
    target_link_options(app PRIVATE
        -sINITIAL_MEMORY=67108864
        -sALLOW_MEMORY_GROWTH=1
        --preload-file fonts@/fonts)
else()
    # Native: link GLFW and OpenGL
    find_package(OpenGL REQUIRED)
    target_link_libraries(app PRIVATE glfw OpenGL::GL)
endif()
```

The WebAssembly build preloads fonts into a virtual filesystem, while native builds load from disk at runtime.

## OpenGL Integration: Why Immediate Mode Shines

ImGui's rendering model maps naturally to modern OpenGL. Each frame:

1. ImGui builds a list of draw commands
2. Commands reference vertices in a single buffer
3. One draw call per texture change

```cpp
void BackendGlfw::EndFrame() {
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    int fb_width = (int)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
    int fb_height = (int)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);

    glViewport(0, 0, fb_width, fb_height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(draw_data);
    glfwSwapBuffers(m_window);
}
```

The `ImDrawData` structure contains everything needed for rendering: vertex buffers, index buffers, and a list of draw commands. ImGui batches similar draws together, so a complex UI might only need 10-20 draw calls.

This is where immediate mode pays off. In a retained mode framework, you'd need to track which widgets changed and update their GPU buffers. ImGui just uploads fresh data every frame - and at 60fps, uploading a few hundred KB is negligible.

## Performance Patterns from Tracy

Tracy renders profiler data with millions of zones, achieving 60fps through several key patterns:

### Two-Phase Rendering

Tracy separates preprocessing from drawing:

```cpp
void TimelineItemThread::Preprocess(const TimelineContext& ctx,
                                     TaskDispatch& td, bool visible) {
    // Queue work for background threads
    td.Queue([this, &ctx, visible] {
        m_depth = PreprocessZoneLevel(ctx, m_thread->timeline, 0, visible);
    });
    td.Queue([this, &ctx, visible] {
        PreprocessContextSwitches(ctx, visible);
    });
}
```

Background threads build compact draw lists while the main thread renders the previous frame. The draw lists use cache-friendly structures:

```cpp
struct TimelineDraw {
    TimelineDrawType type;  // 1 byte
    uint16_t depth;         // 2 bytes
    short_ptr<void*> ev;    // 4 bytes (compressed pointer)
    Int48 rend;             // 6 bytes (48-bit timestamp)
    uint32_t num;           // 4 bytes
};
```

Every byte matters when you have millions of these.

### Visibility Culling with Binary Search

Instead of iterating all zones, Tracy binary-searches to find the visible range:

```cpp
auto it = std::lower_bound(data.begin(), data.end(), vStart,
    [](const auto& l, auto r) { return l.end < r; });
auto end = std::lower_bound(it, data.end(), vEnd,
    [](const auto& l, auto r) { return l.start < r; });

// Only process visible items
while (it < end) { /* render */ ++it; }
```

For 10 million zones, this reduces work from O(n) to O(log n + visible).

### Folding Invisible Details

Items too small to see get folded into composite rectangles:

```cpp
if (zsz < MinVisNs) {
    // Find next visible item
    auto nextTime = end + MinVisNs;
    auto next = std::lower_bound(it + 1, zitend, nextTime, ...);

    // Draw single rectangle for all folded items
    draw->AddRectFilled(folded_rect, 0xFF666666);
    DrawZigZag(draw, ...);  // Visual indicator

    it = next;
}
```

This ensures the number of draw calls is proportional to pixels, not data points.

## Practical Takeaways

If you're building a data-intensive application with ImGui:

**1. Embrace the frame-by-frame model.** Don't try to cache widget state. Let ImGui handle that internally. Your job is to describe what should be on screen right now.

**2. Separate data processing from rendering.** Use background threads for heavy computation. Build compact intermediate representations that the main thread can quickly consume.

**3. Cull aggressively.** Binary search for visible ranges. Fold small items. The goal is O(pixels), not O(data).

**4. Use the docking branch.** It's been stable for years and provides proper panel management that users expect from professional tools.

**5. Invest in fonts.** Tracy uses FreeType with light hinting, multiple weights, and icon fonts. Good typography makes immediate mode UIs feel polished.

The imgui-starter project packages all of this into a clean starting point. It's not a framework - it's a distillation of patterns that work in production. Fork it, delete what you don't need, and build something fast.
