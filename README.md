CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### Windows build

With Visual Studio 2026 C++ tools, CUDA 13.4, and a compatible CMake on `PATH`:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T cuda=13.4
cmake --build build --config Debug
```

Use a fresh build directory if an existing CMake cache selects a different
generator, platform, or toolset. CMake enables MSVC's conforming preprocessor
(`/Zc:preprocessor`) for CUDA sources, as required by CUDA 13.4's Thrust headers.

In Debug and RelWithDebInfo builds, project kernels retain CUDA device debugging
(`-G`), but `thrust.cu` uses line information (`-lineinfo`) without `-G`.
Unoptimized Thrust/CUB kernels can exceed the default GPU stack and fail with
CUDA error 719. This exception preserves source line information without changing
the device-wide stack limit; full device debugging is unavailable for `thrust.cu`.

### CPU timing

`STREAM_COMPACTION_CPU_ENABLE_TIMING` in `cpu.cu` controls automatic CPU timing
for `scan`, `compactWithoutScan`, and `compactWithScan`. With the macro set to
`1`, only the call that starts `CPU::timer()` ends it; nested calls preserve the
outer measurement. Set it to `0` to disable automatic timing without affecting
explicit caller-side timing or the previous elapsed value. Empty scans return
without changing timer state. No CMake option is needed.

### Efficient timing

Automatic GPU timing in `Efficient::scan` and `Efficient::compact` is enabled by
default. A call starts and stops `Efficient::timer()` only if that timer was not
already running. Nested calls leave the outer measurement untouched; explicit
caller-side timing must use the same `Efficient::timer()` instance.

The source-level macro `STREAM_COMPACTION_EFFICIENT_ENABLE_TIMING` at the top of
`efficient.cu` defaults to `1`; set it to `0` to disable automatic timing.
No CMake option is needed. Explicit caller-side timing remains available.

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
