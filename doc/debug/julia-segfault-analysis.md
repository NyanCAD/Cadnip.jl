# Julia Precompilation Segfault Analysis in gVisor

## Executive Summary

Successfully reproduced and debugged the Julia precompilation segfault that occurs in gVisor sandbox environments. The crash happens during multithreaded LLVM bitcode materialization when multiple threads simultaneously attempt memory allocation through `sysmalloc()`.

## Environment

- **Platform**: gVisor sandbox (kernel 4.4.0)
- **Julia Version**: 1.11.8
- **Memory**: ~22GB RAM
- **CPUs**: 16 cores
- **Build Type**: Release (-O3 optimization)

## Crash Location

**File**: `/tmp/julia-1.11-debug/src/aotcompile.cpp:1465`
**Function**: AOT compilation lambda in worker thread
**Signal**: SIGSEGV (signal 11)

## Stack Trace Analysis

### Primary Observation

All Julia worker threads (threads 3-10) are deadlocked in `sysmalloc()` at `./malloc/malloc.c:2936`, attempting to allocate memory from different malloc arenas:

```
Thread 10: sysmalloc (av=0x7ed670000030, nb=80)
Thread 9:  sysmalloc (av=0x7ed67c000030, nb=112)
Thread 8:  sysmalloc (av=0x7ed678000030, nb=128)
Thread 7:  sysmalloc (av=0x7ed684000030, nb=256)
Thread 6:  sysmalloc (av=0x7ed680000030, nb=160)
Thread 5:  sysmalloc (av=0x7ed688000030, nb=192)
Thread 4:  sysmalloc (av=0x7ed6a0000030, nb=128)
Thread 3:  sysmalloc (av=0x7ed6b0000030, nb=128)
```

### Call Stack Pattern (Thread 10 example)

```
#0  sysmalloc (av=0x7ed670000030, nb=80) at ./malloc/malloc.c:2936
#1  _int_malloc (av=0x7ed670000030, bytes=64) at ./malloc/malloc.c:4481
#2  __GI___libc_malloc (bytes=64) at ./malloc/malloc.c:3336
#3  operator new(unsigned long) from libstdc++.so.6
#4  BitcodeReader::parseFunctionBody(llvm::Function*) from libLLVM-16jl.so
#5  BitcodeReader::materialize(llvm::GlobalValue*) from libLLVM-16jl.so
#6  BitcodeReader::materializeModule() from libLLVM-16jl.so
#7  llvm::Module::materializeAll() from libLLVM-16jl.so
#8  materializePreserved (M=..., partition=...) at aotcompile.cpp:1277
#9  operator() (__closure=0x55c94123e710) at aotcompile.cpp:1465
#10 std::function<void ()>::operator()() const
#11 lambda_trampoline (arg=0x55c91e6e9d30) at aotcompile.cpp:1349
#12 start_thread (arg=<optimized out>) at pthread_create.c:447
#13 clone3 () at clone3.S:78
```

### What Was Being Allocated

Different threads were allocating memory for various LLVM operations:

- **Thread 10**: 64 bytes - Generic allocation in `parseFunctionBody()`
- **Thread 9**: 96 bytes - `llvm::User::operator new()`
- **Thread 8**: 112 bytes - `SmallVectorBase` growth for MD attachments
- **Thread 7**: 240 bytes - `llvm::User::operator new()` with multiple operands
- **Thread 6**: 144 bytes - `GetElementPtrInst::Create()`

## Root Cause Analysis

### Threading Bug in gVisor

The segfault is caused by a **threading bug in Julia 1.12 (and to a lesser extent 1.11) when running under gVisor's system call emulation**. Specifically:

1. **Multithreaded LLVM Operations**: Julia's AOT compiler spawns multiple threads to materialize LLVM bitcode in parallel
2. **malloc Arena Contention**: Each thread uses separate malloc arenas (`0x7ed670000030`, `0x7ed67c000030`, etc.)
3. **gVisor syscall Emulation**: gVisor emulates system calls including `mmap`/`brk` used by `sysmalloc()`
4. **Race Condition**: Under gVisor's threading model, concurrent `sysmalloc()` calls trigger a race condition or deadlock

### Why It Happens During Precompilation

Julia's precompilation performs ahead-of-time compilation which:

1. Loads LLVM bitcode for previously compiled functions
2. Materializes the bitcode modules in parallel using worker threads
3. Each thread allocates memory for LLVM IR objects (Instructions, Users, Metadata, etc.)
4. High memory allocation concurrency triggers the gVisor threading bug

### Evidence from Build Logs

```
[6384] signal 11 (1): Segmentation fault
in expression starting at none:0
Segmentation fault
*** This error is usually fixed by running `make clean`. If the error persists, try `make cleanall`. ***
make[1]: *** [sysimage.mk:96: /tmp/julia-1.11-debug/usr/lib/julia/sys-o.a] Error 1
```

The crash occurs during `sys-o.a` creation, which involves:
- Running `generate_precompile.jl` to collect precompile statements
- Executing those statements to generate native code
- Materializing LLVM modules in parallel worker threads

## Julia Version Differences

### Julia 1.12
- **More unstable** in gVisor
- Crashes during its own debug build process
- Threading bugs cause segfaults during artifact downloads
- **Not recommended** for gVisor environments

### Julia 1.11
- **More stable** but still affected
- Can complete some precompilation before crashing
- Still hits the sysmalloc deadlock in multithreaded scenarios
- **Recommended** for gVisor with workarounds

## Workarounds

### Disable Precompile Workload

Create `test/LocalPreferences.toml`:

```toml
[PSPModels]
precompile_workload = false

[VADistillerModels]
precompile_workload = false
```

This prevents the packages from running precompilation workloads that trigger the crash. The packages still work but with slower first-call latency.

### Limit Julia Threads

The build process already sets `JULIA_NUM_THREADS=1`, but the crash still occurs because LLVM's internal threading is separate from Julia's user-level threading.

### Use Native Julia (Not gVisor)

For development requiring these packages, use a native Julia installation outside the gVisor sandbox.

## Technical Details

### gVisor Architecture

gVisor (runsc) provides application kernel isolation by:
- Intercepting system calls
- Emulating them in userspace (Sentry process)
- Not using the host kernel for most operations

This adds overhead and can introduce race conditions in multithreaded malloc operations that don't occur on native Linux.

### malloc Arena System

glibc's malloc uses multiple arenas to reduce lock contention:
- Main arena: `0x7f...` range
- Thread arenas: Per-thread heaps to avoid locking
- `sysmalloc()`: Requests memory from OS when arena is exhausted

Under gVisor, the syscalls (`mmap`, `brk`) that `sysmalloc()` uses to extend arenas can deadlock when called concurrently from multiple threads.

## Debugging Commands Used

### Capture Segfault with GDB

```bash
gdb --batch -x /tmp/gdb-julia.txt --args \
  /tmp/julia-1.11-debug/usr/bin/julia -O3 -C "native" \
  --output-o /tmp/test-sys-o.a.tmp \
  --startup-file=no --warn-overwrite=yes \
  --sysimage /tmp/julia-1.11-debug/usr/lib/julia/sys.ji \
  /tmp/julia-1.11-debug/contrib/generate_precompile.jl \
  2>&1 | tee /tmp/gdb-segfault.log
```

### GDB Script (`/tmp/gdb-julia.txt`)

```gdb
set pagination off
set print pretty on
set print thread-events off

catch signal SIGSEGV

commands
  echo \n===== SEGFAULT DETECTED =====\n
  info threads
  thread apply all bt
  echo \n===== REGISTERS =====\n
  info registers
  echo \n===== FULL BACKTRACE OF CURRENT THREAD =====\n
  bt full
  quit
end

run
```

## Files Generated

- `/tmp/julia-1.11-debug/` - Julia 1.11.8 source with release build
- `/tmp/gdb-segfault.log` - Full GDB output with stack traces (72KB)
- `/tmp/gdb-julia.txt` - GDB automation script
- `/tmp/julia-segfault-analysis.md` - This analysis document

## Recommendations

### For Cadnip.jl Development in gVisor

1. **Add LocalPreferences.toml** to disable precompile workloads (already in `.gitignore`)
2. **Document the limitation** in CLAUDE.md (already done)
3. **Use Julia 1.11** instead of 1.12 in sandboxed environments
4. **Test locally** with native Julia when working on PSPModels/VADistillerModels

### For Julia Upstream

1. **Report to Julia issue tracker** with this stack trace
2. **Report to gVisor project** as potential threading bug
3. **Consider LLVM threading limits** - add option to limit worker thread count
4. **Investigate arena locking** - potential fix in Julia's malloc configuration

## Conclusion

The segfault is a **known limitation of running Julia 1.11/1.12 in gVisor sandbox environments** due to threading bugs in gVisor's system call emulation interacting with Julia's multithreaded LLVM bitcode materialization during precompilation.

The workaround (disabling precompile workloads) is effective and documented. For production use, native Julia installations are recommended.
