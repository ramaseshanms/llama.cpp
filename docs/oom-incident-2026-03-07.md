# OOM Kill Incident — 2026-03-07

## System at Time of Incident

| Resource | Value |
|----------|-------|
| Instance | AWS g5.2xlarge |
| vCPU | 8 × AMD EPYC 7R32 |
| RAM | 32 GB |
| Swap | **0 MB (none configured)** |
| Disk | 100 GB NVMe (root) + 419 GB NVMe (data) |
| GPU | 1 × NVIDIA A10G (24 GB VRAM) |
| OS | Ubuntu, kernel 6.8.0-1044-aws |

### Kernel VM Settings

| Setting | Value | Meaning |
|---------|-------|---------|
| `vm.overcommit_memory` | `0` | Heuristic overcommit — kernel **can refuse** allocations if `Committed_AS >= CommitLimit` |
| `vm.overcommit_ratio` | `50` | `CommitLimit = RAM × 50% + Swap = 32 GB × 0.5 + 0 = **16 GB**` |
| `vm.swappiness` | `60` | Irrelevant — no swap device exists |

**Critical implication:** With no swap and overcommit mode 0, the effective
commit ceiling is ~16 GB. Any allocation attempt that would push
`Committed_AS` above ~16 GB causes `ENOMEM`, triggering an OOM kill.

---

## What Was Running

### Primary Process Killed

```
PID 78592: /home/ubuntu/llama.cpp/build/bin/llama-cli
           -m ~/models/llama-3.2-3B-q4_hqq.gguf
           -p "Once upon a time"
           -n 30
           --no-display-prompt
           -ngl 0
           2>/dev/null
           </dev/null
```

The process was a **CPU-only** inference run (`-ngl 0`) of the
Llama-3.2-3B model quantized in Q4_HQQ format.

### Concurrent Background Load

At the time of the kill, the following were also active in the same
session:

- **Vulkan SPIR-V shader compilation** (`glslc`) — compiling ~80+ shader
  variants for the Q4_HQQ Vulkan kernel work. Each `glslc` invocation
  can consume 200–500 MB of RAM.
- **CMake build system** (`cmake --build`) — linking C++ objects for
  `ggml-vulkan.cpp` and `vulkan-shaders-gen.cpp`, each with large
  translation units.
- **Claude Code agent** processes — several background bash tasks running
  in parallel.

---

## Sequence of Events

1. **Model load began.** `llama-cli` started and the llama.cpp spinner
   (`|-\|/-\...`) ran for several seconds while mmap-ing the 2.0 GB GGUF
   file into virtual address space.

2. **First inference completed.** The prompt `"Once upon a time"` was
   answered (32 tokens generated at ~9 t/s). Output was incoherent
   (`couldglyPPPiquer...`) because the model at that path was still in
   the old **Phase 2 format** (FP16 zero-point) while the code expected
   **Phase 3 format** (uint8 zero-point). This is a separate correctness
   bug, not related to the OOM.

3. **Interactive session loop began.** Despite `< /dev/null` closing
   stdin and `-n 30` limiting generation, `llama-cli` entered its
   interactive chat loop and began printing `> ` prompts at high
   frequency. Over the course of the run it emitted **118,462,918 `>`
   prompts**, filling a **452 MB output file** on disk.

4. **Memory pressure accumulated.** While `llama-cli` held its model
   weights (~2.0 GB mapped), the concurrent Vulkan shader builds and
   linker processes were committing additional memory. With `CommitLimit`
   hard-capped at ~16 GB and no swap to spill into, `Committed_AS` crept
   toward the ceiling.

5. **Kernel OOM killer fired.** The kernel selected PID 78592
   (`llama-cli`) as the OOM victim — likely because it had a high
   `oom_score` (large RSS, no `oom_score_adj` protection) — and sent
   `SIGKILL`.

6. **Process exited with code 137.** `128 + SIGKILL(9) = 137`. The shell
   reported:
   ```
   /bin/bash: line 1: 78592 Killed   /home/ubuntu/llama.cpp/build/bin/llama-cli ...
   ```

> **Note:** `dmesg` was not readable (`Operation not permitted`) so the
> exact OOM kill log entry (`oom_kill_process`, score, per-process RSS
> breakdown) could not be retrieved. The 137 exit code and `Killed`
> message from bash are definitive.

---

## Memory Budget Analysis

### Model alone (steady state)

| Component | Approximate Size |
|-----------|-----------------|
| GGUF model weights (mmap'd) | 2.0 GB |
| KV cache (2048 ctx, f16, 28 layers, 3B) | ~0.7 GB |
| llama.cpp runtime + scratch buffers | ~0.3 GB |
| **llama-cli total** | **~3.0 GB** |

### Concurrent build pressure (peak)

| Process | Approximate RSS |
|---------|----------------|
| `glslc` (per shader, ×N parallel) | 200–500 MB each |
| `ld` / `lld` linker (ggml-vulkan.cpp) | 1–3 GB |
| `cmake` orchestration | 50–100 MB |
| Claude Code agent (node/python) | 300–500 MB |

With 8 vCPUs, `cmake --build -j8` spawns up to 8 parallel compile or
shader jobs simultaneously, each potentially consuming 500 MB+.
Combined with the linker and the running `llama-cli`, it is straightforward
to exceed the 16 GB `CommitLimit` even with 32 GB of physical RAM.

---

## Why No Swap Made This Worse

On a system with swap (e.g., 8 GB of swap), the `CommitLimit` would have
been `32 GB × 50% + 8 GB = 24 GB`, and anonymous pages under memory
pressure could have been paged out rather than triggering an OOM kill.
With **zero swap**, the kernel had no safety valve:

- `CommitLimit = 16 GB`
- No pages can be swapped out
- When `Committed_AS` hits the ceiling, the OOM killer fires immediately

---

## Why g5.4xlarge Fixes This

| Resource | g5.2xlarge | g5.4xlarge | Change |
|----------|-----------|-----------|--------|
| RAM | 32 GB | **64 GB** | +32 GB |
| vCPU | 8 | **16** | +8 |
| GPU | 1 × A10G (24 GB) | 1 × A10G (24 GB) | same |
| `CommitLimit` (no swap) | ~16 GB | **~32 GB** | +16 GB |

With 64 GB RAM and `CommitLimit` ≈ 32 GB:

- `llama-cli` (3B model, ~3 GB) leaves ~29 GB headroom for builds
- 8 parallel `glslc` jobs at 500 MB each = 4 GB — well within budget
- Linker at 3 GB peak still leaves ample margin
- Vulkan GPU inference (`-ngl 99`) offloads all 28 transformer layers to
  the A10G, reducing CPU RAM usage for activations to <100 MB

---

## Secondary Issue: Phase 2 Model Format

The garbage output (`couldglyPPPiquer...`) on the first token generation
is **not an OOM symptom**. It is a model format mismatch:

- Model file `llama-3.2-3B-q4_hqq.gguf` was quantized by an earlier
  build (pre-commit `44ebed103`) using **Phase 2 format** (16-bit FP
  zero-point, 2 bytes at offset 2–3 of each block).
- Current code reads zero as **Phase 3 format** (8-bit integer zero-point
  at offset 2, padding byte at offset 3).
- Both layouts are 20 bytes per block but the zero value reads as garbage
  (the low byte of an FP16 near-zero ≈ 0, distorting every weight).
- **Fix:** Re-quantize from the F16 source:
  ```bash
  ./build/bin/llama-quantize Llama-3.2-3B-F16.gguf llama-3.2-3B-q4_hqq.gguf Q4_HQQ
  ```
  After re-quantization, CPU inference produces coherent output:
  `"The capital of France is Paris. It is the largest city in France..."`

---

## Recommendations

1. **Upgrade to g5.4xlarge** (64 GB RAM) for Vulkan GPU inference testing.
2. **Add swap** (even 8 GB) on any instance running concurrent builds +
   inference: `fallocate -l 8G /swapfile && mkswap /swapfile && swapon /swapfile`
3. **Serialize builds and inference** — don't run `cmake --build -j$(nproc)`
   at the same time as `llama-cli` on memory-constrained hosts.
4. **Re-quantize models** from F16 source when switching between Phase 2
   and Phase 3 Q4_HQQ code (models are not cross-compatible).
5. **Use `-ngl 99`** on GPU instances to offload transformer layers to
   VRAM, dramatically reducing the CPU RAM footprint during inference.
