## 2026-08-13 — CompilerGym Gate 0 environment smoke test

### Goal
Validate only that CompilerGym can reset an LLVM environment, apply a pass, build and run a program, and measure runtime.

### Frozen protocol
CompilerGym 0.2.5 with Python 3.10, NumPy 1.26.4, and `libtinfo5`; infrastructure-only benchmark `generator://csmith-v0/1`; action `-mem2reg` resolved from the environment; one warmup plus three runtime measurements. PASS requires reset and an effective action, buildable and runnable outputs, and three positive finite runtime samples. No train/dev/test split or model is used in Gate 0.

### Changes
Added the pinned Gate 0 dependencies, frozen JSON config, smoke runner, and targeted tests.

### Result
The environment exposed 124 actions and resolved `-mem2reg` to action 103. Reset and the action succeeded; the action changed the IR; the program was buildable and runnable. Runtime samples were 0.002758 s, 0.002011 s, and 0.003117 s; median runtime was 0.002758 s.

### Decision
PASS.

### Artifacts
- `outputs/gate0_compilergym_smoke_v1/config.json` — SHA256 `40f2937e7c71a7fa313dbdf6ff0c652279145833d86211721eb0b1612f2f0360`
- `outputs/gate0_compilergym_smoke_v1/experiment_report.json` — SHA256 `8320da2ef825171f9ae1d9d692a5033a5157e9c642735ca269f38ddd24ce7fe4`

### Git
Experiment code commit `31045c5150e1e001dda5a803e8f4d462e2b239a4`.

## 2026-08-13 — Gate 1 search headroom v1

### Goal
Determine whether fixed-budget Random or Greedy phase-ordering search can reproducibly outperform LLVM `-O3` on PolyBench without a learned model.

### Frozen protocol
PolyBench/C 4.2.1-beta; nine representative MEDIUM workloads; LLVM 10; a fixed 24-pass action subset; sequence length at most 16; Random and Greedy budgets of 128 evaluated candidates per kernel; CPU 24; warm-cache timing with 20 qualification warmups and two independent 10-run blocks. Measurement qualification required runtime at least 20 ms, CV at most 1%, relative MAD at most 0.5%, and block-median drift at most 1%. Search could proceed only while every kernel passed qualification. Final Gate 1 PASS additionally required geomean speedup and its bootstrap 95% CI lower bound above 1.0, at least 25% of kernels stably at least 1% faster, independent confirmation, equal budgets, and no correctness failure.

### Changes
Added the frozen Gate 1 config, fixed-toolchain search runner, checkpointed evaluation records, bootstrap intervals, independent confirmation, output-hash correctness validation, and targeted tests.

### Result
`2mm` passed measurement qualification and completed 128 Random plus 128 Greedy evaluations with no compile or run failures. The selected Greedy sequence was `-dse`. Its search-time speedup was 1.3934x, but independent confirmation retained only 1.0174x with bootstrap 95% CI `[0.9992, 1.0375]`; output SHA256 matched `-O3`. The next kernel, `3mm`, failed measurement qualification: block CV was 1.67% and 1.03%, and relative MAD was 1.22% and 0.55%. No `3mm` search evaluations or later-kernel searches were run.

### Decision
FAIL at measurement qualification. This is not a search-headroom conclusion; Gate 1 stopped before the full kernel set could be searched.

### Artifacts
- `outputs/gate1_search_headroom_v1/report.json` — SHA256 `e4e8edfe03a4941c51d12f24e0e1ee2eef00265e985fe7f60268b3e405e777d3`
- `configs/gate1_search_headroom_v1.json` — SHA256 `ab397546cec5b99511996eff11dffe8599dce541141019bd0fd1a352ecc2be9c`

### Git
Experiment code commit `3039d2af3ec636989c85f1e462aa1b74f34019a5`.
