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
