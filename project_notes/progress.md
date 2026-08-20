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

## 2026-08-13 — Gate 1 runtime measurement qualification

### Goal
Determine whether kernel-only timing, per-process CPU binding, and larger official PolyBench datasets make the existing Gate 1 runtime qualification stable for `2mm` and `3mm`.

### Frozen protocol
LLVM `-O3` baseline only; PolyBench `POLYBENCH_TIME` with cache flushing disabled; `taskset -c 24`; 20 warmups and two independent 10-run groups; MEDIUM, then LARGE, then EXTRALARGE. Existing thresholds were unchanged: runtime at least 20 ms, CV at most 1%, relative MAD at most 0.5%, and block-median drift at most 1%. Identical-baseline sanity required a ratio within 1% of 1 and a bootstrap 95% CI covering 1.

### Changes
Made CPU binding explicit through `taskset`, added standard deviation reporting and a baseline-only qualification runner, and tested a minimum five-second warmup after MEDIUM samples exposed a frequency-state transition. No pass search ran and no global CPU settings changed.

### Result
Neither kernel qualified at any official size. At EXTRALARGE, `2mm` median runtime was 73.6613 s, with block CV 1.361% / 1.156% and relative MAD 0.923% / 0.966%; identical-baseline ratio was 1.00160 with bootstrap 95% CI `[0.98543, 1.01761]`. `3mm` median runtime was 117.2311 s, with block CV 1.422% / 1.207% and relative MAD 0.321% / 0.622%; ratio was 1.00328 with CI `[0.97924, 1.01099]`. Both output-hash correctness checks passed. A follow-up MEDIUM run with at least five seconds of warmup still failed for `2mm` and was stopped before repeating already-disproven larger settings.

### Decision
FAIL. Identical-baseline comparisons no longer show a clear fake improvement at LARGE/EXTRALARGE, but the current host retains roughly 1–2% within-group runtime variation and does not meet the frozen Gate 1 qualification thresholds. Gate 1 search remains stopped.

### Artifacts
- `outputs/gate1_runtime_measurement_qualification_v1/report.json` — SHA256 `780e22d16e1dfb46128d6fe29a0a91e03b3e074fc010ea0f67f8c6994502a2e6`
- `outputs/gate1_runtime_measurement_qualification_v2/report.json` — SHA256 `d0b85f48e60b80b60d2390a87336b8a1df3d17f0830914a89d55cf8d9eef65e9` (INVALID warmup follow-up)

### Git
Experiment code commit `c18ea905105136237f321a7c645ae3a0ba98037f`.

## 2026-08-13 — Gate 1 paired search headroom v2

### Goal
Determine whether fixed-budget Random or Greedy phase-ordering search can reproducibly outperform LLVM `-O3` on all nine frozen PolyBench kernels using local paired runtime comparisons.

### Frozen protocol
PolyBench/C 4.2.1-beta; nine MEDIUM workloads; LLVM 10; the existing 24-pass action subset; sequence length at most 16; Random and Greedy budgets of 128 candidate evaluations per kernel; CPU 24. Every comparison uses consecutive `B1 -> C -> B2` measurements with ratio `sqrt(B1 * B2) / C`; initial search uses two sandwiches, selected candidates receive three additional sandwiches, and the final winner requires a newly built binary and ten independent sandwiches. Absolute baseline CV and MAD are diagnostic only. Paired qualification retains the frozen maximum ratio CV of 1% and relative MAD of 0.5%.

### Changes
No code, measurement, search, dataset, or toolchain setting changed. Ran the frozen v2 experiment from commit `3882cb7` in a new output directory.

### Result
The first kernel, `2mm`, failed identical-baseline paired qualification before search. Its ten paired ratios ranged from 0.97406 to 1.02433, with geometric mean 0.99704, CV 1.482%, and relative MAD 1.075%. Both paired stability limits were exceeded. No candidate was evaluated, so Random/Greedy results, independent winner confirmations, and an overall headroom estimate do not exist.

### Decision
FAIL at paired measurement qualification. This is not a phase-ordering search-headroom conclusion; the frozen protocol correctly stopped before candidate search, and retrying until qualification happened to pass would invalidate the formal run.

### Artifacts
- `outputs/gate1_search_headroom_v2/report.json` — SHA256 `ec8aa7d0b6f899c694cc3821dc5969227cdd74696c7549ff85c5fe0486e6a80c`
- `configs/gate1_search_headroom_v2.json` — SHA256 `a18cdd44883f7ffabffbeba7c4280d161d02be1b8e492fddd9981d0ce08ca34e`

### Git
Experiment code commit `3882cb78ba3d17826ae077e9061daba4e9689462`.

## 2026-08-13 — Gate 1 paired search headroom v3

### Goal
Determine whether fixed-budget Random or Greedy phase-ordering search can reproducibly outperform LLVM `-O3` on all nine frozen PolyBench kernels using independent paired confirmation.

### Frozen protocol
PolyBench/C 4.2.1-beta; nine MEDIUM workloads; LLVM 10; the fixed 24-pass action subset; sequence length at most 16; Random and Greedy budgets of 128 candidate evaluations per kernel; CPU 24. Candidate ranking used local `B1 -> C -> B2` ratios with two initial sandwiches and successive refinement. Every selected winner was rebuilt and measured with ten independent sandwiches. Noise CV/MAD checks were diagnostic only. A kernel counted as confirmed only when its independent paired speedup and bootstrap 95% CI lower bound were both strictly above 1; Gate PASS additionally required geometric-mean speedup and its CI lower bound above 1, at least 25% confirmed kernels, equal budgets, and no correctness failure.

### Changes
No code, measurement, search, dataset, or toolchain setting changed during the run. Executed the frozen experiment from commit `ad57835` in a new output directory.

### Result
All nine kernels completed 128 Random plus 128 Greedy evaluations, for 2304 total candidates. There were no compile or run failures, and all nine correctness hashes matched `-O3`. Independent paired speedups had geometric mean 1.00316, median 1.00594, and overall bootstrap 95% CI `[0.99781, 1.00802]`. Only `3mm` at 1.01156 with CI `[1.00005, 1.02365]` and `nussinov` at 1.01015 with CI `[1.00312, 1.01783]` were confirmed improvements. The maximum confirmed speedup was 1.01156. Greedy supplied six final winners and Random supplied three.

### Decision
FAIL. The geometric mean was above 1, but its CI lower bound was not above 1, and only 2/9 kernels (22.2%) were confirmed improvements, below the frozen 25% requirement. This is a completed search/performance result, not a measurement-qualification failure.

### Artifacts
- `outputs/gate1_search_headroom_v3/report.json` — SHA256 `f38ca6eed90469fac5341bebf7523571938a0129e968f076a7f9ef8f5b151de0`
- `configs/gate1_search_headroom_v2.json` — SHA256 `a18cdd44883f7ffabffbeba7c4280d161d02be1b8e492fddd9981d0ce08ca34e`

### Git
Experiment code commit `ad578355b3c3b28adfb87ac5b7ec9a363b42eef0`.

## 2026-08-15 — ObjectTextSize Dataset v0

### Goal
Generate the first program-disjoint ObjectTextSize trajectory dataset for MambaPO v0 training.

### Frozen protocol
CompilerGym 0.2.5 LLVM environment; Jotaibench v0; seed 41; 1000 programs split into 900 train and 100 dev programs; 16 uniformly sampled trajectories per program; sequence length 1-32; 124 actions read from the environment; 70-dimensional InstCount state before and after every pass; deterministic ObjectTextSizeBytes primary labels and LLVM -Oz bytes; cBench remained sealed.

### Changes
Added the frozen Dataset v0 config, parallel trajectory generator, gzip JSONL schema, deterministic program-level split, and focused tests. Installed the official Jotaibench dataset through CompilerGym and generated the formal dataset in a new output directory.

### Result
Generated 14,400 train and 1,600 dev trajectories from exactly 1000 disjoint programs. The observed sequence lengths covered 1-32, feature dimension was 70, action-space size was 124, and there were zero invalid episodes. All five frozen result checks passed.

### Decision
PASS.

### Artifacts
- `outputs/object_text_size_dataset_v0_seed41/config.json` — SHA256 `ef4db4ef4518efdc17149cf6cbd04875b20e2c2df17bf4b63a41dd5a4c849d3a`
- `outputs/object_text_size_dataset_v0_seed41/program_splits.json` — SHA256 `54d617e8c55f4292aae34c016250be5d799c73d58c382d9c2217503e3d54a06e`
- `outputs/object_text_size_dataset_v0_seed41/train.jsonl.gz` — SHA256 `9dfde34466748e26b3e4f97ba72045bea9b265d2972fbfac5ba538477d836197`
- `outputs/object_text_size_dataset_v0_seed41/dev.jsonl.gz` — SHA256 `dbaa6f250d775e4443717833ac85d90cd95bd71a4f5002d93072795e9694c9c1`
- `outputs/object_text_size_dataset_v0_seed41/experiment_report.json` — SHA256 `fbfdaded97000b088d9e1ad6b3f8e2c38c5a5c990873f142823a0de089d07c4f`

### Git
Experiment code commit `53cb6a1b9dc7544526229ee9f9b7d8aac483dc4c`.

## 2026-08-15 — MLP value baseline v0

### Goal
Train the first order-agnostic value baseline on Dataset v0 before introducing sequence models.

### Frozen protocol
Program-disjoint Dataset v0 with 14,400 train and 1,600 dev trajectories; target `size_reduction_vs_oz`; final 70-dimensional InstCount state, normalized 124-dimensional pass histogram, and normalized sequence length; MLP hidden dimensions 256 and 128; Huber regression plus 0.1-weight same-program pairwise logistic loss; AdamW for exactly 30 epochs with no early stopping or hyperparameter scan; final epoch used for reporting.

### Changes
Added the frozen MLP config, order-agnostic feature loader, grouped pairwise objective, learning-curve metrics, checkpoint writer, and focused tests. A tiny integration smoke exposed and fixed an action-dimension inference bug before formal training.

### Result
Final dev MAE was 0.075943 and RMSE was 0.097225 versus constant-predictor MAE 0.133755 and RMSE 0.194657. Pearson correlation was 0.878388, Spearman correlation was 0.762282, and same-program pairwise accuracy was 0.851627. All 30 epochs completed with finite metrics on the NVIDIA GeForce RTX 3090.

### Decision
PASS.

### Artifacts
- `outputs/mlp_value_baseline_v0_seed41/config.json` — SHA256 `c72b60ec605853daca3668a288a0fd67abc1b0c14ad308157857f43da78f7b8d`
- `outputs/mlp_value_baseline_v0_seed41/model.pt` — SHA256 `b3b793183383b3ceab9a0499a7f9260e0a4e8437178039d78deac332e37a9b4b`
- `outputs/mlp_value_baseline_v0_seed41/learning_curve.json` — SHA256 `cf370ebe6b50e15d1bb8f5419fdcc532384fff4c0c724bcb5a3d1b78cef20492`
- `outputs/mlp_value_baseline_v0_seed41/experiment_report.json` — SHA256 `c422578b40badda734b488ddbe1c155d48b6d8683b6c5418929d171936f56223`

### Git
Experiment code commit `369959be4941457eeb367fffe010c4dc3754ecb6`.

## 2026-08-15 — MambaPO value model v0

### Goal
Train the frozen two-layer Mamba sequence value model on Dataset v0 after establishing the MLP baseline.

### Frozen protocol
The same program-disjoint Dataset v0 and `size_reduction_vs_oz` target as MLP; tokens are normalized current InstCount state plus previous-pass embedding plus position embedding, beginning with `state_0 + START`; d_model 128, two Mamba layers, d_state 16, d_conv 4, expand 2; Huber regression plus 0.1-weight same-program pairwise loss; AdamW for exactly 30 epochs with no early stopping or hyperparameter scan; final epoch used for reporting.

### Changes
Added the frozen MambaPO config, ordered token builder, state normalization, two-layer residual Mamba model, checkpoint/curve writer, and focused tests. Fixed a direct-script import-root bug before the formal training process created its output directory.

### Result
Final dev MAE was 0.086138, RMSE was 0.113585, Pearson correlation was 0.838114, Spearman correlation was 0.721830, and same-program pairwise accuracy was 0.777486. All 30 epochs completed with finite metrics. The frozen final Mamba metrics were weaker than the MLP baseline, and the curve showed early improvement followed by overfitting; no post-hoc epoch selection was applied.

### Decision
PASS for completion of the frozen MambaPO v0 training protocol; comparative performance remains unresolved until the remaining sequence baselines and same-budget search are complete.

### Artifacts
- `outputs/mambapo_value_v0_seed41/config.json` — SHA256 `40537146ee07f6a9afeb0de821478abeef778c495df90f60e1f3f1251e7d9b2c`
- `outputs/mambapo_value_v0_seed41/model.pt` — SHA256 `dd7a4c80d479ca61eaa4ce89c1d2d577e4c27cd55945a7b458805fffee1af450`
- `outputs/mambapo_value_v0_seed41/learning_curve.json` — SHA256 `76ee173b4396bd4d6d87a5ee9fc0c793e9015008a6307782384a1baec02515eb`
- `outputs/mambapo_value_v0_seed41/experiment_report.json` — SHA256 `3345db30c054b69f4311b9d3dded5a9352b34d96360d4a41a12f55f17ef52ed6`

### Git
Experiment code commit `14f6e96e7021fd864b31a140f9c15751248b2b2b`.

## 2026-08-15 — LSTM value baseline v0

### Goal
Train the frozen LSTM sequence baseline on the same ordered Dataset v0 representation used by MambaPO.

### Frozen protocol
The same program split, target, normalized state/pass/position tokens, Huber plus pairwise objective, optimizer, 30 epochs, no early stopping, and final-epoch reporting as MambaPO; the only model change was a two-layer d_model 128 LSTM with zero dropout.

### Changes
Added the shared frozen LSTM/Transformer sequence-baseline trainer, independent LSTM config, padding-aware sequence handling, checkpoint/curve writer, and focused tests. Ran one two-program integration smoke before the formal experiment.

### Result
Final dev MAE was 0.074661, RMSE was 0.098889, Pearson correlation was 0.878970, Spearman correlation was 0.781104, and same-program pairwise accuracy was 0.832911. All 30 epochs completed with finite metrics.

### Decision
PASS.

### Artifacts
- `outputs/lstm_value_baseline_v0_seed41/config.json` — SHA256 `257bec3906450ad2defc0feae16a86d9af7e61d8ccff42cf4cbab0df99429ea8`
- `outputs/lstm_value_baseline_v0_seed41/model.pt` — SHA256 `8bb974f924c60b1281d0af4cb14a7096a41d96f4783fa6e5df906883638ae1c9`
- `outputs/lstm_value_baseline_v0_seed41/learning_curve.json` — SHA256 `3f2f10c5d4f5d0e46dc8c34c3551c94766cb2c4a4110ff7965fd2e8325ace79a`
- `outputs/lstm_value_baseline_v0_seed41/experiment_report.json` — SHA256 `f1b0c68600bc21d321dbdc16ba0989498403079145a9827ecd2bbb0dca828b26`

### Git
Experiment code commit `ed8b68775e7eb4d71481539f9dcbe30ef49cddf9`.

## 2026-08-15 — Transformer value baseline v0

### Goal
Train the frozen Transformer sequence baseline on the same ordered Dataset v0 representation used by MambaPO and LSTM.

### Frozen protocol
The same program split, target, normalized state/pass/position tokens, Huber plus pairwise objective, optimizer, 30 epochs, no early stopping, and final-epoch reporting as MambaPO; the only model change was a two-layer d_model 128 Transformer with four heads, feedforward dimension 256, GELU, zero dropout, and explicit padding masks.

### Changes
Reused the already tested shared sequence-baseline trainer with the independent frozen Transformer config. Ran one two-program integration smoke before the formal experiment; no model or data setting changed after seeing results.

### Result
Final dev MAE was 0.077881, RMSE was 0.104930, Pearson correlation was 0.860206, Spearman correlation was 0.768361, and same-program pairwise accuracy was 0.816184. All 30 epochs completed with finite metrics.

### Decision
PASS.

### Artifacts
- `outputs/transformer_value_baseline_v0_seed41/config.json` — SHA256 `8d0fbb650893d75e126ea932e688f76e6e52362784ad00f6f04d059eeb3025f1`
- `outputs/transformer_value_baseline_v0_seed41/model.pt` — SHA256 `591d32507258777bce202cc4931c91c5b58d5f40f53dbabda5c5873b0d7cce27`
- `outputs/transformer_value_baseline_v0_seed41/learning_curve.json` — SHA256 `153f49b4e595670eb7840bd831eff0f87fc634b231606f843fdf25fbfe304cd8`
- `outputs/transformer_value_baseline_v0_seed41/experiment_report.json` — SHA256 `94f358f49d0f3680127478524c404bdc52c81f3a80a284f07cd91f0c0fd16e07`

### Git
Experiment code commit `c5b2f4aea0641ff25d60500fc94728fbeff84917`.

## 2026-08-15 — Same-budget held-out search v0

### Goal
Test whether MambaPO finds smaller object code than LLVM -Oz, Random, MLP, LSTM, and Transformer under the same true candidate-evaluation budget on held-out programs.

### Frozen protocol
The 100 program-disjoint Jotaibench dev programs; cBench sealed; LLVM action space read as 124; sequence length at most 32; Random plus four frozen learned checkpoints; learned beam width 8 and top-k action expansion 8; exactly 128 complete candidate sequences per method and program whose deterministic ObjectTextSizeBytes was read once; budget curves at 8, 16, 32, 64, and 128; primary result is code-size reduction relative to LLVM -Oz.

### Changes
Added the frozen search config, checkpoint scorers, model-guided beam candidate generation, equal-budget true size evaluation, gzip per-program results, aggregate search curves, and focused tests. Ran a one-program/two-candidate smoke before the formal experiment.

### Result
All 100 programs completed with exactly 12,800 true candidate evaluations per method and zero invalid episodes. At budget 128, geomean size reduction versus -Oz was -4.7942% for Random, -9.3731% for MLP, -4.7932% for LSTM, -7.7790% for Transformer, and -5.4337% for Mamba. Positive-program counts were 27, 29, 41, 22, and 36 respectively. Mamba did not beat -Oz, Random, or LSTM in aggregate.

### Decision
FAIL for the research hypothesis. The generated report's execution checks are PASS, but the primary comparative compiler-performance criterion is not met. cBench remains sealed and runtime evaluation is not started.

### Artifacts
- `outputs/same_budget_dev_search_v0_seed41/config.json` — SHA256 `90edff6c8d94a492ed516995b27e88bbf50791c1457e4fd3f888a2e20f65c30a`
- `outputs/same_budget_dev_search_v0_seed41/program_results.jsonl.gz` — SHA256 `0a9f4bf2c028fb062f2a934d677f57c6c831aaeb0bb08a5e3b4cf4d4c816731f`
- `outputs/same_budget_dev_search_v0_seed41/experiment_report.json` — SHA256 `c13a202cb0f4292d529494d303493ff50471eb19b7253d122fe8cdb466405594`

### Git
Experiment code commit `c74af32c1fb87e6a0a8da4e1286adc28df2d46a5`.

## 2026-08-15 — ObjectTextSize Dataset v1 extension

### Goal
Expand the training distribution after the v0 learning/search curves identified data-limited overfitting and out-of-distribution candidate ranking, while preserving the sealed held-out protocol.

### Frozen protocol
Generate exactly 1,000 new Jotaibench training programs with 16 deterministic trajectories per program, sequence lengths 1–32, seed 42, the unchanged 124-pass action space and ObjectTextSize labels; exclude every program in Dataset v0. The combined v1 training set is the unchanged 14,400 v0 training trajectories plus these 16,000 extension trajectories; the original 100-program/1,600-trajectory v0 dev split remains byte-for-byte unchanged. cBench remains sealed.

### Changes
Extended the generator with an explicit prior-split exclusion input and added its focused test. Generated only the disjoint training extension; no model, label, metric, search budget, dev program, or pass/fail criterion changed.

### Result
Generated 16,000 trajectories from 1,000 new training programs with zero invalid episodes. Independent verification counted 16,000 train records, zero dev records, and zero overlap with all 1,000 Dataset v0 train/dev programs. All generator checks passed.

### Decision
PASS.

### Artifacts
- `outputs/object_text_size_dataset_v1_extension_seed42/config.json` — SHA256 `1b73e282c66f7698826b916728e14b741c76b2906b0246249050a4b222ed3011`
- `outputs/object_text_size_dataset_v1_extension_seed42/program_splits.json` — SHA256 `1569a23b82d77b4e99420afb1b20fae9dec8a371a4e4c19e78e03669eec331a6`
- `outputs/object_text_size_dataset_v1_extension_seed42/train.jsonl.gz` — SHA256 `56d0ea4de1806fa30eede5ae5359ed6dfc790fe7039ad8c49bf3f800c37331e4`
- `outputs/object_text_size_dataset_v1_extension_seed42/dev.jsonl.gz` — SHA256 `0fc602c2f84ecf53f9a6a2eb28eebc393c3e92de6dfe54fa25727825997f60fc`
- `outputs/object_text_size_dataset_v1_extension_seed42/experiment_report.json` — SHA256 `47b4b63ef0a96f625b4552032cafc8d7a50643dacc82019fe6da6ce189acb980`

### Git
Experiment code commit `31022906334f6a451f0c581acd4da45aaaca77b5`.


## 2026-08-15 — MLP value baseline v1

### Goal
Measure whether the single justified Dataset v1 expansion improves the frozen order-agnostic MLP on the unchanged Dataset v0 dev split.

### Frozen protocol
The unchanged MLP v0 representation, architecture, target, loss, optimizer, seed, 30-epoch schedule, final-epoch reporting, metrics, and completion gate; training data is the 30,400 trajectories from 1,900 programs in Dataset v0 train plus the disjoint v1 extension; dev remains the original 1,600 trajectories from 100 Dataset v0 programs.

### Changes
Only the training data input was expanded. The loader reads the two immutable gzip inputs in order; no model or evaluation setting changed.

### Result
All 30 epochs completed with finite metrics. Final dev MAE was 0.073223, RMSE 0.096011, Pearson 0.898142, Spearman 0.800201, and same-program pairwise accuracy 0.864828. Compared with frozen MLP v0, final RMSE improved from 0.097225 and Spearman improved from 0.762282.

### Decision
PASS for completion of the frozen expanded-data training protocol; compiler-performance success remains gated on the same-budget held-out search.

### Artifacts
- `outputs/mlp_value_baseline_v1_seed41/config.json` — SHA256 `ea8ab98f034d8260c984f505377e4793bedcb8b60e46b021c13ef886d499c4d5`
- `outputs/mlp_value_baseline_v1_seed41/model.pt` — SHA256 `d810603e4326007e85dc69014a82a582d022e46a5eb2360c25ff3f9ffddcf170`
- `outputs/mlp_value_baseline_v1_seed41/learning_curve.json` — SHA256 `ccf7e34723e67ae890f3720072c34b8a107ffcadc7c42211a262a4cabdb67329`
- `outputs/mlp_value_baseline_v1_seed41/experiment_report.json` — SHA256 `d86838069d0c87d8668317249fd2fcf14ecb69db724c801025473d1d67ebcd53`

### Git
Experiment code commit `42250e123fdd3442043465fd8aaf226035720c78`.


## 2026-08-15 — MambaPO value model v1

### Goal
Measure whether the single justified Dataset v1 expansion improves the frozen two-layer Mamba value model on the unchanged Dataset v0 dev split.

### Frozen protocol
The unchanged Mamba v0 representation, architecture, target, loss, optimizer, seed, 30-epoch schedule, final-epoch reporting, metrics, and completion gate; training data is the 30,400 trajectories from 1,900 programs in Dataset v0 train plus the disjoint v1 extension; dev remains the original 1,600 trajectories from 100 Dataset v0 programs.

### Changes
Only the training data input was expanded. No architecture, loss, normalization rule, or evaluation setting changed.

### Result
All 30 epochs completed with finite metrics. Final dev MAE was 0.080973, RMSE 0.106254, Pearson 0.859649, Spearman 0.766751, and same-program pairwise accuracy 0.809132. Compared with frozen Mamba v0, final RMSE improved from 0.113585 and Spearman improved from 0.721830. Earlier epochs were stronger than the final epoch, but no post-hoc epoch selection was applied.

### Decision
PASS for completion of the frozen expanded-data training protocol; compiler-performance success remains gated on the same-budget held-out search.

### Artifacts
- `outputs/mambapo_value_v1_seed41/config.json` — SHA256 `98c12d4c4460132610e71def0a4f953e5906cefd3303d970cdb9b4be41a2aa01`
- `outputs/mambapo_value_v1_seed41/model.pt` — SHA256 `3414b08c418ed16e9c0e98fbe3c5e4bb2a8b6a0aebfee885dac101be7e6aa451`
- `outputs/mambapo_value_v1_seed41/learning_curve.json` — SHA256 `c23aa97c1611612449ed52b14c3d1614ba5af661d2b5717e7f27d3ddef3ff263`
- `outputs/mambapo_value_v1_seed41/experiment_report.json` — SHA256 `1aa94b2cb233cd57ec49fb982cd61357bcd28a27ddfd266e333adcce41c98808`

### Git
Experiment code commit `42250e123fdd3442043465fd8aaf226035720c78`.


## 2026-08-15 — LSTM value baseline v1

### Goal
Measure whether the single justified Dataset v1 expansion improves the frozen LSTM baseline on the unchanged Dataset v0 dev split.

### Frozen protocol
The unchanged LSTM v0 representation, architecture, target, loss, optimizer, seed, 30-epoch schedule, final-epoch reporting, metrics, and completion gate; training data is the 30,400 trajectories from 1,900 programs in Dataset v0 train plus the disjoint v1 extension; dev remains the original 1,600 trajectories from 100 Dataset v0 programs.

### Changes
Only the training data input was expanded. No architecture, loss, normalization rule, or evaluation setting changed.

### Result
All 30 epochs completed with finite metrics. Final dev MAE was 0.078625, RMSE 0.103243, Pearson 0.892101, Spearman 0.802256, and same-program pairwise accuracy 0.849186. Compared with frozen LSTM v0, Spearman and pairwise accuracy improved, while final RMSE worsened from 0.098889. Earlier epochs were stronger than the final epoch, but no post-hoc epoch selection was applied.

### Decision
PASS for completion of the frozen expanded-data training protocol; the mixed predictive result does not establish compiler-performance success, which remains gated on the same-budget held-out search.

### Artifacts
- `outputs/lstm_value_baseline_v1_seed41/config.json` — SHA256 `1c6defd028295a4e854bfdd1a8b510c2a5a798717790e766540654b3e745d8ee`
- `outputs/lstm_value_baseline_v1_seed41/model.pt` — SHA256 `3a5ecbf79dc1b4e267901a93c2004bc40be31565f86b10f3ddb05738466fd3e4`
- `outputs/lstm_value_baseline_v1_seed41/learning_curve.json` — SHA256 `b96b22ea1a8a23fbeacbd45b8d1aa5c5bb47bd6eae9ba24126136603e99bd564`
- `outputs/lstm_value_baseline_v1_seed41/experiment_report.json` — SHA256 `1a2eab13bfa00cf73a1c7985d61ef5c34ee9ba8070e72714723de35e22780c80`

### Git
Experiment code commit `42250e123fdd3442043465fd8aaf226035720c78`.


## 2026-08-15 — Transformer value baseline v1

### Goal
Measure whether the single justified Dataset v1 expansion improves the frozen Transformer baseline on the unchanged Dataset v0 dev split.

### Frozen protocol
The unchanged Transformer v0 representation, architecture, target, loss, optimizer, seed, 30-epoch schedule, final-epoch reporting, metrics, and completion gate; training data is the 30,400 trajectories from 1,900 programs in Dataset v0 train plus the disjoint v1 extension; dev remains the original 1,600 trajectories from 100 Dataset v0 programs.

### Changes
Only the training data input was expanded. No architecture, loss, normalization rule, or evaluation setting changed.

### Result
All 30 epochs completed with finite metrics. Final dev MAE was 0.075235, RMSE 0.098542, Pearson 0.884164, Spearman 0.794873, and same-program pairwise accuracy 0.850181. Compared with frozen Transformer v0, final RMSE improved from 0.104930, Spearman from 0.768361, and pairwise accuracy from 0.816184. Earlier epochs were stronger than the final epoch, but no post-hoc epoch selection was applied.

### Decision
PASS for completion of the frozen expanded-data training protocol; compiler-performance success remains gated on the same-budget held-out search.

### Artifacts
- `outputs/transformer_value_baseline_v1_seed41/config.json` — SHA256 `5ad30b755352ac291a3f0dac0c8a5a12e666aae9453835103f80e80c38a279a0`
- `outputs/transformer_value_baseline_v1_seed41/model.pt` — SHA256 `2f8f25f37cb23ffcdecd7733da50afc5b34a915fbfafe8a60118621de7c62bbc`
- `outputs/transformer_value_baseline_v1_seed41/learning_curve.json` — SHA256 `6e255d4b2a49d2c113b556a14ddaa98f32302bb49cbe6e91dfa3712cc4589efb`
- `outputs/transformer_value_baseline_v1_seed41/experiment_report.json` — SHA256 `694180735069609ddb144163d0f291e1b99d8fcf70e75b56d166fed893504711`

### Git
Experiment code commit `42250e123fdd3442043465fd8aaf226035720c78`.


## 2026-08-15 — Same-budget held-out search v1

### Goal
Test whether the single justified Dataset v1 expansion enables MambaPO to find smaller code than LLVM -Oz, Random, MLP, LSTM, and Transformer under the unchanged true candidate-evaluation budget on the original held-out programs.

### Frozen protocol
The identical v0 search protocol and original 100 program-disjoint Jotaibench dev programs; cBench sealed; Random plus the four frozen v1 final-epoch checkpoints; sequence length at most 32; learned beam width 8 and top-k 8; exactly 128 deterministic ObjectTextSize candidate evaluations per method and program; budget curves at 8, 16, 32, 64, and 128; primary result remains code-size reduction relative to LLVM -Oz. Only the checkpoint paths changed from search v0.

### Changes
Replaced the four v0 checkpoints with their expanded-data v1 checkpoints after a one-program/two-candidate compatibility smoke. No dev program, seed, search parameter, evaluation metric, baseline, budget, or gate changed.

### Result
All 100 programs completed with exactly 12,800 true candidate evaluations per method and zero invalid episodes. At budget 128, geomean size reduction versus -Oz was -4.7942% for Random, -7.6934% for MLP, -5.3315% for LSTM, -6.8634% for Transformer, and -5.8771% for Mamba; positive-program counts were 27, 30, 40, 33, and 36. Mamba did not beat -Oz, Random, or LSTM. Read-only candidate diagnostics found Mamba mean per-program Spearman 0.1775 and pairwise accuracy 0.5856 among its 128 evaluated candidates, but this improved local ranking did not translate into compiler-performance success. All four v1 learning curves still had an earlier best dev RMSE than the frozen final epoch.

### Decision
FAIL for the research hypothesis. Execution and reproducibility checks are PASS, but the primary compiler-performance gate remains unmet after the route-authorized data expansion. Per the frozen route, no further small model/search variant is started; cBench remains sealed and runtime evaluation is not started.

### Artifacts
- `outputs/same_budget_dev_search_v1_seed41/config.json` — SHA256 `7ef9fd4620ac0ee973688fdf8a01f44b2280e94a930e32ba2ae2f43648e94b1f`
- `outputs/same_budget_dev_search_v1_seed41/program_results.jsonl.gz` — SHA256 `708fd8cf6b0eb573e8d2ce84f34050e6945ad9f146f8c6f9ec7949e6d9867a19`
- `outputs/same_budget_dev_search_v1_seed41/experiment_report.json` — SHA256 `e3aba856cd172e9bdfcea8e7c6e4ae5be7f025802eed4900b8db65e9dee3ae5d`

### Git
Experiment code/config commit `23cfefe6fbd6f3567989bec63de44852d87c9501`.

## 2026-08-19 — Route A ObjectText K=50 labels v6

### Goal
Generate frozen official RLCompOpt Route-A K=50 ObjectText labels for the official train and validation program populations without accessing final/OOD data.

### Frozen protocol
Official 50 candidate pass sequences; independent reset from the original benchmark for every candidate; absolute `ObjectTextSizeBytes` observations with `reward_space=None`; official train (28,167 programs) and validation (4,490 programs) only; generic CompilerGym 0.2.5 compatibility adaptation with LLVM 10.0.0; 12 program-level workers, sequential candidate rollout within each program, no automatic retry. A program is valid only with all 50 candidate rollouts complete.

### Changes
Added observation-only K=50 label generation, atomic per-program shards, fixed one-thread worker limits, and an exact-config resume path. The run resumed only after its existing frozen config was byte-for-byte validated; existing completed shards were read and included in final counts.

### Result
All 28,167 training and 4,490 validation program shards were generated. Training had 28,159 complete-K50 programs and 8 incomplete programs; validation had 4,488 complete-K50 programs and 2 incomplete programs. The incomplete programs were marked invalid after LLVM 10 backend code-generation failures (`Cannot emit physreg copy instruction`); no candidate retry, penalty, or imputation was applied.

### Decision
COMPLETE under the frozen data-validity policy. Usable supervised population: 28,159 complete-K50 training programs and 4,488 complete-K50 validation programs. Excluded under the frozen policy: train 8, validation 2.

### Artifacts
- `outputs/rlcompopt_route_a_objecttext_v6_parallel12/config.json` — SHA256 `500239700f566510b13d040be0a7136b7f76e3d8526f9c6cce7714adca1338af`
- `outputs/rlcompopt_route_a_objecttext_v6_parallel12/experiment_report.json` — SHA256 `1b99218614a638785cb92537ac3c71c76cb1be066e72158af3d727038be91ded`
- `configs/rlcompopt_action_seq_50.txt` — SHA256 `5243dd2923da9b392b18b81c86532f45a8eef3619c831a9a3e58b50c8f759cba`

### Git
Label executor commits `ab67b87` and `3ab490d`.

## 2026-08-19 — Route-A fixed-set Oracle v6

### Goal
Compute the frozen K=50 validation Oracle from existing Step-3 validation labels and apply the predefined Route-A/B decision rule.

### Frozen protocol
Existing official validation membership only; every Oracle entry requires `oracle_K50_validity = valid_complete_K50`, `S_Oz > 0`, and exactly all 50 frozen candidate best post-pass ObjectText sizes. No final/OOD access, labels, retries, training, search, or runtime evaluation.

### Changes
Added the minimal read-only Step-4 Oracle computation and audit output. Corrected the prior Step-3 status wording to the frozen per-program validity policy.

### Result
Validation accounting: total 4,490; complete-K50 Oracle 4,488; ratio-valid 4,490; Route-A Oracle valid 4,488; excluded incomplete-K50 2; ratio-invalid 0. `OracleMeanOverOz` by dataset: anghabench-v1 0.049343207670108594; blas-v0 0.0348389800763768; clgen-v0 0.09301890236961312; csmith-v0 0.2520939164830973; github-v0 0.012383779566174397; linux-v0 0.00438178271242194; llvm-stress-v0 0.026718478771508267 (147/149 valid); opencv-v0 0.0596191620394285; poj104-v1 0.203875121095325; tensorflow-v0 0.03809282840272151. The macro `RouteAOracleMeanOverOz` is 0.07743661591867755. Every included Oracle used exactly K=50 candidates.

### Decision
STAY ROUTE A. The branch criterion is defined and positive. No subsequent training or Route-B work was started.

### Artifacts
- `outputs/rlcompopt_route_a_oracle_v6/config.json`
- `outputs/rlcompopt_route_a_oracle_v6/validation_oracle_programs.jsonl.gz`
- `outputs/rlcompopt_route_a_oracle_v6/experiment_report.json`

### Git
Step-4 code commit `fcc47e95`.

## 2026-08-19 — Route-A ObjectText NVP targets v6

### Goal
Recover official RLCompOpt NVP semantics and construct the frozen complete-K50 ObjectText train and validation soft-target datasets.

### Frozen protocol
Only complete-K50 Step-3 records enter: train 28,159 and validation 4,488. The OFFICIAL-CODE-ALIGNED value formula from `rlcompopt/cl/dataset.py` is `(ir_compiler - all_ir_searches) / ir_compiler`; the PROJECT-SPECIFIC OBJECTTEXT ADAPTATION replaces those costs with `S_Oz` and candidate best post-pass ObjectText size. The selected official Autophase dense-label configuration uses target temperature `T=0.05`, applies `softmax(value / T)`, and has `logit_temperature=1`. No label regeneration, model training, search, final/OOD access, or runtime evaluation occurred.

### Changes
Added a Step-6 builder that reads only existing shard records and writes one K=50 value vector plus one normalized soft target per eligible program. The implementation enforces exactly candidate IDs 0–49 and excludes all existing incomplete-K50 records.

### Result
Official-code recovery: dense labels are formed as `softmax(labels / T)` in `rlcompopt/cl/models/gnn_pyg.py`; the objective is PyTorch soft-label cross entropy on logits (`CrossEntropyLoss(..., reduction='none')`, then mean); deterministic evaluation sorts predicted logits descending in `rlcompopt/model_testing.py`, while the nonzero-temperature path samples without replacement. The Autophase anchor is a 56-dimensional `env.observation["Autophase"]` vector, normalized by raw feature index 51 in official evaluation; the official graph alternative is `Programl` processed through `FeatureExtractor` and `dgl2pyg`, recorded but not implemented. Train targets: 28,159 included and 8 excluded; validation targets: 4,488 included and 2 excluded. Focused checks on one train and one validation record and the dataset-wide checks verified 50 finite nonnegative target values summing to one and higher value -> higher target mass.

### Decision
COMPLETE. Step 6 target construction is complete; NVP training was not started.

### Artifacts
- `outputs/rlcompopt_route_a_nvp_targets_v6/config.json`
- `outputs/rlcompopt_route_a_nvp_targets_v6/train_targets.jsonl.gz`
- `outputs/rlcompopt_route_a_nvp_targets_v6/validation_targets.jsonl.gz`
- `outputs/rlcompopt_route_a_nvp_targets_v6/experiment_report.json`

### Git
Step-6 code commit `4b0274c1`.

## 2026-08-19 — Route-A ObjectText Autophase-NVP anchor v6

### Goal
Train the paper-style Autophase NVP anchor against frozen Step-6 ObjectText K=50 targets, select only by validation policy-45 dataset-macro MeanOverOz, then stop before controlled-model or final/OOD work.

### Frozen protocol
Inputs are exactly 28,159 complete-K50 train and 4,488 complete-K50 validation target records. The OFFICIAL-CODE-ALIGNED Autophase `CLSLearner` path is `Linear(56,256)`, ReLU, `Linear(256,256)`, ReLU, `Linear(256,50)` (93,234 parameters), with Autophase divided by raw feature 51. Adam (lr `5e-4`, weight decay `1e-5`), batch 256, warmup 500, cosine-to-1%-lr through 10,000 steps, target temperature `0.05`, logit temperature `1`, seed 0, and mean soft-label cross entropy are used. The PROJECT-SPECIFIC interface adaptation replaces the official on-policy target database with frozen ObjectText target files. Offline policy-45 ranks K=50 logits descending with candidate-ID tie breaks and consumes only independent-reset prefix labels; no recompilation occurs.

### Changes
Added the minimal frozen-target Autophase-NVP trainer, exact offline policy-45 evaluator, configuration, and focused unit tests. Checkpoints were evaluated every 100 steps and selected only by validation policy-45 dataset-macro MeanOverOz.

### Result
The 10,000-step run completed. Selected checkpoint: step 5,900; validation policy-45 dataset-macro `MeanOverOz = 0.06292471734915961`; validation CE `3.7053382396698`; mean program-level `policy45_regret = S_policy45 - S_oracle = 7.230614973262032` bytes. Dataset values: anghabench-v1 `0.03625893578559407`; blas-v0 `0.028421896761382594`; clgen-v0 `0.08715286242721984`; csmith-v0 `0.2424069521191263`; github-v0 `0.010079968644686688`; linux-v0 `0.0027308075980932967`; llvm-stress-v0 `-0.05328204871619967`; opencv-v0 `0.051984020502374204`; poj104-v1 `0.1923645206449619`; tensorflow-v0 `0.031129257724356863`. Relative to frozen Oracle `0.07743661591867755`, opportunity recovered is `0.06292471734915961 / 0.07743661591867755 = 0.812596426156354` (81.26%). All 4,488 validation programs were valid; no candidate execution occurred.

### Decision
COMPLETE. This is the requested Autophase-NVP anchor. No MLP, LSTM, Transformer, Mamba, candidate regeneration, Route-B, final/OOD, or runtime experiment was started.

### Artifacts
- `configs/autophase_nvp_objecttext_v6.json`
- `outputs/autophase_nvp_objecttext_v6/config.json` — SHA256 `77b65b2d6abe314b187f289267f521a7767443fa91d656e2fbff5ccfee9365da`
- `outputs/autophase_nvp_objecttext_v6/model.pt` — SHA256 `602adb770f56e7dc1ee4d39bf7336eecca4fd7d589cf9e55a5ac10e1c8ebb733`
- `outputs/autophase_nvp_objecttext_v6/learning_curve.json`
- `outputs/autophase_nvp_objecttext_v6/experiment_report.json` — SHA256 `ed1e577e8304740a9454aafdffa56429d29430f1e104794ead210a046acdf050`

### Git
Anchor implementation commit `67306aa8`; result record commit `f6a590c9`.

## 2026-08-19 — Route-A controlled architecture comparison Stage A v6

### Goal
Select one frozen Stage-A configuration each for MLP, LSTM, Transformer, and Mamba under a common explicit candidate-sequence interface, using only validation policy-45 dataset-macro MeanOverOz.

### Frozen protocol
All four methods use normalized 56-D Autophase (`raw / raw[51]`), the same frozen ordered K=50 LLVM action-ID sequences, vocabulary IDs 0–123 plus `PAD=124`, and right padding to the inspected maximum length 20 (actual lengths 4–20). Each receives program conditioning and scores every candidate individually to produce 50 logits. All use the frozen Step-6 target and mean soft-label cross entropy, Adam (`lr=5e-4`, weight decay `1e-5`), batch 256, 500-step warmup then cosine decay, 10,000 steps, seed 0, no early stopping, and evaluation every 100 steps. One configuration per architecture was allowed before seeing results. Policy validation is sampling-disabled, deterministic tie-broken descending logit ranking and offline independent-reset prefix simulation with exactly 45 scored passes; no LLVM candidate execution occurred.

### Changes
Added the shared controlled-model interface/configuration, four candidate scorers, offline policy-45 evaluator, and focused structural tests. Actual parameters: MLP 99,393; LSTM 79,809; Transformer 80,321; Mamba 78,785.

### Result
All four 10,000-step runs completed. Selected macro MeanOverOz / checkpoint / validation CE / mean-median regret bytes / positive programs are: MLP `0.061663946640718934` / step 8500 / `3.7238727546630694` / `11.533645276292335, 0.0` / 2230; LSTM `0.06262883725528644` / step 7400 / `3.7210916427367513` / `11.442513368983958, 0.0` / 2192; Transformer `0.06316299598765236` / step 7400 / `3.724460540608289` / `11.37655971479501, 0.0` / 2176; Mamba `0.06417084565779806` / step 7400 / `3.720366789907908` / `9.438725490196079, 0.0` / 2224. Per-dataset MeanOverOz (MLP / LSTM / Transformer / Mamba): anghabench-v1 `0.040928401188955556 / 0.03574650541387411 / 0.03846328113654883 / 0.038986685529899714`; blas-v0 `0.022153961024919518 / 0.018698873368181188 / 0.02661873046075893 / 0.022698144622708454`; clgen-v0 `0.0855566938139966 / 0.08669634225786711 / 0.08567577144400616 / 0.08747341114534274`; csmith-v0 `0.22462241197786434 / 0.23039139434798558 / 0.22515431593527946 / 0.24036140179995746`; github-v0 `0.008578494820972659 / 0.008196340930782904 / 0.007676138005660981 / 0.008880935853575493`; linux-v0 `-0.0016254641143507186 / -0.0010682840834723229 / -0.001214510100457079 / -0.003481568868062433`; llvm-stress-v0 `-0.03785979477250976 / -0.030997558943146233 / -0.025458079874166942 / -0.029024794417755306`; opencv-v0 `0.050874172597977635 / 0.05110349375228405 / 0.05288385284740128 / 0.05236931859965227`; poj104-v1 `0.19240048236882917 / 0.1921998336198982 / 0.1920132841045731 / 0.19245584081946757`; tensorflow-v0 `0.03101010750053432 / 0.03532143188860968 / 0.02981717591691892 / 0.03098908149319463`. Cohort is unchanged for every model: total 4,488, primary-valid 4,488, invalid 0. Frozen Autophase-NVP reference is `0.0629247173`; fixed Route-A Oracle is `0.07743661591867755`.

### Decision
COMPLETE. Mamba is the highest selected Stage-A controlled architecture under the frozen metric. This is not Stage-B replication or final/OOD evidence. No NVP retraining, Stage B, final/OOD, Route B, candidate search, or runtime experiment was started.

### Artifacts
- `configs/controlled_nvp_stage_a_v6.json`
- `outputs/controlled_nvp_stage_a_objecttext_v6/shared_interface_config.json`
- `outputs/controlled_nvp_stage_a_objecttext_v6/{mlp,lstm,transformer,mamba}/model.pt`
- `outputs/controlled_nvp_stage_a_objecttext_v6/{mlp,lstm,transformer,mamba}/learning_curve.json`
- `outputs/controlled_nvp_stage_a_objecttext_v6/comparison_report.json`

### Git
Stage-A implementation commit `12ad4f32`.

## 2026-08-19 — Route-A Stage B three-seed replication v6

### Goal
Replicate the exact frozen NVP anchor and Stage-A controlled configurations for final seeds `{1,2,3}` on the existing validation cohort.

### Frozen protocol
Existing complete-K50 ObjectText targets only: 28,159 training and 4,488 validation programs; fixed configurations, 10,000 steps, fresh initialization per seed, 100-step validation cadence, deterministic offline policy-45 selection, sampling disabled, and no Stage-A checkpoint reuse. No candidate regeneration, LLVM candidate execution, final/OOD, cBench, Route B, runtime, label, or search work.

### Result
Execution is `COMPLETE`; seed set is `[1,2,3]`; `stage_a_configurations_unchanged=true`; `stage_a_checkpoint_reused=false`; every selected cohort is `4488/4488/0` total/primary-valid/invalid. Selected step:macro MeanOverOz (seed 1 / 2 / 3): NVP `5800:0.06284381550402421 / 7100:0.062365030568491374 / 4800:0.06310418253161729`; MLP `8500:0.062238837856702735 / 6800:0.06152506722228346 / 3900:0.06296220394000068`; LSTM `6900:0.05742434922335341 / 7500:0.05917726290521683 / 6400:0.06530461999389482`; Transformer `6300:0.06394440755644565 / 8200:0.06447248758569549 / 8400:0.060495106504070736`; Mamba `6300:0.063095014307107 / 5400:0.0628207157907167 / 6200:0.0647430152083397`.

Three-seed macro means: NVP `0.06277100953471096`; MLP `0.06224203633966229`; LSTM `0.06063541070748835`; Transformer `0.06297066721540395`; Mamba `0.06355291510205446`. Mean policy-45 regret bytes: NVP `7.785130718954249`; MLP `10.90121806298277`; LSTM `10.662878787878787`; Transformer `10.384878193701724`; Mamba `9.502822341057636`; every seed median is `0.0`. Opportunity recovered: NVP `0.81061147610882`; MLP `0.8037804286931609`; LSTM `0.7830328067430852`; Transformer `0.8131898129630889`; Mamba `0.8207088384233696`. Mamba remains highest controlled and exceeds NVP by `0.0007819055673435049`; exact per-dataset values are in the comparison report.

### Decision
COMPLETE. Mamba replicates as the highest controlled validation-only model; this is not final/OOD or generalization evidence. Stop here pending review.

### Artifacts
- `configs/route_a_stage_b_v6.json`
- `outputs/route_a_stage_b_v6/{nvp,mlp,lstm,transformer,mamba}/seed{1,2,3}/{config.json,model.pt,learning_curve.json,experiment_report.json}`
- `outputs/route_a_stage_b_v6/comparison_report.json`

### Git
Stage-B implementation commits `f786c27b` and `f8ac8738`; result-record commit pending.
## 2026-08-20 — Route-A final/OOD ObjectText evaluation v6

### Goal
Evaluate the frozen Stage-B checkpoints once on the recovered official RLCompOpt final/OOD population, with no post-unseal selection.

### Frozen protocol
The official `benchmarkdataset_all-test.json` manifest contains 4,683 disjoint programs across 14 datasets, including cBench, CHStone, MiBench, and NPB. ObjectText used the existing ordered K=50 sequences, independent resets, no automatic retries, and policy-45 deterministic ranking. The only learned checkpoints were Stage-B NVP/MLP/LSTM/Transformer/Mamba seeds `{1,2,3}`. Primary aggregates use the predeclared H1/H2a/H2b family-specific common cohorts; runtime was not accessed.

### Result
K=50 labeling completed for all 4,683 programs: 4,679 complete-K50 and 4 incomplete-K50; feature failures were zero. All 15 fixed model/seed evaluations completed and every required family/dataset common cohort was nonempty. Offline K=50 Oracle dataset macro MeanOverOz is `0.10518465654492251`. Final three-seed macro MeanOverOz: H1 Mamba `0.08462666303481921` versus native `-Oz` `0.0`; H2a Mamba `0.08462666303481921` versus NVP `0.08715469206982522`; H2b MLP `0.08199445421951396`, LSTM `0.08445278226557085`, Transformer `0.08439936883039871`, Mamba `0.08462666303481921`. Thus Mamba is the highest controlled final model, but NVP is higher in the system-level H2a comparison. Validation's Mamba-over-NVP ordering did not replicate on final/OOD.

### Decision
COMPLETE. The one-way final evaluation is closed; no model, checkpoint, candidate, or protocol change followed final inspection.

### Artifacts
- `outputs/route_a_final_objecttext_v6/config.json` — SHA256 `a0e058d0a530d164bad4a7b99e77d876aa212776d2377e6756619c61d48c6bb1`
- `outputs/route_a_final_objecttext_v6/final_program_manifest.json` — SHA256 `0beab51064d189d5aee5b89ed2802b39a6ddae9f6c1c0f04731632cdfcbe96fd`
- `outputs/route_a_final_objecttext_v6/shards/final/`
- `outputs/route_a_final_objecttext_v6/model_results/`
- `outputs/route_a_final_objecttext_v6/comparison_report.json` — SHA256 `9904b9f2a3d3d63e6a7e875fe18cdb950e25029070b159f88ae8e700f0f03339`

### Git
Final evaluator commits `8fa3ddff`, `93d10df3`, and `236370bb`; result-record commit pending.

