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
