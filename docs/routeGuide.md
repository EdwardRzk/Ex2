# MambaPO Experiment Guide v6

## RLCompOpt-Anchored ObjectTextSize Optimization with Mamba

Version status:

```text
v6
Frozen execution protocol
```

This final v6 clarification keeps the research route unchanged and freezes the
remaining reporting and data-validity semantics before implementation:
validation configuration selection, three-seed final reporting, measurement
failure handling, K-way target completeness, incomplete-rollout handling,
common-cohort aggregation, and the claim boundary for execution-only runtime
results.

No new research branch, search method, Gate, or feasibility experiment is
introduced by this revision.

This document replaces the earlier v3/v4/v5 execution guides as the main project route.

The purpose of v6 is not to introduce another research direction. It freezes the remaining implementation semantics discovered during the v5 audit before large-scale ObjectText label generation begins.

The project must not return to the legacy raw 32-pass search route, independent headroom experiments, or open-ended validation tuning.

---

# Part A. Scientific Protocol

## 0. Frozen Project Objective

Project name:

```text
MambaPO
Mamba-guided LLVM Phase Ordering
```

Core research objective:

> Use Mamba to model the compatibility between a program and ordered LLVM optimization-pass candidates, then select better phase-ordering candidates than existing value-prediction and sequence-model baselines under the same literature-supported candidate space and evaluation budget.

Primary optimization target:

```text
Object code size
```

Primary metric:

```text
ObjectTextSizeBytes
```

Primary LLVM baseline:

```text
LLVM -Oz
```

Secondary metric:

```text
CPU runtime
```

The final project must retain all of the following:

```text
Mamba
+
LLVM Phase Ordering
+
compiler optimization improvement
+
unseen-program generalization
```

The formal research question is:

```text
For the learned methods:

ObjectText-NVP
MLP
LSTM
Transformer
Mamba

use the same frozen RLCompOpt-derived candidate space,
the same program split,
and the same 45-scored-pass learned-method evaluation protocol.

Under those shared learned-method constraints,
can Mamba use explicit ordered candidate-pass information
to select optimization candidates that produce smaller
ObjectTextSize on unseen programs?

LLVM -Oz is the native compiler external baseline.
It is not constrained to the learned methods' 45-pass budget.
```

Runtime is a secondary reality check after the code-size experiment is frozen.

The project does not require every training candidate to execute a runtime benchmark.

The project does not use a separate headroom experiment to decide whether training may begin.

---

## 1. Primary Literature Anchor

The primary scientific anchor is:

```text
Learning Compiler Pass Orders using Coreset
and Normalized Value Prediction
ICML 2023
RLCompOpt
```

Paper:

```text
https://proceedings.mlr.press/v202/liang23f/liang23f.pdf
```

Official repository:

```text
https://github.com/facebookresearch/RLCompOpt
```

Official evaluation implementation:

```text
https://github.com/facebookresearch/RLCompOpt/blob/main/rlcompopt/model_testing.py
```

The project inherits the following concepts from RLCompOpt wherever applicable:

```text
LLVM phase-ordering formulation

124-action LLVM pass space

45-pass evaluation horizon / scored budget

paper-derived K=50 generalized-action coreset

coreset discovery method

Normalized Value Prediction (NVP)

official dataset split philosophy

independent evaluation of ranked candidate sequences

official evaluation orchestration
```

The intended scientific adaptations are limited to:

```text
1. original IR-count objective
   ->
   ObjectTextSize objective

2. paper-style value prediction
   ->
   explicit candidate pass-sequence representation

3. baseline architecture
   ->
   Mamba as the main sequence architecture
```

The project must not silently redefine:

```text
candidate orchestration

45-pass scoring semantics

dataset roles

candidate-space construction

train/validation/final-test separation
```

For concrete executable semantics, the official RLCompOpt evaluation implementation takes precedence over an informal interpretation of paper prose unless this Guide explicitly declares a project adaptation.

---

## 2. What Is Paper-Aligned and What Is Project-Specific

Every important setting belongs to one of three categories.

### 2.1 Paper-aligned definitions

Examples:

```text
RLCompOpt action space

K=50 coreset concept

approximately 17,500 coreset-discovery programs

approximately 200 random discovery episodes per program

45 passes per discovery episode

independent ranked-candidate evaluation

NVP normalized-value framework

paper dataset roles
```

### 2.2 Official-implementation semantics

Examples:

```text
candidate reset behavior

candidate evaluation orchestration

scored trajectory handling

official action mapping

official reward/value implementation details
```

When paper prose and implementation details differ, record the difference.

For executable evaluation semantics, prefer the official implementation unless this Guide explicitly freezes a project-specific adaptation.

### 2.3 Project-specific adaptations

Examples:

```text
ObjectTextSize as the primary objective

ObjectText-specific candidate labels

explicit candidate pass-token encoding

Mamba architecture

finite fair model-selection budget

synthesized runtime protocol
```

Every project-specific choice must be labeled when it could otherwise be mistaken for a paper-defined parameter.

---

## 3. Environment and Compiler Freeze

Before formal ObjectText label generation, freeze and record:

```text
LLVM version

clang/compiler version

CompilerGym version

CompilerGym fork/branch

target triple

target CPU if explicitly configured

host architecture

compiler target configuration

ObjectText observation API used
```

Paper-aligned RLCompOpt environment information includes:

```text
LLVM / clang 10.0.0

Python 3.8

RLCompOpt-specific CompilerGym fork/branch
```

If the current project uses a different compatibility environment, for example:

```text
Python 3.10

generic CompilerGym 0.2.5
```

record this as:

```text
PROJECT-SPECIFIC COMPATIBILITY ADAPTATION
```

Do not describe a compatibility environment as an exact paper reproduction.

ObjectTextSize is deterministic for a frozen compiler/target setup but platform-dependent.

Therefore compiler and target metadata are part of the scientific experiment definition.

---

## 4. One-Time Action Mapping Compatibility

The official K=50 sequences depend on LLVM action identities.

Before using the official candidate set, perform one compatibility check:

```text
paper/repository action index
->
paper/repository action name
->
current environment action name
```

The purpose is only to prevent an action-index mismatch.

This is:

```text
required compatibility work
```

It is not:

```text
a Headroom Gate

a search-quality experiment

a feasibility experiment

a reason to benchmark many programs
```

If the exact official environment and action mapping are recovered, do not repeat the mapping check.

---

# 5. ObjectText API Semantics

This section is mandatory.

Formal Step 3 label generation must not start until this distinction is implemented correctly.

CompilerGym can expose an ObjectText metric through both:

```text
observation space
```

and:

```text
reward space
```

These have different semantics.

The same human-readable metric name does not imply the same API meaning.

---

## 5.1 Absolute ObjectText size comes only from observation

The canonical absolute current size is:

```python
absolute_current_size = env.observation["ObjectTextSizeBytes"]
```

The canonical baseline observations are:

```python
S_O0 = env.observation["ObjectTextSizeO0"]
S_Oz = env.observation["ObjectTextSizeOz"]
```

CompilerGym may expose these cost observations as shape `(1,)` values rather
than guaranteed Python scalars. Before writing formal JSON/CSV/Parquet labels,
convert every absolute size observation to one scalar integer value.

Canonical serialized form:

```text
1234
```

Do not persist values such as:

```text
array([1234])
```

into the formal label schema.

All training fields that mean:

```text
"this compiler state contains X object-text bytes"
```

must be derived from observation space.

---

## 5.2 Reward is not absolute size

CompilerGym reward semantics are transition-based.

Conceptually:

```text
reward_t =
previous_cost - current_cost
```

or an equivalent normalized transition reward depending on the chosen reward space.

Therefore a reward space named similarly to:

```text
ObjectTextSizeBytes
```

must never be stored in a field whose meaning is:

```text
absolute object-text bytes
```

Forbidden:

```python
prefix_object_text_size_bytes = env.reward
```

or any equivalent implementation that stores a step reward as an absolute size.

---

## 5.3 Formal label generation is observation-only

For formal ObjectText label generation:

```text
absolute sizes:
read observation

candidate labels:
derive from absolute observations

reporting metrics:
derive from absolute observations
```

Preferred formal configuration:

```text
reward_space = None
```

when the API permits this cleanly.

If a reward configuration is required for unrelated environment mechanics, the reward must still not be used to construct absolute ObjectText labels.

---

## 5.4 Canonical field names

Do not use an ambiguous schema field such as:

```text
ObjectTextSizeBytes
```

without indicating its role.

Use:

```text
initial_object_text_size_bytes

oz_object_text_size_bytes

prefix_object_text_size_bytes

best_object_text_size_bytes

final_object_text_size_bytes
```

If a transition reward is stored for compatibility or debugging, use a distinct field such as:

```text
objecttext_step_reward
```

and never interpret it as an absolute size.

---

## 5.5 Only one focused API check is required

Observation/reward confusion is a concrete failure mode, so one focused check is allowed before large-scale Step 3:

```text
1 program
1 candidate
a few passes
```

Confirm:

```text
absolute label fields are read from env.observation

and

no same-named reward is written as absolute size
```

Once this passes, stop validating this issue.

Do not create:

```text
ObjectText API Qualification Gate

large benchmark validation

hash report

repeated deterministic measurement
```

---

## Measurement Validity and Failure Semantics

This section freezes the final data-validity semantics required before formal
large-scale Step 3.

The purpose is to prevent the Agent from:

```text
inventing a penalty after failures appear

silently shrinking K=50 to K<50

using different program denominators for different methods

dropping unfavorable failures

confusing a valid measurement with a valid ratio metric
```

Step 0 must first recover the official RLCompOpt / CompilerGym behavior for:

```text
compiler/environment exception

premature environment failure

missing observation

invalid observation

failed or incomplete candidate rollout

training-sample construction after candidate failure
```

If an official rule is directly applicable to the ObjectText adaptation, use
it and record the recovered rule.

If the official path does not define an applicable ObjectText rule, use the
frozen project-specific rules below.

These rules are:

```text
PROJECT-SPECIFIC DATA-VALIDITY POLICY
```

They are data-definition semantics, not a Gate and not a new experiment.

### Validity layers

Do not collapse all validity into one Boolean.

Formal records distinguish:

```text
measurement_validity

ratio_metric_validity

training_target_validity

oracle_K50_validity

failure_reason
```

#### measurement_validity

This answers only:

```text
Did the compiler/environment successfully produce
the required finite ObjectText observation?
```

A measurement can be valid even when a later ratio is undefined.

For example:

```text
ObjectTextSizeBytes = 0
```

may be a successfully returned measurement.

Do not automatically label it as a measurement failure merely because a ratio
using it as denominator would be undefined.

#### ratio_metric_validity

This answers:

```text
Can the required ratio-based metric be computed
mathematically under the frozen definition?
```

For `reduction_vs_Oz`:

```text
S_Oz <= 0
->
ratio_metric_validity =
invalid_for_ObjectText_ratio_metric
```

Do not add an epsilon.

For Route-B positive-ratio construction, the required quantities must satisfy
the frozen positive-domain requirements before:

```text
S_O0 / S_candidate
```

is computed.

A successfully measured zero may therefore have:

```text
measurement_validity = valid

ratio_metric_validity = invalid
```

#### training_target_validity

This answers:

```text
Can this program contribute the complete supervised target
required by the frozen K=50 learning problem?
```

It is independent of whether the measured optimization result is favorable.

### Candidate-rollout completeness

A candidate rollout is complete only when the complete frozen candidate pass
sequence has been executed under the expected environment semantics and all
required post-pass observations needed for the candidate label are valid.

If a candidate has length L and execution fails after pass q < L:

```text
the rollout is incomplete
```

Valid prefixes observed before the failure may be retained for diagnostics,
but by default they do not create a valid completed-candidate training label.

Frozen rule:

```text
incomplete candidate rollout
->
candidate training_target_validity = invalid
```

Do not silently use:

```text
min(valid prefixes before failure)
```

as though the full candidate had completed.

If an applicable official RLCompOpt data-generation rule recovered in Step 0
requires a different treatment, record that exact official rule before formal
Step 3 and use it consistently.

### K-way target completeness rule

The frozen learned candidate space is:

```text
K = 50
```

NVP supervision requires one complete K-dimensional target vector for each
eligible program.

The failure-handling requirement:

```text
50 / 50 valid candidate labels
```

is a:

```text
PROJECT-SPECIFIC DATA-VALIDITY POLICY
```

motivated by the fixed K-way NVP target structure.

Do not describe this 50/50 compiler-failure rule as an explicit RLCompOpt paper
statement unless an applicable official data-generation rule is actually
recovered.

The controlled MLP/LSTM/Transformer/Mamba comparison must use the same eligible
program population.

Therefore the formal supervised-sample rule is:

```text
A program is eligible for K=50 supervised training/validation
only if all 50 frozen candidate labels required for that program
have training_target_validity = valid.
```

Equivalently:

```text
K-way target completeness = 50 / 50
```

Do not silently convert:

```text
K=50
->
K=49
```

for one program.

Do not invent:

```text
failed candidate reward = 0

failed candidate size = huge penalty

missing candidate imputation
```

after observing failures.

If one or more required candidate labels are invalid under the frozen policy:

```text
program_training_target_validity =
invalid_incomplete_K50_target
```

The program is excluded from the supervised train/validation target population
for every compared learned model that uses this K=50 supervision.

This exclusion is driven only by target completeness, never by improvement
sign or model performance.

### Oracle K=50 validity

The Route-A Offline K=50 Oracle is valid only when the entire frozen K=50
candidate set has valid completed-candidate labels for that program.

Define:

```text
oracle_K50_validity =
valid_complete_K50
```

only when all 50 required frozen candidate labels are valid completed rollouts.

If any required candidate label is invalid:

```text
oracle_K50_validity =
invalid_incomplete_K50
```

Do not compute the minimum over a valid subset smaller than K=50 and report it
as the Offline K=50 Oracle.

For Route A, `oracle_K50_validity = valid_complete_K50` and supervised
`program_training_target_validity = valid_complete_K50` currently require the
same 50/50 candidate-label completeness, but keep them as distinct semantic
fields.

### Program-level baseline validity

`S_O0` and `S_Oz` remain program-level cached measurements.

Record separately:

```text
S_O0_measurement_validity

S_Oz_measurement_validity
```

A program cannot form the required ObjectText ratio-based primary metric when:

```text
S_Oz <= 0
```

and must be marked:

```text
ratio_metric_validity =
invalid_for_ObjectText_ratio_metric
```

Do not silently divide by zero.

### Aggregate eligibility and common-cohort rule

Primary validation/final method comparisons must not use a different silent
program denominator for each method.

For each predefined dataset d:

```text
P_total_d =
the frozen dataset program population
```

Common-cohort aggregation is a:

```text
PROJECT-SPECIFIC ROBUSTNESS / REPORTING TREATMENT
```

used to keep cross-method denominators comparable when evaluation validity
differs between methods/seeds.

A common-cohort MeanOverOz is conditional on joint successful evaluation of
the predefined comparison family. When `N_primary_valid < N_total`, report the
result as performance on the predefined common valid cohort and report failures
separately. Do not describe that conditional result as unconditional
performance over the entire frozen dataset population.

### Frozen comparison families

The primary comparison families are frozen before final-test unseal.

#### H1 cohort

H1 compares MambaPO vs native LLVM `-Oz`.

For dataset d:

```text
P_H1_common_d =
programs with a valid S_Oz ratio denominator
and a valid frozen policy-45 result for every final
reported Mamba seed in final_seed_set = {s1, s2, s3}
```

Native `-Oz` is deterministic and contributes no learned seed instance. All
three Mamba H1 seed results use the same `P_H1_common_d`.

#### H2a cohort

H2a compares MambaPO vs paper-style ObjectText-NVP.

For dataset d:

```text
P_H2a_common_d =
intersection of programs with a valid S_Oz ratio denominator,
valid final Mamba results for every reported Mamba seed,
and valid final NVP results for every reported NVP seed instance
```

If the frozen NVP implementation is deterministic, it contributes one final
result instance. If it is stochastic and participates in final seed
replication, it uses the shared `final_seed_set = {s1, s2, s3}`. All H2a
MeanOverOz values use the same `P_H2a_common_d`.

#### H2b cohort

H2b compares MLP, LSTM, Transformer, and Mamba.

For dataset d:

```text
P_H2b_common_d =
intersection of programs with a valid S_Oz ratio denominator
and valid final results for every reported seed of
MLP, LSTM, Transformer, and Mamba
```

All stochastic controlled methods use `final_seed_set = {s1, s2, s3}`. All
H2b MeanOverOz values use the same `P_H2b_common_d`.

### Additional combined tables

If the final report includes another table with a method set different from
H1/H2a/H2b, that method set must be declared before its aggregate is computed.
Its common cohort is the intersection for exactly that predeclared method set.
Such a table is secondary and must not replace the frozen H1/H2a/H2b primary
comparison families. Do not change the comparison-family method set after
observing final results.

### Empty common cohort and undefined dataset metric

For a predefined primary comparison family F and dataset d:

```text
if |P_F_common_d| == 0:
    MeanOverOz_F,d = undefined
```

Do not drop dataset d from the dataset-macro denominator, set its score to 0,
or impute a synthetic score.

If any dataset required by the predefined primary family has an undefined
dataset-level MeanOverOz because its common cohort is empty, that family's
dataset-macro primary aggregate is unavailable / undefined. Keep the dataset
in the report and show its failure accounting.

### Route-A branch empty-cohort semantics

The Route-A validation oracle cohort requires:

```text
valid S_Oz ratio denominator
and
oracle_K50_validity = valid_complete_K50
```

For validation dataset d, define `P_RouteA_oracle_d` from exactly those
programs.

If `|P_RouteA_oracle_d| == 0` for any required validation dataset, then:

```text
OracleMeanOverOz_d = undefined
RouteAOracleMeanOverOz = undefined
branch_criterion_status = undefined_due_to_invalid_required_data
```

Do not interpret undefined / NaN / empty data as
`RouteAOracleMeanOverOz <= 0`, and do not authorize Route B from missing
validity data. Repair the concrete implementation/data-validity cause before
evaluating the branch again under the same frozen protocol.

### Failure-count reporting

For every dataset x method x final seed where applicable, report:

```text
N_total

N_primary_valid

N_failed_or_invalid
```

and preserve failure categories.

For supervised K=50 data generation also report:

```text
N_programs_total

N_programs_complete_K50

N_programs_incomplete_K50

candidate_failure_count_by_reason
```

These are validity/accounting fields, not a new optimization metric.

A method-specific failure remains visible even when the primary comparable
aggregate uses `P_common_valid_d`.

### Validation configuration selection and failures

The scientific configuration-selection metric remains:

```text
Validation policy-45 dataset-macro MeanOverOz
```

Configuration comparison must use a common validation cohort under the frozen
validity rules.

Do not choose a configuration because excluding its failed programs produces a
better MeanOverOz.

If evaluation failure prevents a configuration from producing the required
primary metric on the frozen comparable cohort, record the failure explicitly
and apply the same frozen validity policy to every architecture/configuration.

Do not invent a configuration-specific denominator after seeing results.

### Final-test common cohort

The same principle applies after final-test unseal.

The primary H1/H2 comparison must use a common valid program cohort per dataset
for the frozen compared methods/seeds.

Also report the full method/seed failure counts so the common-cohort aggregate
does not hide robustness differences.

No program may be removed because:

```text
its code-size result is unfavorable

its runtime result is unfavorable

one model loses on it
```

Only the frozen validity policy may determine exclusion from a ratio-based
primary aggregate.

### Route-B reward-matrix completeness

Route-B coreset construction requires the complete frozen pre-greedy reward
matrix `R[N, M]` under the applicable official/frozen validity semantics.

If any required cell `R[i,j]` cannot be validly constructed after applying the
frozen failure/retry policy, then:

```text
Route-B coreset selection = undefined
```

Do not drop program rows, drop candidate columns, impute cells, set reward to
zero, or replace candidates.

Frozen rule:

```text
complete valid R[N,M]
-> row normalization
-> greedy K=50

incomplete R[N,M]
-> stop Route-B selection
```

Any future salvage rule requires an explicit protocol amendment before Route B
is rerun. This is a project-specific conservative failure rule that preserves
the frozen RLCompOpt matrix construction rather than silently modifying it.

### Failure retry policy

Formal automatic retry policy:

```text
automatic_retry_count = 0
```

Do not automatically retry a failed candidate/program evaluation. This keeps
formal failure handling deterministic and close to the official evaluation
path, which catches failed rollout evaluation rather than silently retrying it.

A successfully produced deterministic ObjectText measurement is never repeated
for reliability estimation.

If a formal job is interrupted by a clearly identified infrastructure failure
such as a CompilerGym service outage, RPC transport failure, filesystem/storage
outage, or worker infrastructure crash, fix the infrastructure problem first.
Then rerun the affected formal work under the same frozen protocol and record
the infrastructure failure separately.

Do not selectively rerun samples because their optimization result is
unfavorable. Do not relabel a compiler/candidate semantic failure as an
infrastructure failure to recover completeness. Any change from
`automatic_retry_count = 0` requires an explicit protocol amendment before
affected formal data are accepted.

### No outcome-driven data repair

Validity decisions must never depend on:

```text
whether the candidate improves code size

whether Mamba performs well

whether excluding the sample improves the aggregate

whether a baseline looks stronger after exclusion
```

Do not silently invent:

```text
failed_candidate_size = 999999999

failed_candidate_reward = 0

epsilon added to a zero denominator

candidate imputation

method-specific sample deletion
```

unless an exact applicable official rule recovered in Step 0 requires that
behavior and the rule is recorded before formal data generation/evaluation.

Any deviation from the frozen policy must stop the affected formal run until
the protocol is explicitly amended; it must not be patched ad hoc in the
middle of a dataset.

---

# 6. Primary Candidate Space: Route A

The default main route is:

```text
Route A
```

Route A uses:

```text
official RLCompOpt K=50 candidate coreset
```

The original RLCompOpt coreset is derived under the original RLCompOpt reward/metric.

The correct project description is:

```text
IR-count-derived RLCompOpt candidate coreset
transferred to an ObjectTextSize optimization task
```

Do not describe it as:

```text
ObjectText-derived coreset
```

unless Route B actually rebuilds the coreset using the ObjectText objective.

---

## 6.1 Metric-transfer limitation

The paper reports that K=50 covers approximately 95% of the improvement available under the original RLCompOpt candidate/reward construction.

That statement belongs to:

```text
the original RLCompOpt reward matrix
and original IR-count objective
```

It does not imply:

```text
95% ObjectTextSize improvement coverage
```

The ObjectText quality of the transferred K=50 set is determined only from ObjectText labels generated by this project.

No additional search experiment is required to determine this.

The fixed-set oracle is calculated directly from the same label matrix required for training and validation.

---

# 7. Official Candidate Evaluation Orchestration

This section defines the main inference semantics.

The RLCompOpt evaluation implementation evaluates ranked candidate sequences independently.

For one benchmark:

```text
reset original benchmark

obtain initial program representation

score/rank K=50 candidates once

for candidate in ranked candidates:

    reset the same original benchmark

    independently execute that candidate

    record candidate rollout measurements

    append the candidate observations into
    the scored evaluation trajectory

stop according to the 45-pass evaluation semantics
```

Every ranked candidate starts from the same original benchmark.

Candidates do not inherit modified state from earlier ranked candidates.

---

## 7.1 Forbidden sequential candidate chaining

The following behavior is forbidden in the v6 main route:

```text
candidate 1
->
take candidate 1 best state
->
candidate 2 starts from that modified state
->
candidate 3 starts from another modified state
```

Training labels have the semantic form:

```text
(initial program, candidate_j)
->
candidate_j value
```

Sequential cross-candidate chaining would change inference into:

```text
(modified state after earlier candidate, candidate_j)
->
unknown value
```

while model scores were still produced from the initial program.

That creates a train/inference distribution mismatch.

Sequential candidate composition is outside v6.

---

# 8. Candidate-Local Prefix Semantics

Within one independently executed candidate, prefix measurements matter.

For candidate:

```text
[p1, p2, p3, ..., pn]
```

execute from the initial benchmark:

```text
p1 -> state_1
p2 -> state_2
p3 -> state_3
...
pn -> state_n
```

Read the absolute ObjectText observation after every actually executed pass:

```text
size_1
size_2
size_3
...
size_n
```

The candidate-local best ObjectText label is:

```text
best_candidate_size =
min(size_1, size_2, ..., size_n)
```

---

## 8.1 Candidate labels exclude the initial state

The candidate training label must exclude the state before pass 1.

Do not define:

```text
best_candidate_size =
min(initial_size, size_1, ..., size_n)
```

Instead define:

```text
best_candidate_size =
min(size_1, ..., size_n)
```

Reason:

RLCompOpt candidate rewards are able to represent candidates that are worse than O0.

If the initial O0 state were always included in candidate-local best selection, every bad candidate could fall back to the initial state and many worse-than-O0 candidates would collapse into ties.

This would distort the candidate value distribution and NVP supervision.

---

## 8.2 Final policy metric is separate from the candidate label

Candidate-label semantics and final 45-pass policy scoring are distinct concepts.

Candidate training label:

```text
best post-pass candidate state
initial state excluded
```

Final policy metric:

```text
follow the official RLCompOpt evaluation implementation
for whether/how the initial observation participates
in final trajectory scoring
```

Do not invent an additional final-policy fallback rule in this Guide.

---

# 9. 45-Pass Scored Budget

The scientific evaluation budget is:

```text
45 scored LLVM pass applications
```

The model:

```text
scores K=50 once
sorts candidates
evaluates candidates in that order
```

Each candidate is independently reset and rolled out.

---

## 9.1 Official implementation and physical execution

The official evaluation implementation can physically execute a complete final candidate before checking that cumulative candidate length has reached or exceeded the maximum step budget.

The scientific metric still uses the allowed scored prefix.

Therefore distinguish:

```text
scientific scored budget:
45 pass applications
```

from:

```text
physical passes executed by the implementation:
may slightly exceed 45 for the final candidate
```

Preferred v6 implementation:

```text
follow official RLCompOpt evaluation behavior
```

This minimizes semantic drift.

---

## 9.2 Optional physical truncation

If engineering constraints require executing only the remaining number of passes in the final candidate, label that behavior:

```text
PROJECT-SPECIFIC METRIC-PRESERVING IMPLEMENTATION ADAPTATION
```

It must preserve the same first-45 scored observations.

Do not silently call physical truncation the exact repository behavior.

---

## 9.3 Forbidden budget substitutions

Do not replace the 45-pass scored budget with:

```text
budget 128

beam width

MCTS evaluations

custom candidate-search budget

raw pass-by-pass exploration
```

Those belong to the legacy route and are not part of v6.

---

# 10. Deterministic Candidate Ranking

Formal v6 inference freezes:

```text
sampling = False
```

Protocol:

```text
score K=50 candidates once

sort candidate values/probabilities
in descending order

evaluate candidates in that order
```

Do not sample candidates from a categorical distribution.

The paper reports that sampling-based selection was tried and performed worse; the formal NVP/BC route uses maximum-value/probability ordering.

Do not treat a Python function default argument such as:

```text
sampling=True
```

as the final paper protocol if the published experimental method uses deterministic ranking.

No further sampling audit is required unless the official experiment configuration directly contradicts the recovered paper setting.

---

# 11. Data Populations Must Remain Distinct

The following populations have different roles.

## 11.1 Coreset discovery pool

Approximately:

```text
17,500 programs
```

Purpose:

```text
large random candidate discovery
and coreset construction
```

This is not the full model-training population.

---

## 11.2 Official model-training population

RLCompOpt Table 2 contains a substantially larger model-training population.

The paper reports approximately:

```text
Train:
728,219 programs

Validation:
4,495 programs

Test:
4,683 programs
```

The exact recovered official split files/configuration take precedence if there is any discrepancy.

---

## 11.3 Curated OOD test population

Curated suites such as:

```text
cBench
CHStone
MiBench
NPB
```

are held out as OOD evaluation data in the RLCompOpt protocol.

They are not used for model tuning.

---

# 12. Official-Data-First Rule

Before generating any large resource, inspect:

```text
official RLCompOpt downloadable data

official repository configs

official coreset files

official split files

official program IDs

official preprocessed representations

official reward/value resources

official action mapping
```

Reuse every compatible artifact that actually exists.

Generate only the ObjectText-specific information that is missing.

Do not assume a downloadable package contains a specific resource until it is actually inspected.

Do not automatically launch:

```text
728,219 x 50 recompilations
```

or another massive data job merely because the full paper training population exists.

---

# 13. ObjectText Adaptation Population

The first ObjectText adaptation/training population should be recovered from paper-derived program identities whenever possible.

If the approximately 17,500 discovery program IDs are recoverable and compatible, they may be used as the initial ObjectText adaptation/model-training population.

This must be labeled:

```text
PROJECT-SPECIFIC OBJECTTEXT ADAPTATION CHOICE
```

It is not the same as claiming:

```text
17,500 = complete RLCompOpt model training population
```

All compared ObjectText models must use the same adaptation/model-training population.

---

## 13.1 If the original 17,500 IDs cannot be recovered

If the original discovery IDs cannot be recovered completely:

```text
the replacement subset selection rule
must be frozen before any ObjectText label
from that replacement subset is observed
```

The replacement rule may use:

```text
official dataset identity

program availability

deterministic sampling seed

program-ID ordering

compatibility constraints
```

It must not use:

```text
ObjectText improvement

candidate oracle result

Mamba performance

validation result
```

to choose a favorable subset.

Record the replacement rule once in the experiment config/progress log.

Do not repeatedly redesign the subset.

---

# 14. Step-3 Label Population

Formal Route-A label generation must include both:

```text
1. ObjectText adaptation / model-training population

2. validation population
```

It must not include:

```text
final test population

curated OOD final test
```

Reason:

Step 4 requires the validation program x K=50 label matrix to evaluate the predefined Route A/B branch.

This is part of the same required label-generation task.

It is not an additional validation experiment.

---

# 15. Route-A ObjectText Label Schema

For each:

```text
program i
candidate j
```

save at minimum:

```text
program_id

dataset_id

candidate_id

ordered_pass_sequence

candidate_length

initial_object_text_size_bytes

oz_object_text_size_bytes

prefix_object_text_size_bytes

best_prefix_index

best_object_text_size_bytes

final_object_text_size_bytes

compiler_version

target_triple

target_cpu_if_configured

host_architecture

measurement_validity

ratio_metric_validity

training_target_validity

failure_reason
```

The field:

```text
prefix_object_text_size_bytes
```

must contain absolute post-pass observations, not rewards.

The field:

```text
best_object_text_size_bytes
```

must exclude the initial pre-pass state.

ObjectText is deterministic under the frozen compiler/target setup.

A deterministic label is measured once.

Do not repeat:

```text
5 times

10 times

median

variance

bootstrap

confidence interval

B-C-B
```

for ObjectText labels.

---

# 16. Fixed Candidate Oracle

After Route-A train/adaptation and validation K=50 labels exist, compute the fixed-set oracle.

For program p:

```text
S_oracle_p =
min over candidate j
best_object_text_size_bytes[p,j]
```

The oracle is:

```text
an offline upper bound for the frozen K=50 candidate set
```

It is not:

```text
a learned method

a 45-pass policy

a baseline with the same information budget
```

Do not present the oracle as though it were a deployable learned method.

Its purpose is to answer:

```text
How much ObjectText opportunity is contained
inside the fixed candidate set?
```

and:

```text
How much of that opportunity does each model capture?
```

No new search is required.

---

# 17. Route A -> Route B Aggregate Definition

The Route A/B branch must use one unique aggregation formula.

Do not allow the Agent to choose between:

```text
pooled mean

size-weighted mean

geometric mean

dataset macro mean
```

after seeing results.

v6 freezes a paper-style dataset macro MeanOverOz criterion.

For predefined validation dataset d:

```text
OracleMeanOverOz_d =
mean over programs p in dataset d of:

    (S_Oz_p - S_oracle_p)
    ---------------------
             S_Oz_p
```

If validation contains D predefined datasets:

```text
RouteAOracleMeanOverOz =
(1 / D) *
sum over d of OracleMeanOverOz_d
```

If validation contains only one predefined dataset:

```text
D = 1
```

and the formula reduces to the program arithmetic mean for that dataset.

Do not substitute a different aggregate for the branch decision.

---

# 18. Predefined Route A/B Decision Rule

This is the only branch rule.

Label:

```text
PROJECT-SPECIFIC DECISION RULE
```

Decision:

```text
RouteAOracleMeanOverOz > 0
->
stay on Route A

RouteAOracleMeanOverOz <= 0
->
Route B is authorized
```

Do not use vague criteria such as:

```text
useful

meaningful

large enough

sufficient
```

Do not add:

```text
0.5%

1.0%
```

or another post-hoc threshold unless the user explicitly freezes that value before looking at the relevant validation result.

This branch is calculated from the already-required validation label matrix.

It is not a new Headroom Gate.

---

# 19. Route B Is a Major-Compute Fallback

Route B is not a default step.

It is a:

```text
MAJOR-COMPUTE FALLBACK
```

Route B may start only if:

```text
the predefined Route-A validation criterion
has already authorized it
```

using the existing label matrix.

The Agent must not start Route B because:

```text
Route A "looks weak"

Mamba has not won yet

more search may be safer
```

Route B is expensive because the original discovery stage alone is approximately:

```text
17,500 programs
x
200 random episodes/program
x
45 passes/episode
```

That discovery stage is not the whole coreset-construction cost.

After discovery produces the retained candidate pool `M`, the RLCompOpt
coreset method also requires cross-evaluating the retained candidates on the
required discovery programs to construct the full program-by-candidate reward
matrix before row normalization and greedy selection.

Therefore the full Route-B cost includes:

```text
random discovery cost
+
program x retained-candidate cross-evaluation cost
+
reward-matrix construction
+
greedy K=50 selection
```

ObjectText evaluation is more expensive than simple IR instruction-count
measurement because it requires object-code lowering/measurement.

If Route B is actually authorized, recover the official retained-candidate and
reward-matrix data structure first and estimate the required resources from the
recovered candidate-pool size `M`.

This resource estimate is planning for an already-authorized paper method.
It must not become a new Route-B Feasibility Gate.

Therefore Route B must never auto-start from vague language.

---

# 20. Route-B Discovery Protocol

If Route B is authorized, reuse the RLCompOpt discovery algorithm as closely as possible.

Paper-derived structure:

```text
approximately 17,500 discovery programs

approximately 200 random episodes/program

45 random LLVM pass applications/episode

paper-compatible candidate extraction

paper-compatible same-state handling

paper-compatible deduplication/tie handling

retained candidate pool M

full program x candidate reward-matrix construction

per-program reward normalization over the full pre-greedy pool M

greedy submodular coreset selection

K=50
```

The mandatory Route-B sequence is:

```text
random discovery
->
retain/deduplicate candidate pool M
->
for every required discovery program i:
    for every retained candidate j in M:
        independently apply candidate j to program i
        measure its candidate reward
->
construct R[N, M]
->
normalize each row across the full M candidates
->
greedy submodular selection
->
K=50
```

`R[N,M]` is a required part of coreset construction.

Do not approximate the paper method by keeping only each discovery program's
own locally discovered best candidate reward.

Only the optimization objective changes:

```text
original RLCompOpt reward
->
ObjectText-derived reward
```

Do not replace Route B with:

```text
beam search

MCTS

genetic algorithm

Bayesian optimization

custom 32-pass search

custom 64-pass search

custom pass subset

128-candidate random search
```

---

# 21. Route-B Positive Reward Domain

Route-B coreset construction must preserve the positive ratio-style reward domain required by the RLCompOpt coreset method.

For program i and candidate j:

```text
S_O0_i =
absolute O0 ObjectText observation

S_candidate_ij =
best post-pass ObjectTextSize
for candidate j
(initial state excluded)
```

Define:

```text
raw_coreset_reward_ij =
S_O0_i / S_candidate_ij
```

This gives:

```text
raw_coreset_reward_ij > 0
```

and:

```text
reward > 1
->
candidate is smaller than O0

reward < 1
->
candidate is larger than O0
```

This is preferred over a signed difference such as:

```text
S_O0 - S_candidate
```

because the signed difference can be negative and does not preserve the paper's positive reward-domain semantics.

If the recovered official implementation adds a paper-required detail to the ratio convention, follow the official implementation and record the difference.

---

# 22. Route-B Per-Program Row Normalization

Before the greedy submodular coreset objective, normalize each program row
of the complete pre-greedy reward matrix:

```text
R[N, M]
```

For program i:

```text
normalized_coreset_reward_ij =
raw_coreset_reward_ij
/
max over j in the full retained candidate pool M
    (raw_coreset_reward_ij)
```

The `max_j` is taken over the complete retained candidate pool before K=50
greedy selection. It is not computed only over the final K=50 set.

Therefore:

```text
best available candidate for each program
has normalized reward = 1
```

Then use the paper-compatible greedy submodular K=50 selection.

This normalization belongs to:

```text
coreset construction
```

It is not the same as:

```text
NVP temperature softmax normalization
```

The two must remain separate in code, schema, and documentation.

---

# 23. Algorithmic IR Hashing Exception

RLCompOpt discovery may use IR identity such as:

```text
IrSha1
```

for same-state detection/candidate processing.

This is:

```text
algorithmic state identity
```

and is allowed when required by the paper method.

It is not the same as:

```text
artifact-integrity hashing
```

The general project ban on arbitrary SHA256 validation must not accidentally remove an algorithmic IR-state identity operation required by RLCompOpt.

---

# 24. Learning Quantity Definitions

Three quantities must remain separate.

## 24.1 Absolute ObjectText cost

```text
raw_object_text_size
```

Meaning:

```text
absolute bytes
lower is better
```

Source:

```text
ObjectText observation
```

Used for:

```text
label storage

candidate oracle

final compiler reporting
```

---

## 24.2 Candidate learning value

```text
candidate_value
```

Meaning:

```text
higher is better
```

Used for:

```text
NVP supervision

MLP/LSTM/Transformer/Mamba ranking
```

The exact NVP value/reward transformation must first be recovered from the official RLCompOpt code/paper.

The ObjectText adaptation must preserve:

```text
smaller ObjectText
->
larger candidate value
```

Do not guess an undocumented formula when the official implementation defines it.

---

## 24.3 Reporting metric

Primary reporting quantity:

```text
reduction_vs_Oz =
(S_Oz - S_model) / S_Oz
```

Higher is better.

Optionally also store:

```text
size_ratio_vs_Oz =
S_model / S_Oz
```

Lower is better.

---

# 25. NVP Soft-Target Direction

Never use:

```text
Softmax(ObjectTextSizeBytes / T)
```

Raw bytes are lower-is-better, while softmax gives larger mass to larger numbers.

Correct structure:

```text
absolute ObjectText cost

->
paper-compatible higher-is-better candidate value

->
Softmax(candidate_value / T)

->
NVP normalized soft target
```

Do not confuse this softmax normalization with Route-B coreset row normalization.

---

# 26. Program Representation

The paper anchor may use:

```text
Autophase-NVP

GEAN-NVP / graph representation where feasible
```

The exact paper baseline should be reused as closely as the recovered official resources allow.

For the controlled architecture comparison:

```text
MLP
LSTM
Transformer
Mamba
```

must receive semantically equivalent information.

Required common information:

```text
same program representation

same K=50 candidate set

same ordered candidate pass tokens

same candidate length

same position information

same target values

same training population

same validation population

same inference budget
```

---

# 27. MLP Role and Input

MLP is retained as a mandatory controlled baseline.

Purpose:

```text
separate the value of explicit candidate information
from the value of a sequence architecture
```

MLP must not receive only:

```text
candidate_id
```

while the sequence models receive full ordered pass tokens.

MLP should receive a deterministic fixed representation of the same candidate sequence information, for example:

```text
fixed maximum candidate length

ordered pass-token embeddings or equivalent encoding

padding

explicit mask

explicit positional information
```

Padding/mask rules must be fixed and shared across programs/candidates.

The implementation must avoid using padding artifacts as an accidental candidate-ID shortcut.

MLP must appear in:

```text
validation comparison

final ObjectText comparison

final report
```

If it is trained, it must be reported.

---

# 28. Mamba Research Role

Mamba does not act as:

```text
raw next-pass policy

124-way pass generator

beam-search controller

dynamic candidate composer
```

Mamba models:

```text
(program representation,
 ordered candidate pass sequence)

->
candidate utility
```

The research motivation is:

```text
pass order

pass repetition

internal pass interaction

program-conditioned sequence compatibility
```

Do not make the central claim:

```text
Mamba is guaranteed to win because the sequences are long
```

The paper coreset sequence length is moderate.

Whether Mamba beats LSTM/Transformer is an empirical result.

---

# 29. Hypotheses

The project uses hypotheses, not Research PASS/FAIL Gates.

## H1 - Compiler outcome

```text
Mamba-selected sequences reduce ObjectTextSize
relative to LLVM -Oz on unseen programs.
```

## H2a - System / representation gain

Compare:

```text
MambaPO
vs
paper-style ObjectText-adapted NVP
```

Interpretation:

This comparison may change:

```text
program representation / program encoder
+
explicit candidate sequence representation
+
model architecture
```

For example, a paper-style GEAN-NVP anchor and an Autophase-based controlled
Mamba model do not share the same program encoder.

Therefore H2a is a system-level comparison.

If NVP and Mamba use the same program representation, the program-encoder
difference disappears, but the explicit candidate representation and model
architecture still differ.

Do not attribute an H2a gain solely to candidate sequence representation or
solely to Mamba architecture unless the corresponding variable is actually
controlled.

## H2b - Architecture gain

Compare Mamba with:

```text
MLP
LSTM
Transformer
```

under:

```text
same program representation

same candidate sequence tokens

same target

same data

same inference protocol

same model-selection budget

reported trainable parameter count

predefined comparable capacity scale / capacity envelope
```

No arbitrary percentage tolerance is required.

The rule is:

```text
prefer materially comparable parameter scale

report trainable parameter count for every controlled model

disclose any material capacity difference
in validation and final comparison tables
```

This comparison can primarily support an architecture-level claim only when
the capacity differences are controlled or explicitly disclosed.

## H3 - Runtime secondary question

```text
What runtime effect do the frozen size-optimized
MambaPO binaries have relative to -O3 and -Oz?
```

Runtime does not become the training objective.

---

# 30. Fair Finite Model-Selection Budget

Open-ended Mamba tuning is prohibited.

Before architecture model selection starts, freeze one common finite procedure covering:

```text
maximum number of configurations per architecture

maximum training budget per configuration

early-stopping rule

hyperparameter-selection seed policy

checkpoint-selection rule

capacity envelope / target parameter scale

trainable parameter-count reporting
```

The scientific configuration-selection metric is frozen:

```text
Validation policy-45 dataset-macro MeanOverOz
```

For validation dataset d:

```text
ValidationMeanOverOz_d =
mean over programs p in d of:

    (S_Oz_p - S_policy45_p)
    -----------------------
             S_Oz_p
```

For D predefined validation datasets:

```text
ValidationFinalMeanOverOz =
(1 / D) *
sum over d of ValidationMeanOverOz_d
```

Higher `ValidationFinalMeanOverOz` selects the better hyperparameter
configuration.

Training loss or another frozen loss-based quantity may be used for early
stopping, but it must not replace `ValidationFinalMeanOverOz` as the scientific
configuration-selection metric.

Separate hyperparameter selection from final stochastic replication.

Paper-aligned final replication rule:

```text
Stage A:
use ValidationFinalMeanOverOz to select one
hyperparameter configuration per architecture
under the frozen finite model-selection budget

Stage B:
retrain/evaluate the selected configuration
with 3 predefined random seeds
```

The use of three final random seeds is paper-aligned.

The exact cross-seed reporting formula below is a frozen MambaPO reporting
definition unless the official RLCompOpt implementation is later recovered to
prove the identical aggregation hierarchy.

Freeze one shared seed-ID set before final training:

```text
final_seed_set = {s1, s2, s3}
```

The same `final_seed_set` must be used for every stochastic controlled learned
method:

```text
MLP
LSTM
Transformer
Mamba
```

and for any stochastic ObjectText-NVP reproduction included in the same
seed-level comparison.

Do not assign architecture-specific seed sets.

For each final seed s:

```text
1. compute MeanOverOz_d,s for every final dataset d

2. compute:
   FinalMeanOverOz_s =
   macro mean over the predefined final datasets
```

The final report must show all three individual values:

```text
FinalMeanOverOz_s1
FinalMeanOverOz_s2
FinalMeanOverOz_s3
```

The primary cross-seed result is:

```text
FinalMeanOverOz_3seed =
(
    FinalMeanOverOz_s1
  + FinalMeanOverOz_s2
  + FinalMeanOverOz_s3
) / 3
```

Do not choose the best seed, median seed, or a hand-picked representative seed
as the primary result.

An additional dispersion statistic may be reported, but if the exact RLCompOpt
paper convention cannot be recovered, label that statistic:

```text
PROJECT-SPECIFIC REPORTING STATISTIC
```

The individual three seed results and their arithmetic mean remain mandatory.

If 3-seed replication is impossible for a concrete resource reason, freeze and
record a project-specific alternative before final training. Never select a
favorable seed after seeing final results.

Use paper/repository values where an appropriate setting exists.

If project-specific values are needed, label them:

```text
PROJECT-SPECIFIC MODEL-SELECTION BUDGET
```

All controlled architectures must receive comparable development resources.

Forbidden:

```text
Mamba:
40 configurations

Transformer:
3 configurations

then claim:
Mamba architecture wins
```

Do not keep tuning Mamba until validation becomes positive.

After the finite model-selection procedure is exhausted, accept the validation result.

---

# 31. Mandatory Metrics

Only four metrics are mandatory for the main code-size study.

## Metric 1

```text
ObjectTextSizeBytes
```

## Metric 2

```text
reduction_vs_Oz
```

## Metric 3

```text
fixed candidate oracle
```

## Metric 4

```text
policy45_regret
```

For program p:

```text
S_policy45_p =
best ObjectTextSize observed by the actual frozen
deterministic 45-scored-pass learned policy

S_oracle_p =
offline best post-pass ObjectTextSize among
all candidates in the frozen K=50 set

policy45_regret_p =
S_policy45_p - S_oracle_p
```

Lower is better.

The mandatory regret is the actual 45-pass policy regret.

If top-1 candidate quality is analyzed separately, name it explicitly:

```text
top1_regret
```

and keep it optional.

Do not mix top-1 regret and policy-45 regret across methods.

At most one tie-aware ranking metric may be used as an optional diagnostic.

Remove the undefined metric:

```text
candidate coverage
```

from mandatory execution.

Do not create new coverage definitions during Step 4.

---

# 32. Candidate Oracle Interpretation

The fixed candidate oracle is:

```text
offline best-of-K=50 upper bound
```

It answers:

```text
Does the frozen candidate space contain
a better ObjectText choice for this program?
```

It is not:

```text
a deployable learned method

a method with the same information budget as NVP/Mamba

a 45-pass online policy baseline
```

In final tables, clearly label it:

```text
Offline K=50 Oracle
```

or equivalent.

Do not present it as a fair learned-method competitor.

---

## Final Primary Aggregate

The final H1/H2 primary aggregate is frozen before final-test unseal.

Use the same dataset-macro MeanOverOz structure used by the Route-A validation
criterion.

For final dataset d:

```text
MeanOverOz_d =
mean over programs p in dataset d of:

    (S_Oz_p - S_method_p)
    ---------------------
             S_Oz_p
```

For D predefined final datasets:

```text
FinalMeanOverOz =
(1 / D) *
sum over d of MeanOverOz_d
```

The primary final aggregate is therefore:

```text
per-program arithmetic mean within each dataset
->
macro mean across predefined datasets
```

Per-dataset values must also be reported.

Do not change the primary final aggregate after unsealing the final test.

Do not substitute pooled program mean, size-weighted mean, or geometric mean as
the primary result because one of those looks more favorable.

Additional secondary aggregates may be reported only if clearly labeled and
predefined before final-test inspection.

---

# 33. cBench and Final-Test Semantics

Model/configuration development uses:

```text
training population

validation population
```

Final test remains sealed until Step 10 freeze.

After freeze, final code-size evaluation may include:

```text
official held-out test

curated OOD code-size sets

cBench code-size
```

Runtime remains separate.

Do not inspect cBench code-size results during model development if cBench is part of the final OOD test.

Do not use runtime to choose a different sequence after final ObjectText evaluation.

---

# 34. Runtime Position

Runtime is secondary and final-stage only.

Runtime does not perform:

```text
headroom verification

candidate-by-candidate training reward

model selection

sequence tuning

training permission
```

Order:

```text
freeze model/config

->
final ObjectText evaluation

->
freeze selected binaries

->
runtime correctness + timing
```

This prevents runtime from silently becoming a second tuning objective.

---

# 35. Runtime Methodology Sources

The runtime method is literature-informed but project-specific.

It draws from:

```text
CITROEN
ACPO
Protean
2026 LLVM per-pass empirical study
```

CITROEN motivates:

```text
adaptive repeated execution

RSE < 1% for ordinary candidate measurement

stricter final-best measurement around RSE < 0.3%
```

ACPO motivates:

```text
single-thread execution

CPU binding

repeated measurement

remeasurement when variance is high
```

Protean contributes:

```text
small fixed initial repeat count

CPU binding

repeat when variance threshold is exceeded
```

The 2026 LLVM per-pass study contributes:

```text
fixed CPU core

warmups

explicit noise control
```

The final MambaPO runtime protocol is:

```text
PROJECT-SPECIFIC SYNTHESIZED RUNTIME PROTOCOL
```

It is not an exact copy of one paper.

---

# 36. Runtime Workload Must Be Frozen Before Final Results

Step 10 must freeze the runtime workload definition before Step 11 final code-size results are used to make any runtime-selection decision.

Freeze:

```text
runtime_benchmark_ids

runtime_input_ids

runtime_workload_amplification rule/value

runtime_cache_policy

runtime_correctness_method
```

Only benchmarks that are actually runnable in the frozen environment may enter the runtime subset.

The runtime subset must be determined from:

```text
benchmark executability

available input

available correctness mechanism

predefined experiment scope
```

It must not be chosen from:

```text
whether Mamba's final code-size result looks favorable

whether a benchmark's runtime result looks favorable
```

Do not delete an unfavorable runtime benchmark after Step 11 simply because the result regresses.

---

## 36.1 If workload amplification is required

The amplification rule must be frozen before comparing final Mamba runtime results.

Preferred principle:

```text
use the benchmark's standard larger input when available
```

or:

```text
repeat the same workload inside one timed invocation
using the same repetition count for every compared binary
```

If a deterministic amplification count cannot be selected from benchmark metadata alone, use only a pre-final baseline calibration rule that does not inspect Mamba performance, and record that rule before final runtime comparison.

The purpose is measurement stability, not result selection.

---

# 37. Runtime Correctness

Correctness is required before accepting a runtime number.

This is not optional validation bureaucracy.

A binary with:

```text
large speedup
+
very low RSE
```

is invalid if it computes the wrong result.

For each runtime benchmark/method, store:

```text
correctness_result

correctness_method

semantic_correctness
```

The semantic-correctness state is three-valued:

```text
semantic_validated_pass

semantic_validated_fail

execution_only_unverified
```

Correctness method priority:

```text
1. benchmark-provided semantic validator

2. reference output

3. benchmark-provided checksum
```

If only exit status is available:

```text
correctness_method = exit_status_only
semantic_correctness = execution_only_unverified
```

`exit_code == 0` proves only that the program executed without reporting
failure through its exit status.

It must not be reported as `semantic_validated_pass` unless a stronger
semantic mechanism actually validates program behavior.

An `execution_only_unverified` timing result may be retained and reported, but
it must not be described as a semantically validated runtime result.

H3 conclusions must distinguish:

```text
semantically validated runtime benchmarks
```

from:

```text
execution-only unverified runtime benchmarks
```

Do not claim:

```text
correctness was preserved
```

for an execution-only benchmark solely because `exit_code == 0`.

Use CompilerGym validation mechanisms where supported and appropriate.

Do not use:

```text
SHA256 binary

SHA256 logs

SHA256 arbitrary output files
```

as a substitute for semantic correctness.

---

# 38. Runtime Measurement Protocol

For every compared binary use:

```text
same machine

same benchmark input

same fixed CPU core

single-thread execution

same compiler target

same workload amplification

same cache policy
```

Run:

```text
3 warmup runs
```

Warmups are not included in timed statistics.

Then start with:

```text
5 timed runs
```

Compute:

```text
mean

median

sample standard deviation

RSE

timed run count
```

RSE:

```text
standard_error =
sample_std / sqrt(n)

RSE =
standard_error / mean
```

Stopping:

```text
if RSE <= 1%:
    stop

if RSE > 1%:
    append timed runs
    recompute
```

Project hard cap:

```text
20 timed runs
```

The following are project-specific synthesized choices:

```text
3 warmups

5 initial timed runs

20-run hard cap
```

Do not misattribute them as the exact CITROEN protocol.

---

## 38.1 Final publication winners

For final publication-quality winners, a stricter target may be used:

```text
RSE <= 0.3% when practical
```

This does not create a Gate.

It does not authorize unlimited reruns.

If the project-specific hard cap is exceeded only because final publication-quality measurement requires it, record the reason explicitly rather than silently expanding all measurements.

---

# 39. Runtime Short-Benchmark Rule

If a timed invocation is clearly too short for reliable measurement:

prefer:

```text
larger benchmark input
```

or:

```text
repeat the workload inside one timed invocation
```

Aim for approximately:

```text
1 second or longer effective workload
```

when practical.

Apply exactly the same amplification to every compared binary.

Do not first respond to a millisecond-scale workload by adding increasingly complex statistics.

---

# 40. Runtime Cache Policy

Do not automatically flush page cache merely because another paper did.

Default:

```text
warm-cache protocol
```

unless the benchmark's intended semantics explicitly require cold-cache I/O.

If cold-cache behavior is required:

```text
apply the same procedure to every compared binary
```

and record:

```text
cache_policy = cold
```

Otherwise:

```text
cache_policy = warm
```

Do not request root privileges solely to imitate a paper's cache-flush procedure when that procedure is not necessary for the benchmark semantics.

---

# 41. Runtime Interleaving and Bootstrap Policy

Do not enable B-C-B or complex interleaving by default.

Only add minimum interleaving if raw timing data show a concrete systematic drift.

Forbidden default escalation:

```text
speculative drift concern

->
B-C-B for every binary

->
bootstrap every binary

->
another qualification phase
```

Do not bootstrap individual candidates or individual binaries by default.

A bootstrap confidence interval may be used only for the final cross-benchmark aggregate when needed for publication.

---

# 42. Runtime Result Schema

For each runtime result store:

```text
benchmark_id

method

binary_or_sequence_id

compiler_config

cpu_core

thread_count

input_id

workload_amplification

cache_policy

warmup_run_count

timed_runs_raw

mean_runtime

median_runtime

sample_std

RSE

timed_run_count

stopping_reason

correctness_result

correctness_method

semantic_correctness
```

Keep raw timed values.

Do not create a SHA256 manifest for these ordinary runtime artifacts.

---

## Claim Boundary for Fixed K=50 Candidate Sequences

In the frozen K=50 setting, each candidate token sequence uniquely identifies
one candidate.

Therefore the current experiment can support:

```text
structured sequence-aware candidate modeling
improves the overall selection system
```

and, under H2b controls:

```text
Mamba performs better or worse than
MLP/LSTM/Transformer under the same
structured candidate representation
```

The current protocol does not by itself prove the stronger causal claim:

```text
pass order itself caused the improvement
```

because no order-shuffling or candidate-ID-only causal ablation is mandatory in
v6.

Do not add such an ablation automatically.

Only add the minimum necessary ablation later if the final paper explicitly
requires the stronger causal claim.

---

# Part B. Execution Plan

# 43. Execution Philosophy

The execution plan is causal.

It must not be treated as a sequence of invented Gates.

The route is:

```text
recover official semantics

->
recover candidates/data

->
generate ObjectText labels

->
choose Route A or paper-derived Route B using the frozen formula

->
train fair baselines and Mamba

->
validation/model selection

->
freeze

->
final ObjectText

->
freeze runtime binaries

->
runtime correctness and timing

->
final report
```

---

# Step 0. Recover Official RLCompOpt Foundation

Recover and record:

```text
official RLCompOpt repository

official paper

official CompilerGym requirements

official action mapping

official K=50 resource

official evaluation code

official reward/value implementation

official data/split resources

official compiler/candidate failure handling

official invalid-sample / invalid-observation semantics

official K-way supervised-target completeness behavior

official incomplete-candidate-rollout handling
```

Confirm/freeze:

```text
LLVM/compiler environment

target/platform metadata

independent candidate rollout

45-pass scored budget

sampling = False deterministic ranking

ObjectText observation API

NVP value/reward direction

measurement-validity / failure policy
```

No performance experiment is required.

Do not benchmark headroom.

Do not recreate K=50 if the official candidate set is available.

---

# Step 1. Recover the Official K=50 Candidate Set

Priority:

```text
official candidate resource
```

Confirm action mapping once.

Record:

```text
candidate_id

ordered pass sequence

candidate length
```

Do not start discovery search.

Do not modify the candidate set based on ObjectText performance.

This is Route A.

---

# Step 2. Recover Program Populations and Splits

Recover separately:

```text
coreset_discovery_pool

model_train_pool

validation_pool

final_test_pool

curated_ood_test_pool
```

If the paper-derived 17,500 discovery IDs are available, recover them.

If they are not completely recoverable:

freeze a deterministic replacement-subset rule before any ObjectText result from that replacement subset is observed.

Do not touch final-test labels.

Do not touch curated OOD code-size results.

---

# Step 3. Generate Route-A ObjectText Labels

This is the first formal large-scale ObjectText task after Step 0-2 semantics
and the complete data-validity policy are frozen, including:

```text
measurement validity

ratio-metric validity

candidate rollout completeness

K=50 target completeness

common-cohort aggregate eligibility

failure-count reporting
```

Generate K=50 ObjectText labels for:

```text
A. ObjectText adaptation/model-training population

B. validation population
```

Do not generate final-test labels.

Do not generate curated OOD final-test labels.

Formal Step-3 execution structure:

```text
for each program p:

    measure/cache S_O0[p] once
    measure/cache S_Oz[p] once

    convert both baseline observations
    to scalar integer values

    for each candidate j in K=50:

        reset the original benchmark p

        execute candidate j independently

        after each pass:
            read absolute ObjectTextSizeBytes observation
            validate it using the frozen data-validity policy

            if valid:
                convert it to scalar integer
                append to prefix_object_text_size_bytes

            if invalid/failure:
                record measurement_validity + failure_reason
                apply only the already-frozen failure rule

        candidate rollout completeness:
            valid only if the complete frozen candidate
            sequence has executed successfully under
            the frozen validity semantics

        candidate best:
            minimum valid post-pass absolute size
            initial state excluded
            only for a completed valid candidate rollout

        if the rollout is incomplete:
            preserve valid prefixes only as diagnostics
            candidate training_target_validity = invalid
            do not convert the partial prefix into a completed label

        save the candidate record
        using the cached program-level S_O0/S_Oz
```

`S_O0` and `S_Oz` are deterministic program-level baselines under the frozen
compiler/target setup.

Do not recompute them inside the K=50 candidate loop.

Formal label generation must not use ObjectText reward as absolute size.

After all K=50 candidate records for program p are available:

```text
if all 50 candidate training targets are valid:
    program_training_target_validity = valid_complete_K50
else:
    program_training_target_validity = invalid_incomplete_K50_target
```

Only `valid_complete_K50` programs enter the formal K=50 supervised
train/validation target population.

The same eligible population is used by all compared supervised learned models.

ObjectText measurements are deterministic and are taken once.

---

## Step 3 focused preflight

Before the full Step 3 job, one tiny focused API check is allowed:

```text
1 program

1 candidate

a few passes
```

Confirm:

```text
label absolute size is read from observation

candidate best excludes initial state

reward is not stored as absolute size
```

If this focused check passes, begin formal Step 3.

Do not turn the check into:

```text
24 programs

128 candidates

repeated measurements

bootstrap

hash validation
```

---

# Step 4. Compute Validation Fixed-Set Oracle and Apply Route Rule

From the validation K=50 label matrix, first establish the Route-A oracle
cohort.

A validation program is eligible for the Offline K=50 Oracle only when:

```text
S_Oz ratio denominator is valid
and
oracle_K50_validity = valid_complete_K50
```

Do not compute the Offline K=50 Oracle from fewer than 50 valid frozen
candidates.

For each eligible program p:

```text
S_oracle_p = min over all 50 frozen valid candidate labels
```

For every predefined validation dataset d, calculate `OracleMeanOverOz_d` on
`P_RouteA_oracle_d` only. Then calculate `RouteAOracleMeanOverOz` as the macro
mean across all predefined validation datasets only when every required
`OracleMeanOverOz_d` is defined.

Report per validation dataset:

```text
N_total
N_complete_K50_oracle
N_ratio_valid
N_RouteA_oracle_valid
N_failed_or_invalid
```

If any required validation dataset has an empty valid oracle cohort:

```text
OracleMeanOverOz_d = undefined
RouteAOracleMeanOverOz = undefined
branch_criterion_status = undefined_due_to_invalid_required_data
```

Do not treat undefined as `<= 0`. Do not authorize Route B from NaN, empty,
missing, or invalid oracle data.

Apply the branch only when the criterion is defined:

```text
RouteAOracleMeanOverOz > 0
-> stay Route A

RouteAOracleMeanOverOz <= 0
-> Route B authorized
```

Do not calculate undefined candidate coverage for the branch and do not invent
a new threshold after seeing the result.

# Step 5. Route B Only If Authorized

If:

```text
RouteAOracleMeanOverOz > 0
```

skip Step 5 completely.

If:

```text
RouteAOracleMeanOverOz <= 0
```

Route B may run.

Route B uses the complete paper-derived coreset-construction sequence:

```text
1. run paper-derived random discovery

2. apply paper-compatible same-state handling,
   candidate retention, and deduplication

3. freeze the retained pre-greedy candidate pool M

4. for every required discovery program i
   and every retained candidate j in M:
       independently evaluate candidate j on program i
       compute raw_coreset_reward_ij

5. construct the complete reward matrix R[N, M]

6. normalize every program row across the full M candidates

7. run the paper-compatible greedy submodular objective

8. select K=50
```

The full `program x candidate` reward matrix is mandatory.

Every required `R[i,j]` cell must be valid under the frozen Route-B validity
and retry policy before row normalization begins.

If the complete `R[N,M]` matrix cannot be constructed:

```text
Route-B coreset selection = undefined
```

Stop Route-B selection. Do not drop rows/columns or impute cells.

Do not skip cross-evaluation and do not normalize only over the eventual K=50.

Before launching the cross-evaluation stage, estimate resources using the
recovered retained-pool size `M`.

This is execution planning for an already-authorized Route B, not a new Gate.

After Route B construction:

```text
freeze the ObjectText-derived K=50 candidate set
```

All later models must use that same set.

If Route B changes the candidate set, generate the required train/adaptation and validation K=50 labels for the new frozen set using the same observation-only label semantics.

Do not give Mamba a different candidate set from the baselines.

---

# Step 6. Build ObjectText-Adapted NVP Paper Anchor

Reuse official NVP implementation/data interface as far as compatible.

Recover the exact official value/reward transform.

Adapt it so:

```text
smaller ObjectText
->
larger candidate value
```

Use the same:

```text
candidate set

ObjectText labels

adaptation/training population

validation population
```

as the controlled models.

Original RLCompOpt numeric IR-count performance is paper context only.

It is not a direct ObjectText numeric baseline.

---

# Step 7. Build the Common Controlled-Model Interface

Create one shared semantic interface for:

```text
MLP

LSTM

Transformer

Mamba
```

Common information:

```text
program representation

ordered candidate pass tokens

candidate length

candidate positions/mask

candidate target value
```

The architecture may change.

The scientific input information must not.

Implement MLP's fixed representation explicitly.

Do not allow MLP to see only a candidate ID.

---

# Step 8. Freeze and Execute Fair Model Selection

Before training sweeps begin, freeze:

```text
maximum configuration count / architecture

training budget / configuration

early-stopping rule

hyperparameter-selection seed policy

checkpoint-selection rule

capacity envelope / target parameter scale

trainable parameter-count reporting
```

The scientific configuration-selection metric is already frozen:

```text
Validation policy-45 dataset-macro MeanOverOz
```

Compute:

```text
ValidationMeanOverOz_d =
mean over programs p in validation dataset d of:

    (S_Oz_p - S_policy45_p)
    -----------------------
             S_Oz_p

ValidationFinalMeanOverOz =
macro mean of ValidationMeanOverOz_d
across predefined validation datasets
```

Use higher `ValidationFinalMeanOverOz` to select one configuration per
architecture.

Do not use training loss, top-1 accuracy, or another diagnostic metric to
replace this scientific selection metric.

Use the frozen two-stage stochastic protocol:

```text
Stage A:
validation selects one hyperparameter configuration
per architecture using ValidationFinalMeanOverOz
on the frozen common validation cohort

Stage B:
retrain/evaluate each selected configuration
with the shared final_seed_set = {s1, s2, s3}
```

The same three seed IDs are used for every stochastic controlled learned
method.

The concrete integer seed IDs must be written once into the frozen experiment
configuration before Stage B begins. Do not choose different seed triplets for
different architectures.

For each final seed, compute its own per-dataset `MeanOverOz_d,s` and
`FinalMeanOverOz_s`.

Primary cross-seed result:

```text
FinalMeanOverOz_3seed =
mean(
    FinalMeanOverOz_s1,
    FinalMeanOverOz_s2,
    FinalMeanOverOz_s3
)
```

Report all three individual seed values plus this arithmetic mean.

Do not select the best final seed.

Label any non-paper numbers:

```text
PROJECT-SPECIFIC MODEL-SELECTION BUDGET
```

Then train:

```text
ObjectText-NVP anchor

MLP

LSTM

Transformer

Mamba
```

Do not keep increasing only Mamba's tuning budget until it wins.

---

# Step 9. Validation Comparison

Use validation only.

Mandatory reported models:

```text
ObjectText-NVP

MLP

LSTM

Transformer

Mamba
```

Mandatory quantities:

```text
ObjectTextSize

reduction_vs_Oz

fixed K=50 oracle

policy45_regret
```

For every dataset/method/configuration included in validation also report:

```text
N_total

N_primary_valid

N_failed_or_invalid
```

Primary cross-method validation comparisons use the same frozen common valid
program cohort per dataset.

Optionally add one tie-aware ranking diagnostic.

Do not open the final test to resolve an ambiguous validation result.

Do not redesign the compiler-search problem because one architecture loses.

---

# Step 10. Freeze Final Protocol

Before final ObjectText evaluation, freeze:

```text
candidate set

program representation

model architectures

capacity envelope / parameter-scale rule

trainable parameter counts

selected hyperparameter configurations

final 3-seed replication rule

shared final_seed_set = {s1, s2, s3}

final 3-seed reporting / arithmetic-mean aggregation rule

selected checkpoints / seed-specific checkpoints

target transform

candidate sequence encoding

deterministic ranking rule

independent rollout semantics

45-pass scored budget

compiler/target environment

measurement-validity / failure policy

K=50 target-completeness rule

incomplete-candidate-rollout rule

primary common-cohort / aggregate-denominator rule

frozen H1/H2a/H2b comparison-family method sets

empty-cohort / undefined-dataset semantics

Route-A complete-K50 oracle validity rule

Route-B complete-matrix validity rule

formal failure retry policy

failure-count reporting rule

final-test program list

final dataset-macro MeanOverOz aggregate definition

runtime benchmark IDs

runtime input IDs

runtime workload amplification

runtime cache policy

runtime correctness method

runtime timing protocol
```

After Step 10, final-test outcomes cannot change any of these choices.

Runtime workload selection must not use final Mamba code-size results.

---

# Step 11. Final ObjectText Evaluation

Unseal once:

```text
official held-out final test

curated OOD code-size tests

cBench code-size where included in the frozen OOD plan
```

Evaluate the frozen methods:

```text
LLVM -Oz
(native compiler external baseline;
not subject to the learned 45-pass budget)

ObjectText-NVP
MLP
LSTM
Transformer
Mamba
(all learned methods use the frozen 45-scored-pass protocol)
```

For every learned method, final seed, and dataset, compute the frozen
per-dataset:

```text
MeanOverOz_d,s
```

Then compute for each seed:

```text
FinalMeanOverOz_s
```

using the predefined dataset macro.

Finally compute:

```text
FinalMeanOverOz_3seed =
mean(
    FinalMeanOverOz_s1,
    FinalMeanOverOz_s2,
    FinalMeanOverOz_s3
)
```

Report all three seed-level `FinalMeanOverOz_s` values plus the arithmetic
mean.

`FinalMeanOverOz_3seed` is the primary cross-seed H1/H2 aggregate.

For each final dataset, use the already-frozen primary comparison-family
cohort:

```text
H1  -> P_H1_common_d
H2a -> P_H2a_common_d
H2b -> P_H2b_common_d
```

Do not choose the comparison-family method set after final-test unseal.

Also report for every dataset x method x seed:

```text
N_total
N_primary_valid
N_failed_or_invalid
```

so method-specific failures remain visible even when the primary aggregate uses
the common valid cohort.

Also calculate:

```text
Offline K=50 Oracle
```

where appropriate for diagnostic upper-bound analysis.

The oracle must be clearly labeled offline.

Do not run runtime selection/tuning here.

---

# Step 12. Freeze Runtime Binaries

Using the already-frozen final code-size policy, produce runtime binaries for the pre-frozen runtime subset.

Required primary runtime comparison:

```text
LLVM -O3

LLVM -Oz

MambaPO
```

If the frozen publication plan includes additional learned baselines, also include the predeclared:

```text
ObjectText-NVP

MLP and/or LSTM/Transformer
```

Do not choose a new Mamba candidate based on runtime.

Do not drop a benchmark because its expected runtime result looks unfavorable.

---

# Step 13. Runtime Correctness and Timing

For every frozen runtime binary:

```text
run the frozen correctness mechanism first

if semantic validation fails:
    runtime result is invalid

if only exit-status checking is available:
    semantic_correctness =
        execution_only_unverified
```

Execution-only unverified timing may be measured and reported, but it must be
kept separate from semantically validated runtime evidence.

Then apply the frozen timing protocol:

```text
same CPU core

single thread

3 warmups

5 initial timed runs

adaptive RSE stopping

20 timed-run project cap
```

Store raw timing data and correctness metadata.

Do not add default B-C-B, per-binary bootstrap, arbitrary hashing, or system-wide changes.

---

# Step 14. Final Report

Separate three types of result.

## Primary compiler result

```text
ObjectTextSizeBytes

per-dataset MeanOverOz_d

FinalMeanOverOz
```

For each final seed, `FinalMeanOverOz_s` is the frozen dataset-macro result.

The cross-seed primary result is:

```text
FinalMeanOverOz_3seed =
mean(FinalMeanOverOz_s1,
     FinalMeanOverOz_s2,
     FinalMeanOverOz_s3)
```

Report all three seed-level values plus their arithmetic mean.

Report these for:

```text
NVP

MLP

LSTM

Transformer

Mamba
```

## Validity and cohort accounting

For every primary validation/final comparison report:

```text
dataset_id

method

seed_id where applicable

N_total

N_primary_valid

N_failed_or_invalid
```

Also report the common program cohort size used for each dataset-level primary
cross-method aggregate.

Do not present method-specific aggregates computed on different silent
denominators as directly comparable primary results.

When `N_primary_valid < N_total`, describe the primary size result as
performance on the predefined common valid cohort and report failure counts
separately. Do not describe common-cohort MeanOverOz as unconditional
performance over the entire frozen dataset population.

If a required primary dataset common cohort is empty, report that dataset
metric and the affected family-level macro aggregate as undefined / unavailable
under the frozen empty-cohort rule.

## Candidate-space analysis

```text
Offline K=50 Oracle

policy45_regret
```

Optional diagnostics, if used, must be explicitly named, for example:

```text
top1_regret
```

Do not mix top-1 regret with the mandatory policy-45 regret.

## Runtime secondary result

```text
runtime vs -O3

runtime vs -Oz

correctness

RSE

run count
```

Interpret H2 carefully:

```text
MambaPO vs NVP
=
system/representation + architecture comparison

Mamba vs MLP/LSTM/Transformer
=
controlled architecture comparison
```

Do not compare the paper's original IR-count percentage directly against this project's ObjectText percentage as if they were the same metric.

---

# Part C. Agent Execution Policy

# 44. Literature-First Rule

For:

```text
action space

candidate set

sequence/budget semantics

discovery population

episode count

inference orchestration

reward semantics

program split
```

use the priority:

```text
1. primary paper

2. official repository implementation

3. official framework documentation

4. project-specific choice
```

If level 4 is necessary, label it:

```text
PROJECT-SPECIFIC CHOICE
```

Do not invent compiler-search parameters because they are convenient to code.

---

# 45. Official Implementation Before Natural-Language Guessing

For concrete execution semantics, prefer the official RLCompOpt implementation.

Example:

```text
official:
independent candidate reset/rollout

forbidden reinterpretation:
sequential candidate best-state chaining
```

Do not infer a different algorithm from ambiguous prose if the repository defines the evaluation path.

---

# 46. No Independent Headroom Gate

Do not create:

```text
Headroom Gate

Search Feasibility Gate

Oracle Gate

Qualification Gate

Teacher Gate

Research PASS/FAIL Gate
```

The Route A/B branch is already predefined.

Do not re-run:

```text
Random/Greedy headroom search

OpenTuner headroom search

ACPO reproduction

CITROEN reproduction
```

solely to prove that optimization opportunity exists.

---

# 47. No Invented Experiment Numbers

Do not spontaneously introduce:

```text
1000 programs

16 trajectories/program

32-pass horizon

beam width 8

budget 128

custom pass subset
```

into the v6 compiler-search problem.

Paper/repository-defined compiler-search quantities come from RLCompOpt.

Model architecture hyperparameters may be project-specific when necessary, but must be clearly labeled and tuned under the finite fair model-selection budget.

---

# 48. Minimal Validation Philosophy

Validation exists to prevent a concrete error in the current task.

Default after code changes:

```text
syntax/import check if relevant

+
directly affected focused test
```

Cross-module changes may use one tiny integration smoke.

Do not convert a smoke test into a scientific experiment.

Example:

```text
Does the ObjectText labeler read observation instead of reward?
```

Correct validation:

```text
1 program
1 candidate
a few passes
```

Incorrect validation:

```text
dozens of programs
hundreds of candidates
repeated deterministic measurements
bootstrap
hash manifests
```

---

# 49. One Failure Mode, One Adequate Check

If a focused unit/integration test already proves:

```text
candidate rollout resets the same benchmark independently
```

do not also create:

```text
manual proof script

second integration report

hash proof

large rerun
```

Principle:

> One plausible failure mode, one adequate check.

---

# 50. Deterministic Metric Rule

ObjectTextSize under the frozen compiler/target is deterministic.

Therefore formal ObjectText labels are evaluated once.

Do not use:

```text
repeat 10 times

median

variance

CI

bootstrap
```

for deterministic ObjectText values.

Runtime is different and follows the runtime protocol.

---

# 51. Hash Policy

Default:

```text
do not compute SHA256 for ordinary project artifacts
```

Do not hash:

```text
local scripts

normal configs

checkpoints

normal reports

temporary output

every dataset shard

source tree
```

Allowed cases:

```text
external downloaded artifact integrity

explicit frozen-artifact requirement

suspected corruption

explicit user request

algorithmic IR state identity required by RLCompOpt
```

Do not create a hash manifest merely to prove that files the Agent just wrote have not changed.

---

# 52. Expensive-Operation Rule

Before a costly operation, ask whether its output directly becomes:

```text
formal ObjectText labels

paper-required Route-B discovery data

model training result

final evaluation data

evidence required to fix a currently observed concrete bug
```

If none apply:

```text
do not run it
```

Especially forbidden:

```text
hours-long "just to be safe" rerun

pure headroom search

duplicate full experiment

untriggered Route B
```

---

# 53. Route B Cannot Auto-Start

Route B is authorized only by:

```text
RouteAOracleMeanOverOz <= 0
```

under the frozen validation macro formula.

Do not interpret:

```text
+0.1%

+0.2%

+0.8%
```

as "not meaningful enough" and launch Route B.

No vague threshold exists.

---

# 54. Fair Architecture Development

MLP, LSTM, Transformer, and Mamba receive comparable finite model-selection resources.

They must also report trainable parameter count and use the predefined
comparable capacity scale / envelope.

Material capacity differences must be disclosed.

Final stochastic replication follows the frozen seed protocol:

```text
select hyperparameters on validation
->
retrain/evaluate the selected configuration
with 3 predefined random seeds
```

Do not choose the best final seed.

Do not give Mamba substantially more attempts.

Do not continue modifying only Mamba until validation becomes positive.

If Mamba loses after the frozen procedure:

```text
record the result
```

Do not change:

```text
test set

metric

45-pass budget

candidate orchestration

baseline
```

to force a positive outcome.

---

# 55. Final-Test Integrity

Once Step 11 begins, do not:

```text
retune model

change coreset

change target transform

change candidate ranking

change candidate representation

select checkpoint using final test

change runtime subset using final code-size outcome
```

Final test is a one-way transition.

---

# 56. Runtime Is Not a New Optimization Loop

Runtime is secondary.

Do not:

```text
select a different candidate because it runs faster

remove a benchmark because Mamba regresses

add a benchmark because Mamba improves

retune the model using runtime
```

The frozen size-optimized binaries are what runtime evaluates.

---

# 57. Runtime Correctness Is Required but Proportionate

Correctness is a necessary validity condition.

Use the strongest existing benchmark mechanism available.

Do not build a new verification framework if:

```text
reference output

benchmark validator

checksum

exit status
```

already provides the required check.

Do not use arbitrary artifact hashing as correctness.

---

# 58. Runtime Noise Handling

Runtime noise is handled inside the measurement protocol.

Do not create:

```text
Runtime Qualification Gate

Runtime Stability Gate

multiple validation rounds
```

If raw timing data show a concrete issue:

```text
identify the issue

fix the issue if appropriate

perform the minimum necessary remeasurement
```

Do not escalate speculative concerns.

---

# 59. Documentation Restrictions

Keep persistent documentation minimal.

Use:

```text
routeGuide_v6.md

project_notes/progress.md

required configs

real experiment outputs
```

Do not create by default:

```text
audit_report.md

verification_report.md

gate_report.md

final_verification.md

hash_manifest.md

sanity_report.md
```

Temporary smoke/debug files should stay outside the repository or be removed.

Important scientific decisions go into the existing progress document.

---

# 60. Git Policy

Use meaningful commits such as:

```text
Recover RLCompOpt candidate protocol

Add observation-only ObjectText label generation

Add ObjectText-NVP baseline

Add controlled sequence models

Add frozen final evaluation
```

Do not create commits solely for:

```text
hash verification

smoke metadata

another validation report
```

A task is complete when the requested output exists and the focused relevant checks pass.

---

# 61. Legacy Experiments

The following belong only in historical progress notes:

```text
raw pass-by-pass 32-pass search

1000 x 16 random trajectory dataset

budget-128 beam/random experiments

old runtime Headroom Gate
```

Do not use them as the v6 training distribution.

Do not rerun them.

Do not let their old parameters leak into the RLCompOpt-anchored route.

---

# 62. Immediate Next Execution

After adopting this Guide, the next work is:

```text
Step 0:
recover exact RLCompOpt environment/resources/evaluation semantics
and freeze failure/invalid-sample handling before formal Step 3

Step 1:
recover K=50

Step 2:
recover program populations/splits

Step 3:
run the one focused observation-vs-reward check,
then start formal ObjectText label generation
for adaptation/train + validation
```

There is no additional:

```text
headroom experiment

qualification experiment

large pre-label validation

runtime experiment

hash audit
```

before formal Step 3.

---

# 63. References

RLCompOpt paper:

```text
https://proceedings.mlr.press/v202/liang23f/liang23f.pdf
```

RLCompOpt official repository:

```text
https://github.com/facebookresearch/RLCompOpt
```

RLCompOpt official evaluation implementation:

```text
https://github.com/facebookresearch/RLCompOpt/blob/main/rlcompopt/model_testing.py
```

CompilerGym LLVM environment:

```text
https://compilergym.com/llvm/index.html
```

CITROEN:

```text
https://eprints.whiterose.ac.uk/224241/1/ipdps25-1.pdf
```

ACPO:

```text
https://arxiv.org/abs/2312.09982
```

Protean Compiler:

```text
https://arxiv.org/abs/2602.06142
```

LLVM per-pass empirical study:

```text
https://arxiv.org/abs/2606.31238
```

---

# 64. Codex Top-Level Rules

Place the following principles at the top of the Agent's project instructions:

```text
MambaPO V6 AGENT RULES

The research route is frozen.

1. Implement the published method before inventing a new one.
2. Prefer official RLCompOpt executable semantics over guessing.
3. ObjectText absolute labels are observation-only.
4. Separate measurement_validity, ratio_metric_validity,
   training_target_validity, oracle_K50_validity, and failure_reason.
5. Candidate training labels require completed rollouts; partial failed
   rollouts are diagnostics only.
6. K=50 supervised eligibility requires 50/50 valid candidate labels.
   This is PROJECT-SPECIFIC DATA-VALIDITY POLICY, not an explicit paper
   compiler-failure rule.
7. Offline K=50 Oracle also requires all 50 valid frozen candidates.
8. Do not invent penalties, epsilon denominators, imputation, or
   outcome-driven deletion.
9. Candidates independently reset/roll out from the same benchmark.
10. sampling=False; learned-method scored budget = 45 passes.
11. LLVM -Oz is a native external baseline, not constrained to 45 passes.
12. Route A uses the published IR-derived K=50 set.
13. Route B runs only when a defined frozen Route-A criterion authorizes it.
14. Undefined/NaN/empty Route-A oracle data never authorizes Route B.
15. Route B requires a complete valid R[N,M] before row normalization and
    greedy K=50. Do not drop rows/columns or impute invalid cells.
16. Cache O0/Oz once per program.
17. formal automatic_retry_count = 0; do not selectively retry failures.
18. S_Oz <= 0 is ratio-invalid, not automatically measurement-invalid.
19. Step 3 uses adaptation/train + validation only; final/OOD stays sealed.
20. All supervised models use the same eligible complete-K50 population.
21. Configuration selection uses validation policy-45 dataset-macro MeanOverOz.
22. Freeze one shared final_seed_set = {s1,s2,s3}; write concrete IDs to config.
23. Report all three seed-level FinalMeanOverOz values plus arithmetic mean.
24. Freeze primary comparison families before final-test unseal:
    H1  = Mamba vs native -Oz
    H2a = MambaPO vs ObjectText-NVP
    H2b = MLP/LSTM/Transformer/Mamba
25. Each family uses its predefined common valid cohort per dataset.
26. Common-cohort MeanOverOz is conditional on joint successful evaluation;
    report N_total, N_primary_valid, and N_failed_or_invalid.
27. If a required primary dataset common cohort is empty, that dataset metric
    and the corresponding family macro aggregate are undefined. Never drop it
    or set it to zero.
28. H2a is system-level; H2b is the controlled architecture comparison.
29. Mandatory regret is policy45_regret against Offline K=50 Oracle.
30. Final primary aggregate is frozen dataset-macro MeanOverOz.
31. Freeze runtime benchmark/input/workload/cache/correctness before final.
32. execution_only_unverified timing is not semantic correctness evidence.
33. Do not repeat successful deterministic ObjectText measurements.
34. Do not create new Headroom/Oracle/Feasibility/Qualification/Teacher/
    Data-Validity/Runtime Stability Gates.
35. Do not rerun legacy 32-pass/budget-128 experiments.
36. One plausible failure mode requires one adequate check.
37. Do not create arbitrary hash manifests.
38. Keep documentation minimal.
39. Stop validating when the requested task is adequately proven.
40. The immediate objective is implementation and formal ObjectText label
    generation under the frozen v6 protocol.
```

---

# 65. Frozen End State

v6 is intended to be the frozen execution protocol.

The remaining work is implementation and experimentation inside this protocol.

Do not add new scientific branches unless actual evidence reveals a concrete protocol bug.

The first large-scale task after the one focused ObjectText API check is:

```text
formal Step 3 ObjectText label generation
```

using:

```text
official/paper-derived K=50

independent candidate reset/rollout

observation-only absolute ObjectText labels

scalar integer serialization of ObjectText observations

O0/Oz measured and cached once per program

frozen measurement/ratio/training validity semantics

completed candidate rollout required for candidate training labels

K=50 supervised target completeness = 50/50 valid candidates

candidate best post-pass state excluding initial state

common-cohort denominator rule for primary comparisons

failure-count reporting

adaptation/train + validation populations only
```

End of Guide.