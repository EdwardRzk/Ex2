# Candidate-ranking error decomposition

## Evidence-backed conclusion

Dominant observed modes are **top-of-list ranking error**, **dataset-specific ranking/policy interaction**, and **residual-correction instability**. Candidate length is not a material primary explanation, and soft-target CE is only weakly aligned with policy45 regret. This is a descriptive diagnosis, not authorization for a new model.

- Frozen final dataset-macro MeanOverOz exactly reproduces NVP `0.08715469`, Mamba `0.08462666`, Direct `0.08765961`, and Anchored `0.08778865`. Validation instead has Mamba `0.06355292` versus NVP `0.06277101`. Mamba validation is not explained by uniformly better ranking: program-micro top-1/top-5 coverage is lower (`0.6299/0.8422` versus `0.6527/0.8651`) and mean regret higher (`9.50` versus `7.79` bytes). The validation macro win is dominated by LLVM-Stress (`+0.01890`), despite losses on CSmith, BLAS, Linux, and GitHub. On final LLVM-Stress reverses to `-0.01485`, alongside CSmith `-0.01433` and BLAS `-0.00595`.

- Final overall, Mamba has lower top-1/top-5 oracle coverage (`0.6183/0.8282` versus NVP `0.6424/0.8490`), worse mean oracle rank (`4.413` versus `4.085`), lower Spearman/Kendall (`0.382/0.301` versus `0.424/0.338`), and higher policy45 regret (`13.446` versus `12.023` bytes). NVP and Mamba agree at top-1 only `42.72%` of seed-program pairs; NVP selects a strictly better true-value candidate on `1,613` pairs versus Mamba on `1,022`.

- CSmith and BLAS are direct ranking failures. On CSmith Mamba oracle rank rises `3.93 -> 7.16`, top-5 coverage falls `0.8125 -> 0.6111`, and regret rises `35.1 -> 165.1` bytes. On BLAS its median oracle rank is `16` versus NVP `6`, top-5 coverage `0.1609` versus `0.4368`, and regret `66.8` versus `28.7` bytes. Neither has a long-sequence mechanism: Mamba admits more candidates with shorter average admitted lengths on both sources.

- LLVM-Stress and OpenCV show why rank summaries alone are insufficient. LLVM-Stress Mamba top-5 coverage is slightly higher (`0.9342` versus `0.9274`) with nearly identical oracle rank, but its global rank correlation collapses (`0.455` versus `0.652`) and its normalized policy result worsens. OpenCV Mamba improves top-1/top-5 coverage and oracle rank, yet policy45 regret rises (`49.9` versus `45.7` bytes) and MeanOverOz falls. Thus ordering among candidates admitted under the 45-pass budget matters beyond oracle coverage.

- Mamba-favored final sources are heterogeneous: CHStone improves top-5 coverage (`0.3889` versus `0.2222`) and utility; CLgen improves top-1/top-5 (`0.7089/0.8956` versus `0.6889/0.8489`); NPB and POJ-104 improve policy regret despite flat or worse coverage/rank. There is no single broad representation effect across sources.

- Length contributes only a small global shift: Mamba top-ranked candidates are `0.142` passes longer on average, admits `0.064` fewer candidates, and has `0.157` longer admitted candidates. The sign reverses on the largest failures (LLVM-Stress and BLAS), so length/budget is not the dominant cause.

- CE does not reliably select policy quality. Across final programs CE vs policy45 regret Spearman is only `0.058` for NVP and `0.088` for Mamba, while CE vs oracle rank is approximately zero (`-0.019`, `-0.002`). Yet CE is lower for NVP than Mamba (`3.6964` versus `3.7130`) on final and also lower on validation while Mamba wins validation macro.

- Direct and Anchored largely preserve NVP: final top-1 is unchanged in `85.78%`/`89.21%` of program-seed pairs, while beneficial changes (`1.72%`/`1.61%`) modestly exceed harmful ones (`1.40%`/`1.17%`). Their gains are mostly policy-order/admission corrections, not wholesale top-1 replacement. On LLVM-Stress Anchored changes no top-1 decisions but changes the admitted set on `31.97%`; its MeanOverOz improves strongly even though byte regret is slightly worse, demonstrating that byte regret is not itself the normalized final objective. Anchored loses CHStone despite better mean oracle rank because regret rises, and loses NPB despite a small byte-regret reduction; the residual correction remains source-dependent.

## Direction

The evidence supports investigating a **policy-aware ranking/loss objective with explicit robustness constraints on top-of-list/admission behavior**, not adding generic representations or a length bias fix. Any future proposal would require a separately authorized frozen protocol; none is trained here.
