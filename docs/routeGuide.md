可以。既然已经确定 **LLVM Phase Ordering**，我建议这次从一开始就按“**先跑通、再扩数据、最后部署**”设计，避免重新陷入 ExReal 那种十几个版本后才发现目标/数据有问题的情况。

核心方案我建议固定为：

> **MambaPO：Mamba-guided LLVM Phase Ordering**
>
> Mamba 建模“程序状态 + 已执行优化 pass 序列”的长期交互，输出每个候选下一 pass 的价值；再用 beam search / best-first search 找优化序列，最终目标是 **unseen program runtime 优于 LLVM `-O3`**。

CompilerGym 已经提供 LLVM phase-ordering 环境、大规模 benchmark、offline state-transition 数据以及 runtime 支持；其 LLVM 环境允许 pass 重复或省略，episode 本身也没有固定长度，天然适合序列建模。

---

# 一、先把整个项目路线锁死

这次不要再频繁换结构。

```text
LLVM IR program
     ↓
当前 IR features
+
已经执行的 pass history
+
历史 reward / state change
     ↓
Mamba
     ↓
Q(s, pass) / future value
     ↓
Beam Search
     ↓
pass sequence
     ↓
LLVM
     ↓
binary
     ↓
runtime
```

**Mamba 不做唯一 next-pass imitation。**

因为 phase ordering 和你之前 instruction scheduling 一样存在多解：

```text
A→B→C
A→C→B
D→A→C
```

可能性能都差不多。

所以监督目标固定成：

> **这个 partial pass sequence 后面还能得到多好的性能**

而不是：

> “下一步唯一正确 pass 是什么”。

这样从架构上绕开你之前 teacher trajectory ambiguity 的坑。

---

# 二、建议使用的数据

## 数据 A：CompilerGym State Transition Dataset —— 主预训练数据

CompilerGym v0.2.1 官方发布了 LLVM **State Transition Dataset**，定义就是大量：

```text
(state, action, reward)
```

数据，并且官方专门提供了 GNN cost-model 监督学习示例。

下载/入口：

```text
https://github.com/facebookresearch/CompilerGym/releases/tag/v0.2.1
```

官方 cost model 示例：

```text
https://github.com/facebookresearch/CompilerGym/tree/development/examples/gnn_cost_model
```

数据读取代码：

```text
https://github.com/facebookresearch/CompilerGym/blob/development/examples/gnn_cost_model/compiler_gym_dataset.py
```

官方 loader 显示数据以数据库形式保存 `States` / `Observations`，可以直接用于监督学习。

### 用途

不要直接拿它作为最终 runtime 标签。

主要用于：

```text
LLVM IR representation pretraining
pass embedding pretraining
单步 state transition effect
reward/value warm start
```

然后再用真正的 trajectory + runtime 数据微调。

---

# 三、CompilerGym 环境

我建议固定：

> **CompilerGym 0.2.5 + LLVM 10**

不要追新版 LLVM。

原因是 CompilerGym 仓库已经在 **2026-05-27** 归档，只读；最后正式 release 是 `v0.2.5`，这个版本增加了 Jotaibench，并明确支持 Python 3.10。

项目：

```text
https://github.com/facebookresearch/CompilerGym
```

Release：

```text
https://github.com/facebookresearch/CompilerGym/releases/tag/v0.2.5
```

官方安装说明：

```text
https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md
```

CompilerGym 官方安装支持：

```bash
pip install -U compiler_gym
```

源码构建则要求 LLVM **10.0.0**。

为了复现性，我建议：

```bash
conda create -n mambapo python=3.10 -y
conda activate mambapo

pip install compiler_gym==0.2.5
```

然后立即检查：

```bash
python - <<'PY'
import compiler_gym
print(compiler_gym.__version__)

env = compiler_gym.make("llvm-v0")
print(env.action_space)
print(env.observation.spaces.keys())
print(env.reward.spaces.keys())
PY
```

**不要在代码里硬编码 pass 数量。**

直接：

```python
n_actions = env.action_space.n
```

因为后面所有模型都从环境自身读取 action space。

---

# 四、主训练 benchmark

我建议数据分成三层。

## 1. 大规模 representation pretrain

使用 CompilerGym 自带：

```text
Jotaibench
AnghaBench
Csmith
state-transition dataset
```

CompilerGym v0.2.5 的 Jotaibench 有 **18,761 个可执行 C 程序**。CompilerGym早期版本还加入了 AnghaBench、大规模 C 函数数据以及 Csmith generator。

这些主要用来训练：

```text
program representation
pass interaction
IR state transition
```

不用全部跑昂贵 runtime。

---

## 2. Runtime training：Csmith

CompilerGym已经提供过 Csmith runtime optimization 支持；其 runtime 工作最初就是围绕 `cbench-v1` 和 `csmith-v0` 开发的。

这是非常有用的一点。

因为你可以不断生成：

```text
generator://csmith-v0/...
```

获得大量**训练程序**。

也就是说：

```text
Csmith programs
→ random / greedy / beam pass sequence
→ compile
→ execute
→ runtime
```

生成真正：

```text
program
+ pass history
+ runtime
```

的数据。

这比你之前手工收集几百个 LLVM region 健康得多。

---

# 五、最终 Test：cBench

这个最好作为**最终冻结 runtime test**。

CompilerGym 后期已经支持 cBench 的全部 20 套 runtime datasets。

原始 cBench：

```text
https://sourceforge.net/projects/cbenchmark/files/cBench/V1.1/
```

cBench 本身就是用于编译器/架构优化研究的开源程序集合。

建议：

```text
Train:
Csmith
+ large CompilerGym corpora

Dev:
PolyBench

Test:
cBench
```

这样不会把 cBench 调烂。

---

# 六、Dev：PolyBench/C

PolyBench/C 有 30 个数值计算 kernel，而且运行 harness 非常统一，特别适合 phase-ordering runtime development。

官方页面：

```text
https://www.cs.colostate.edu/~pouchet/software/polybench/
```

下载：

```text
https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
```

用途：

```text
开发阶段 runtime
模型选择
beam width选择
sequence length选择
```

最终不要用 cBench 选超参数。

---

# 七、AutoPhase 数据：辅助 baseline

AutoPhase 代码和数据都还公开。

项目：

```text
https://github.com/ucb-bar/autophase
```

数据：

```text
https://github.com/ucb-bar/autophase/tree/master/dataset
```

其中直接包含：

```text
train_chstone_act.pkl
train_chstone_pgm.pkl
train_rand.pkl
train_rand_act.pkl
```

官方仓库可以直接看到这些文件。

但注意：

> AutoPhase 面向 HLS + LegUp，不要混进最终 CPU runtime test。

把它用于：

```text
数据格式参考
PPO baseline参考
pass-history baseline
MLP/RL baseline
```

即可。

AutoPhase 的 HLS 实验相对 `-O3` 报告 16% circuit-performance improvement；后续版本报告 28%，说明 phase ordering 有很大的优化空间。

---

# 八、Mamba 环境

官方实现：

```text
https://github.com/state-spaces/mamba
```

论文：

```text
https://arxiv.org/abs/2312.00752
```

Mamba 具有随序列长度线性扩展的 selective SSM，并且官方代码现在提供 Mamba、Mamba-2、Mamba-3。

最快先用 **Mamba-2**。

安装：

```bash
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
```

GPU selective scan：

```bash
MAMBA_KEEP_CUDA_BUILD=TRUE \
pip install mamba-ssm --no-build-isolation
```

这是当前官方推荐的安装方式。

---

# 九、第一版 MambaPO 模型

第一版千万别做大。

建议：

```text
Program state:
Autophase features / InstCount
        ↓
Linear
        ↓
state embedding

previous pass
        ↓
Embedding

reward delta
        ↓
Linear

[state, pass, reward]
        ↓
每一步一个 token
        ↓
4 × Mamba-2
        ↓
last hidden state
        ↓
Q head
        ↓
每个 LLVM pass 一个 value
```

推荐初始大小：

```text
d_model = 256
layers = 4
d_state = 64
d_conv = 4
expand = 2
```

这样模型只有几百万参数级别，不需要 100M 模型。

---

# 十、输入 token

每一步：

```text
x_t =
IR state embedding
+
previous pass embedding
+
reward delta embedding
+
step embedding
```

例如：

```text
t0:
initial IR
START
0

t1:
IR after mem2reg
mem2reg
Δreward1

t2:
IR after instcombine
instcombine
Δreward2

...
```

Mamba看到完整：

```text
pass1
→ IR change
→ pass2
→ IR change
→ pass3
...
```

这才是真正利用 Mamba 长历史建模。

---

# 十一、训练目标

不要 next-pass classification。

使用：

## Q-value / Return-to-go regression

对于 trajectory：

```text
s0,a0,r0
s1,a1,r1
...
sT
```

定义：

```text
G_t = 最终 performance - 当前 performance
```

训练：

```text
Q(s_t, a_t) → G_t
```

如果多个 pass 最终都能得到相同性能：

```text
A → +8%
B → +8%
C → +2%
```

模型自然学：

```text
Q(A) ≈ Q(B) > Q(C)
```

不用规定：

```text
A 正确
B 错误
```

这直接解决你之前 instruction scheduler 的多解监督问题。

---

# 十二、搜索算法

第一版用简单 **Beam Search**。

比如：

```text
beam width = 8
top-k actions = 8
max passes = 32
```

流程：

```text
initial program
      ↓
Mamba Q
      ↓
top 8 passes
      ↓
执行 LLVM pass
      ↓
得到8个新 IR
      ↓
Mamba重新评分
      ↓
保留 best 8
      ↓
重复
```

最终：

```text
best sequence
```

然后编译、运行。

这比纯 Mamba policy安全：

> Mamba 是 heuristic，不是绝对决策者。

---

# 十三、最快的第一轮实验

这次不要一开始就上百万数据。

先跑一个 **Smoke Experiment**：

### 数据

```text
Train:
1000 Csmith programs

Dev:
10 PolyBench kernels

Test:
暂不开 cBench
```

### Sequence

```text
length ≤ 16
```

### 搜索

```text
random
greedy
Mamba beam
```

### 模型

```text
2~4 Mamba blocks
d_model 128/256
```

第一轮只回答：

> **Mamba 是否能在相同 search budget 下超过 random / greedy？**

如果这都做不到，立即查数据/target。

不要再直接跑十几个模型。

---

# 十四、正式实验阶段

Smoke PASS 后才扩：

```text
sequence length:
16 / 32 / 64

beam:
4 / 8 / 16

train programs:
Csmith + transition dataset + large LLVM corpora
```

最终 baseline 建议固定：

| 类别 | Baseline |
|---|---|
| LLVM | `-O3` |
| 搜索 | Random Search |
| 搜索 | Greedy |
| 搜索 | Beam Search |
| 学习 | MLP |
| 学习 | LSTM |
| 学习 | Transformer |
| RL | PPO / AutoPhase-style |
| **Ours** | **Mamba-guided Beam** |

CompilerGym 官方论文本身就比较了多种 autotuning/RL方法，并提供 LLVM autotuning、RL、GNN cost-model 示例，因此这些 baseline 有明确出处。

---

# 十五、正式评价指标

论文最重要的一定是：

## Primary

```text
Runtime speedup over LLVM -O3
```

定义：

\[
Speedup=\frac{T_{O3}}{T_{MambaPO}}
\]

最终报告：

```text
cBench geometric mean runtime speedup
```

---

## 第二指标

同样 compilation/search budget 下：

```text
best runtime
```

这是证明 Mamba 真正价值的核心。

例如：

```text
32 compilations budget

Random       +1.8%
Greedy       +2.3%
Transformer  +3.0%
Mamba        +4.1%
```

这种结果会比：

> Mamba 准确率 84%

强得多。

---

## 还要报告

```text
runtime speedup
search samples
compile time
model inference latency
pass sequence length
binary size
```

---

# 十六、数据划分一定这样做

以前 ExReal 已经吃过这个亏。

**按 program 分。**

绝对不要：

```text
同一个 program 的 state
一半 Train
一半 Test
```

必须：

```text
Train programs
∩
Dev programs
∩
Test programs
=
∅
```

正式：

```text
Train:
Csmith / Jotaibench / AnghaBench

Dev:
PolyBench

Test:
cBench
```

Test 打开以后不再调：

```text
beam width
sequence length
learning rate
model size
```

---

# 十七、Runtime 测量

这次一定从一开始就规范。

每个 binary：

```text
warmup
+
多次正式执行
```

取：

```text
median runtime
```

并：

```text
CPU affinity固定
关闭/控制频率波动
相同输入
相同编译选项
相同机器
```

cBench 的好处之一就是 CompilerGym已经自带执行和 runtime infrastructure。

---

# 十八、部署方式

第一阶段**不要改 LLVM 源码**。

最快：

```text
mambapo-opt program.bc
```

内部：

```text
Python
↓
CompilerGym/LLVM
↓
Mamba
↓
beam search
↓
best pass sequence
↓
opt
↓
clang
```

最终用户体验：

```bash
mambapo input.c -O3 -o output
```

内部其实：

```text
clang -emit-llvm
→ MambaPO
→ optimized.bc
→ clang backend
```

这个已经算真实部署。

---

## 后期再考虑 LLVM 内嵌

等论文结果成立，再做：

```text
clang
↓
Mamba Phase Planner
↓
生成 pass pipeline
↓
LLVM new/legacy PM
```

phase ordering 本身 search/compilation 就远比几毫秒模型推理贵，所以这次**不用像 MachineScheduler 那样死磕 4 ms CPU inference**。

---

# 十九、明确的 Gate

这次我建议整个项目最多设 4 个 Gate。

### Gate 0：环境

```text
CompilerGym可以正常：
reset
apply pass
compile
run
measure runtime
```

---

### Gate 1：Search headroom

不用 Mamba。

Random/beam 在 PolyBench 上：

```text
明显存在 > -O3 的 sequence
```

如果没有：

> 先查环境，禁止训练。

---

### Gate 2：Mamba learning

相同 trajectory 数据：

```text
Mamba > MLP
Mamba ≥ LSTM
```

在 held-out program value prediction / ranking 上成立。

否则：

> Mamba 架构没有贡献。

---

### Gate 3：最终

```text
Mamba-guided search
>
Random/Greedy
```

且：

```text
runtime < LLVM -O3
```

在 unseen programs 成立。

只有这一步才叫成功。

---

# 二十、最重要参考文献

### CompilerGym

**CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research**

```text
https://arxiv.org/abs/2109.08267
```

提供 LLVM phase ordering、百万级 benchmark、offline performance data、runtime/code-size environments。

---

### AutoPhase

**AutoPhase: Compiler Phase-Ordering for High Level Synthesis with Deep Reinforcement Learning**

```text
https://arxiv.org/abs/1901.04615
``` 


后续：

```text
https://arxiv.org/abs/2003.00671
``` 


---

### POSET-RL

**Phase ordering for Optimizing Size and Execution Time using Reinforcement Learning**

```text
https://arxiv.org/abs/2208.04238
```

在 x86/AArch64、SPEC/MiBench 上研究 phase ordering 的执行时间与 code size。

---

### ACPO

**AI-Enabled Compiler-Driven Program Optimization**

```text
https://arxiv.org/abs/2312.09982
```

其 LLVM ML framework 在 PolyBench、cBench 上报告相对 `-O3` 的 runtime 改善。

---

### Protean Compiler（2026，很值得读）

```text
https://arxiv.org/abs/2602.06142
```

它做 LLVM fine-grained phase ordering，报告 cBench 平均最高约 **4.1%**、部分程序可达到双位数 speedup，是非常直接的最新可行性证据。

---

### Mamba

```text
https://arxiv.org/abs/2312.00752
https://github.com/state-spaces/mamba
``` 


---

# 二十一、我建议现在马上执行的版本

不要先写论文，不要先做复杂 RL。

直接做：

```text
MambaPO-v0

CompilerGym 0.2.5
LLVM 10
↓
Autophase/InstCount state
+ pass history
↓
Mamba-2
↓
Q-value head
↓
beam-8
sequence≤16
↓
PolyBench
```

先比较：

```text
LLVM -O3
Random
Greedy
MambaPO
```

**先证明 MambaPO 在相同搜索预算下能找到更快的 sequence。**

这个结果一旦出来，再扩大 Csmith/CompilerGym 数据、加 LSTM/Transformer/PPO baseline、最后封 cBench test。

我建议新项目目录也直接和原 ExReal 隔离：

```text
/root/MambaPO
```

不要继续往之前 instruction-scheduler 的代码里堆东西。旧项目只复用你的实验规范、统计/benchmark经验，不复用模型和数据链路。这样最快也最干净。