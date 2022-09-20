---
theme: seriph
background: /img/green.png
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Efficient NLP, a Pipeline

  by [Bowen Shen](https://sbwww.github.io)
drawings:
  persist: false
title: Efficient NLP
---

# Efficient NLP

## a Pipeline

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/sbwww" target="_blank" alt="GitHub"
    class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<style>
h2 {
  background-color: #fff;
  text-align: center;
}
</style>

---

## Focus on Efficiency

|                                  |                                                                                                                                                                                                                                                                                                   |
| :------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![propotion](/img/propotion.png) | <table><tr><td></td><td>ACL</td><td>EMNLP</td><td>NAACL</td></tr><tr><td>2022</td><td>❌</td><td>✔</td><td>✔</td></tr><tr><td>2021</td><td>❌</td><td>✔</td><td>❌</td></tr><tr><td>2020</td><td>❌</td><td>❌</td><td>-</td></tr><tr><td>2019</td><td>❌</td><td>❌</td><td>❌</td></tr></table> |

<br>
<p style="text-align: center">Gaining more attention<br>(might be a main track in 2023)</p>

<style>
td:first-child {
  width: 60%;
}
img:hover {
  transform: none;
}
</style>

<!-- 在 19 年左右，NLP、CV、general ML 顶会对 efficiency 的关注度，Transformer 刚问世，大家都开始比大模型，总体上 NLP 对效率的关注晚一些
另外，从 EMNLP 这么一个 Emperical 的会也能看出一点趋势，ACL 则还没有这个 topic -->

---

## What is Efficiency?

> The term <font color='green'>Green AI</font>[^green] refers to AI research that yields novel results while taking into account the computational cost, **encouraging a reduction in resources spent**.

$$Cost(R) \propto E \cdot D \cdot H$$

- Cost of AI **_R_**esult is linear with
  1. Cost of processing an **_E_**xample → (model computation and tuning method)
  2. Size of training **_D_**ataset → (full data? convergence?)
  3. Number of **_H_**yperparameter experiments → (reproducibility)

[^green]: [Green AI, Communications of the ACM, 2020](https://dl.acm.org/doi/10.1145/3381831)

<!-- Green AI 是 20 年的文章，比较宏观地讲了对高效，也就是 green 的认识
文章提出一个关系，代价正比于这三个东西，分别是处理一个样本的代价、训练数据集大小、超参搜索次数
这说的比较宏观，转述一下就是，1.模型计算量和训练方法；2.全量还是小样本，这里总结的不太准确，因为一般小样本场景会有更复杂的训练方法，使 E 更大。也可以通过加快收敛减少迭代次数；3.可复现性，要不要一个很严苛的设置 -->

---

## How to be Efficient?

|                                                                                                                                                    |                            |
| :------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------: |
| ![survey_title](/img/survey_title.png)<br>Newly released on [Arxiv](https://arxiv.org/abs/2209.00099), August 31<br>![pipeline](/img/pipeline.png) | ![survey](/img/survey.png) |

<style>
table {
  margin-top: -6.0rem;
}
img {
  margin: 0 0 0 0;
}
td, tr {
  margin: 0 0 0 0 !important;
  padding: 0 0 0 0 !important;
}
td:first-child {
  width: 50%;
}
img {
  width: 90%;
}
img:hover {
  transform: none;
}
</style>

---

## How to be Efficient? --- Minimizing **_E_** and **_D_**

1. Reducing model computation [^turc] [^cofi]
   1. Which parts are more redundant?
   2. Reduce before or after training?
2. Efficient training [^learner]
   1. Which parameters need tuning?
   2. How to achieve efficient inference?

[^turc]: [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models<br>Google Research, ICLR 2020 rejected, 297 citations](https://arxiv.org/abs/1908.08962)
[^cofi]: [Structured Pruning Learns Compact and Accurate Models<br>Princeton U, ACL 2022](https://arxiv.org/abs/2204.00408)
[^learner]: [Efficient Fine-Tuning of Compressed Language Models with Learners<br>McGill U, ICML 2022 Workshop on Hardware Aware Efficient Training](https://arxiv.org/abs/2208.02070)

<!-- 实现高效的方法这边主要针对刚刚的第一点，也就是想办法减小这个 E 和 D
可以把减小的方法分成两种，一个是模型角度减少模型计算量，另一个是训练方法角度的高效训练
模型角度有两个关注点，一是哪些地方的冗余更冗余，比如宽度、深度；二是时间点的选择，也就是说，是在训练好的模型基础上减小体积，还是重新训一个小模型
 -->

<!-- <table>
  <tr>
    <td>
      <ol>
        <li>Reducing model computation [^turc] [^cofi]
        <ol>
          <li>Which parts are more redundant?</li>
          <li>Reduce before or after pre-training?</li>
        </ol></li>
        <li>Efficient training [^learner]
        <ol>
          <li>Which parameters need tuning?</li>
          <li>How to achieve efficient inference?</li>
        </ol></li>
      </ol>
    </td>
    <td>
      $$Cost(R) \propto E \cdot D \cdot H$$
    </td>
  </tr>
</table> -->

---

## Well-Read Students Learn Better: On the Importance of Pre-training Compact Models

> Just pre-training and fine-tuning compact models has been overlooked.

- Contributions
  1. Pre-trained Distillation (PD) is effective with **very little labeled data**.
  2. PD is **just as good** as more elaborate techniques that make restrictive assumptions about the model architecture.
  3. **Extensive** experiments of PD.

<br>

> Ratings: 3 (Weak Reject), 6 (Weak Accept), 1 (Reject)
>
> Importance and magnitude of the contribution are questioned, although it is important to share empirical results. &emsp;&emsp; --- from [OpenReview](https://openreview.net/forum?id=BJg7x1HFvB)

<!-- 文章主要特点就是实验很多，作者本人也说没有什么很新的东西，重在研究蒸馏方面被忽视的问题，但也因此被拒
文章提出一个 Pre-trained Distillation (PD) 方法来研究大小模型间的知识蒸馏，可以比较简单地达到更 fancy 和 restrictive 方法的效果 -->

---

## Pre-trained Distillation (PD)

![PD](/img/PD.png)

- Pre-train a small student from scratch, rather than truncating the teacher. This idea can also be found in TinyBERT.

<style>
img {
  width: 90%;
}
img:hover {
  transform: none;
}
</style>

<!-- 主要看一下和 BERT 预训练的区别，第一步做 MLM 的自监督 pre-train 是一样的，最后有标签的 fine-tune 也是一样的，就是中间多了一个 teacher 到 student 的蒸馏
跟其他的一些蒸馏工作相比，PD 则是多了第一步的直接 pre-train，一些之前的工作，比如 PKD，DistilBERT 都是将大模型截断作为 student，而 PD 就是直接拿一个随机初始化的小模型来 pre-train，这个思路在 TinyBERT 中也有所体现 -->

---

## Why Pre-training a Student?

- Is it **enough to pre-train word embeddings**?
  - **No**.
  - LM pre-training is necessary to unlock the full student potential.
- Is it **worse to truncate** deep pre-trained models?
  - **Yes**, especially for shallow students.
  - Initializing shallow-and-wide students from the bottom layers of their deeper pre-trained counterparts is suboptimal.
- What is the **best student** for a fixed parameter size budget?
  - **Prioritize depth over width**, especially with pre-trained students.

---

## Pre-training V.S. Truncating

|                                    |                                                              |
| :--------------------------------: | :----------------------------------------------------------- |
| ![PVT](/img/pretrain_truncate.png) | Pre-training $>$<br>Truncating $>$<br>Pre-training embedding |

<style>
td:first-child {
  width: 60%;
}
img {
  width: 65%;
}
img:hover {
  transform: none;
}
</style>

---

## Depth V.S. Width

![DVW](/img/24_speedup.png)

- Parameter size _generally_ $\propto L \cdot H^2$, runtime _generally_ $\propto L \cdot H$
  - MHA, FFN matrix size $\propto H^2$
  - Embedding matrix size $\propto H$
- Squat (shallow-and-wide) models have **shorter runtime** than slender (deep-and-narrow) ones.

<style>
img {
  width: 90%;
}
img:hover {
  transform: none;
}
</style>

---

## Depth Outweighs Width

|                              |                                                                                                                                                                |
| :--------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![DVW](/img/depth_width.png) | <ul><li>Deep model is better than shallow model.</li><li>6-12 Layers have similar performance. (<font color="red">task-specific, not always!</font>)</li></ul> |

<style>
td:first-child {
  width: 60%;
}
img {
  width: 90%;
}
img:hover {
  transform: rotate(90deg);
}
</style>

<!-- 在 SST-2 实验看起来 6-12 层效果差不多，但这不一定准确，在做提前退出的时候就发现 SST-2 并不需要深层模型的能力，2-4 层即可做出不错的效果，而对于 NLI 任务则是深层模型效果好很多 -->

---

## Under the Hood: Dissecting Pre-trained Distillation

![robust](/img/robustness.png)

<style>
img {
  width: 75%;
}
img:hover {
  transform: none;
}
</style>

---

## Compound Effects

|                               |                                                                                                                                                                              |
| :---------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![compound](/img/compound.png) | Pre-training + Distillation $>$<br>Pre-training + Fine-tuning $>$<br>Vanilla Distillation<br><br>($\mathcal{D}_{LM}=\mathcal{D}_{T}$ unlabeled<br>$\mathcal{D}_{L}$ labeled) |

<style>
td:first-child {
  width: 60%;
}
img {
  width: 85%;
}
img:hover {
  transform: none;
}
</style>

<!-- NLI* 是从 SNLI MNLI QQP 3 个数据集构造的无标签 LM 数据集，这样在无标签数据集上 pre-train + distill 的小模型仍然能比 pre-train + 下游任务精调的小模型 -->

---

## Structured Pruning Learns Compact and Accurate Models

> Distillation from scratch is costly.

Contributions:

- Structured pruning can obtain **large speedups** and **competitive accuracy** as distillation approaches, while requiring **much less computation**.
  - Propose a flexible granularity CoFi (Coarse and Fine-grained) Pruning for **large speedup**.
  - Propose a layer-wise distillation for unknown student architecture and **competitive accuracy**.
  - Does not include general distillation and additional unlabeled data for **much less computation**.

---

## CoFi Pruning

![cofi](/img/cofi.png)

- Start off from a fine-tuned model, do not need general distillation.
- CoFi is efficient and flexible.

<style>
img {
  width: 85%;
}
img:hover {
  transform: none;
}
</style>

---

## Pruning Granularity

1. Hidden dimensions
   - apply mask $\mathbf{z}_{\text{hidn}} \in \{0, 1\}^{d}$ to every matrix, $\mathbf{z}_{\text{hidn}}$ is shared across layers.
   - Only a small number of dimensions are pruned (e.g., 768 → 760).
2. FFN intermediate dimensions
   - $\operatorname{FFN}(X) \gets \operatorname{gelu}\left(X W_U\right) \cdot \operatorname{diag}\left(\mathbf{z}_{i n t}\right) \cdot W_D$, mask $\mathbf{z}_{\text{int}} \in \{0, 1\}^{d_f}$
3. Attention head
   - $\operatorname{MHA}(X) \gets \sum_{i=1}^{N_h} \mathbf{z}_{\text{head}}^{(i)} \operatorname{Att}\left(W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}, W_O^{(i)}, X\right)$, mask $\mathbf{z}_{\text{head}}^{(i)} \in \{0, 1\}$
4. Entire layer
   - $\operatorname{MHA}(X) \gets z_{MHA}\operatorname{MHA}(X)$, $\operatorname{FFN}(X) \gets z_{FFN}\operatorname{FFN}(X)$
   - 50% of layers can be dropped without a big accuracy drop

<br>

> Masks are trained as real numbers in \[0, 1\], then mapped to 0 or 1 at inference.

---

## Distillation for Unknown Student Architecture

Dynamically search a layer mapping between the full teacher model and the pruned student model.

mapping function: $m(i)=\underset{j: \mathbf{z}_{\mathrm{FFN}}^{(j)}>0}{\arg \min \operatorname{MSE}}\left(W_{\text {layer}} \mathbf{H}_s^j, \mathbf{H}_t^i\right)$

distillation loss: $\mathcal{L}_{\text {layer}}=\sum\limits_{i \in \mathcal{T}} \operatorname{MSE}\left(W_{\text {layer}} \mathbf{H}_s^{m(i)}, \mathbf{H}_t^i\right)$

**Layer mismatch** happens on small datasets, e.g., RTE, MRPC. So, a constraint is added to only match a lower student layer than the previously matched student layer.

<!-- 为教师模型的每一层，找到学生模型中最相似，且 FFN 没被剪掉的一层来蒸馏 -->

---

## CoFi is Pareto Optimal

![cofi_result](/img/cofi_result.png)

Yet another compressed model exceeding uncompressed BERT performance.

<style>
img {
  width: 75%;
}
img:hover {
  transform: none;
}
</style>

---

## Compare to TinyBERT

|                                         |                                |
| :-------------------------------------: | :----------------------------: |
| ![cofi_tinybert](/img/cofi_tinybert.png) | ![cofi_aug](/img/cofi_aug.png) |

- TinyBERT relies on General Distillation and Data Augmentation, which are extremely costly.
- CoFi is efficient and better (better than my EMNLP submission in some cases).
- ⚠ <font color="red">The comparison may be unfair.</font> Embedding is not taken into account, but TinyBERT$_4$ has a much smaller embedding than CoFi.

<style>
img {
  width: 100%;
}
</style>

---

## Ablation Studies

<table>
  <tr>
    <td colspan=2><img src="/img/cofi_abl_1.png" alt="cofi_abl_1"></td>
  </tr>
  <tr>
    <td><img src="/img/cofi_abl_2.png" alt="cofi_abl_2"></td>
    <td>Task-specific layer alignment:<ul>
      <li>SST-2: 7, 9, 10, 11 layers of student to<br>&emsp;&emsp;&emsp;&nbsp;3, 6, 9, 12 layers of teacher</li>
      <li>QQP: 2, 5, 8, 11 layers of student to<br>&emsp;&emsp;&ensp;&nbsp;same layers of teacher</li>
    </ul></td>
  </tr>
</table>

<style>
td:first-child {
  width: 45%;
}
img {
  width: 75%;
}
</style>

---

## Remained Structures

|                                           |                            |
| :---------------------------------------: | :------------------------: |
| ![average_remain](/img/average_remain.png) | ![remain](/img/remain.png) |

- At **medium** sparsity, **deep** FFNs and MHAs are pruned.
- At **high** sparsity, **medium and deep** FFNs and MHAs are pruned.
- **FFNs** are **more frequently pruned** than MHAs.

<style>
td:first-child {
  width: 47%;
}
img {
  width: 65%;
}
</style>

<!-- 右图是 95% 稀疏度 -->

---

## Other Interesting Findings

|                                        |                                            |
| :------------------------------------: | :----------------------------------------: |
| ![cofi_roberta](/img/cofi_roberta.png) | ![tinybert_issue](/img/tinybert_issue.png) |
|   BERT $>$ RoBERTa at high sparsity    |     Reproducibility issue of TinyBERT      |

<style>
td:first-child {
  width: 47%;
}
img {
  width: 95%;
}
img:hover {
  transform: none;
}
</style>

<!-- RoBERTa 差一些可能因为，这种 fancy 一点的模型对结构完整性的要求更强一点，硬要剪掉太多内容，性能下降就会更大 -->

---

## Efficient Fine-Tuning of Compressed Language Models with Learners

> Prototype: Freeze-and-Reconfigure (FAR)[^far]: freezing subsets of FFN parameters that do not learn quickly after priming steps (initial 10% steps).

Contributions:

- Ensuring **quick convergence** with few tunable parameters.
- Parameter-efficient methods can train **compressed models** as well as big models.
- Parameter-efficient tuning can have **no additional overhead** at inference.

[^far]: [Efficient Fine-Tuning of BERT Models on the Edge<br>McGill U, IEEE International Symposium on Circuits and Systems 2022](https://arxiv.org/abs/2205.01541)

---

## Learner Structure

![learner](/img/learner.png)

<style>
img:hover {
  transform: none;
}
</style>

---

## Deja Vu

|                                       |                                     |
| :-----------------------------------: | :---------------------------------: |
|     ![learner](/img/learner2.png)     |       ![lora](/img/LoRA.png)        |
| Learner<br>focus on compressed models | LoRA [^lora]<br>focus on big models |

[^lora]: [LoRA: Low-Rank Adaptation of Large Language Models<br>Microsoft, ICLR 2022](https://arxiv.org/abs/2106.09685)

<style>
td:first-child {
  width: 45%;
}
img {
  width: 55%;
}
img:hover {
  transform: none;
}
</style>

---

## Learner Structure

![learner](/img/learner.png)

Learner $= \bm{P}_1^{H \times H^\prime} \times \bm{P}_2^{H^\prime \times H}$ with no activation → can be collapsed after tuning, unlike adapters, which are permanently fixed to the model

$$\bm{x}\bm{W}+b+\bm{x}\bm{P}_1\bm{P}_2=\bm{x}(\bm{W}+\bm{P}_1\bm{P}_2)+b$$

$H^\prime$ is the low rank, same as the intrinsic dimension of LoRA

<!-- ⚠ <font color='red'>The structure may not be sound, $\bm{P}_1 \bm{P}_2$ **IS ONE** matrix</font> ⚠ -->

<style>
img:hover {
  transform: none;
}
</style>

<!-- 有点怪，因为本质上这样的 P1 P2 就是一个矩阵，文章这样拆开感觉纯粹是为了减小 tunable parameters 的数值，
不确定在理论上是不是 technically sound，多学点理论 -->

---

## Priming

|                              |                                                             |
| :--------------------------: | :---------------------------------------------------------: |
| ![priming](/img/priming.png) | <ol><li>Train MHA + Learner</li><li>Train Learner</li></ol> |
|             FAR              |                           Learner                           |

- Priming was interesting in FAR, but very different and trivial in Learner.
- **Can we use priming to select the intrinsic dimension?**

<style>
td:first-child {
  width: 60%;
}
img {
  width: 75%;
}
img:hover {
  transform: none;
}
</style>

<!-- 本文的 Priming 预热，是先把 自注意力和 Learner 一起训练一会，然后再单独训练 Learner，和之前工作不一样 -->

---

## Advantages in Convergence and Parameters

|                                                |                                                                                                                                                                                                                                                                                                                                                                                  |
| :--------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![learner_converge](/img/learner_converge.png) | <table><tr><td></td><td>Tunable Params (M)</td></tr><tr><td>DistilBERT</td><td>66.9</td></tr><tr><td>FAR<sub>10</sub></td><td>41.4</td></tr><tr><td>Freeze FFN</td><td>14.8</td></tr><tr><td>Learner<sub>64</sub>(p=2)[^lx]</td><td>5.9</td></tr><tr><td>Parallel Adapter</td><td>5.3</td></tr><tr><td>Adapter</td><td>5.3</td></tr><tr><td>BitFit</td><td>0.6</td></tr></table> |

- Fast convergence, few parameters

[^lx]: Learner$_{H^\prime}(p=x)$: hidden size $H^\prime$ of P1 and P2, number of priming epochs $x$.

<style>
td:first-child {
  width: 50%;
}
img {
  width: 95%;
}
img:hover {
  transform: none;
}
</style>

---

## Full Results

![learner_result](/img/learner_result.png)

- Learner w/ priming $\approx$ Fine-tune $>$ Adapter
- w/ priming $>$ w/o priming
- Experiments are done within 5 epochs, indicating the fast convergence of Learner.

<!-- $$(H H^\prime + H^\prime H) 4 L + (H H^\prime + H^\prime F) 2 L + (H H^\prime + H^\prime H) + HC \\
\overset{F=4H}{=} (18L+2) H H^\prime + HC$$ -->

<style>
img:hover {
  transform: none;
}
</style>

---

## Further Thoughts

We can learn that

- There is a collapsible design of low rank training (same as LoRA)
- Compressed models can be trained efficiently as well as big models

However, I don't know

- **WHY** low-rank structure works?
  - (see also LoRA paper Section 7, but I still dunno, my bad)
- What is the **general paradigm** of efficient tuning on compressed models and the **difference** with big models?
- Can we use priming of FAR to select the rank (intrinsic dimension)?

<!-- 为什么低秩结构很好，这一点我还是没有特别理解，尽管 LoRA 的文章做了一些 empirical studies 来证明。但是对这种参数更新这可能是因为理论学得太浅 -->

---

## Recap

$$Cost(R) \propto E \cdot D \cdot H$$

1. Train a small model from scratch \.\.\.\.\.\.\.\.\.\.\.\. <font color="green">$E \searrow$</font>
   1. Costly training \.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\. <font color="red">$D \nearrow$</font>
   2. Simple and intuitive \.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\. <font color="green">$H \searrow \ ?$</font>
2. Derive a small model from big model \.\.\.\.\.\. <font color="green">$E \searrow$</font>
   1. Inexpensive training \.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\. <font color="green">$D \searrow$</font>
   2. Require fancy technics \.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\. <font color="red">$H \nearrow \ ?$</font>
3. Efficient Training \.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\. <font color="green">$E \searrow \ D \searrow$</font>
   1. Good on big models
   2. Applicable on small models
