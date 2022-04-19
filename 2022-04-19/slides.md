---
theme: seriph
background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Efficient PLMs
  from the perspective of Token

  by [Bowen Shen](https://sbwww.github.io)
drawings:
  persist: false
title: Efficient PLMs
---

# Efficient PLMs

## from the perspective of Token

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/sbwww" target="_blank" alt="GitHub"
    class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<style>
h2 {
  background-color: #fff;
}
</style>

---

## Increasingly Verbose

| ![verbose](/img/verbose.jpg#w50) | The sequence is TOOOOO long!<br><br>Where should I pay <font color='red'>attention</font> to?<br><br>I'm <font color='red'>OUT OF MEMORY</font>!<br><br>![bert-confused](/img/bert-confused.jpg#w60) |
| :------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |

<style>
table {
  margin-top: -20px;
  font-size: 22px;
}
th:first-child {
  width: 55%;
}
img:hover {
  transform: none;
}
</style>

---

## Attention is All We Need?

![skimming](/img/skimming.jpg)

Attention is good, but not effective for long docs, $O(n^2)$ is not efficient

Rethinking: Are all the tokens (words) necessary? Not really!

<style>
p {
  text-align: center;
}
img {
  width: 60%;
}
img:hover {
  transform: none;
}
</style>

---

## How?

1. Skimming [^skim]
2. Token pruning [^ltp]
3. Token pruning + early exiting [^mp]
4. Efficient Attention

[^skim]: [Block-Skim: Efficient Question Answering for Transformer<br>[AAAI 2022] SJTU, University of Rochester](https://arxiv.org/abs/2112.08560)
[^ltp]: [Learned Token Pruning for Transformers<br>[arXiv 2021] UC Berkeley, Samsung](https://arxiv.org/abs/2107.00910)
[^mp]: [Magic Pyramid: Accelerating Inference with Early Exiting and Token Pruning<br>[NIPS Workshop 2021] Monash University, Amazon](https://arxiv.org/abs/2111.00230)

<style>
li {
  line-height: 1.5em !important;
}
</style>

---

## Method 1 --- Block-skim [^skim]

![skim-eg](/img/skim-eg.png)

- Attention map is effective for locating the answer position
- Use attention to predict what blocks to skim (and to keep)

[^skim]: [Block-Skim: Efficient Question Answering for Transformer<br>[AAAI 2022] SJTU, University of Rochester](https://arxiv.org/abs/2112.08560)

<style>
img {
  width: 45%;
}
sup {
  background-color: #1f85ad;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  background-clip: text;
  clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

## QA Context is Lengthy

![squad-length](/img/squad-length.png)

But most context are unnecessary!

<style>
img {
  width: 55%;
}
img:hover {
  transform: none;
}
</style>

---

## Block-skim Architecture

![skim](/img/skim.png)

- Use CNN on attention map (`Softmax(QK)`) to calculate block importance
- Do not change Transformer architecture

<style>
img {
  margin-top: -3%;
  width: 100%;
}
img:hover {
  transform: none;
}
</style>

---

## CNN Based Block Relevance Prediction

![skim-cnn](/img/skim-cnn.png#w60)

Hypothesis: **diagonal region** of attention map contains sufficient information to identify the block relevance.

![skim-cnn-acc](/img/skim-cnn-acc.png#w40)

Rethinking: Diagonal region is intra-block. Are there inter-block features?

<style>
img:hover {
  transform: none;
}
</style>

---

## Training Block-skim

Single-task multi-objective

$$
\mathcal{L}_{\text{BlockSkim}} =\sum_{m_{i} \in\{\text{blocks}\}} \operatorname{CELoss}\left(m_{i}, y_{i}\right)
$$

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{QA}+\alpha \sum\limits_{l}^{\#layer}\left(\beta \mathcal{L}_{\text{BlockSkim}}^{l, y=1}+\mathcal{L}_{\text{BlockSkim}}^{l, y=0}\right)
$$

1. No skiming when training
2. Does not affect the backbone model calculation
3. Block-skim Loss **improves** the original QA training
4. Seperate $y=1$ or $0$ and balance ($\beta$) cuz most blocks don't contain answer

<style>
center {
  font-size: 30px;
}
</style>

---

## Block-skim Results --- Score

![skim-main](/img/skim-main.png)

- Deformer preprocess and caches the context paragraphs at early layers to reduce the actual inference sequence length
- Length Adaptive Transformer is toekn pruning

<br>

Block-Skim objective is consistent with QA objective and improves its accuracy!

<style>
img {
  width: 100%;
}
img:hover {
  transform: none;
}
</style>

---

## Block-skim Results --- FLOPs

![skim-flops](/img/skim-flops.png)

- Block-skim provides 2~3x reduction in FLOPs
- Accuracy drop is not significant
- Plug-and-play on various Transformer-based models

<style>
img {
  margin-top: -10px;
  width: 80%;
}
</style>

---

## Block-skim Results --- Ablation

![skim-ablation](/img/skim-ablation.png)

- 2 stage training is less effective (ID-3), Block-skim Loss is helpful
- Multi-hop compatability (ID-10~12)
  - ID-12 set supporting facts (i.e., evidence) to 1 in Block-skim Loss.<br>Get higher score, but average accuracy of skim predictors (CNN) is worse (?)<br>Thus, answer-only objective is enough (?)

<style>
img {
  margin-top: -10px;
  width: 40%;
}
</style>

---

## Method 2 --- LTP [^ltp]

![ltp](/img/ltp.png)

[^ltp]: [Learned Token Pruning for Transformers<br>[arXiv 2021] UC Berkeley, Samsung](https://arxiv.org/abs/2107.00910)

- A **simple, adaptive & robust threshold-based** token pruning method
- 2.10× FLOPs reduction w/ <1% accuracy degradation

<style>
sup {
  background-color: #1f85ad;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  background-clip: text;
  clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
p {
  text-align: center;
}
img {
  width: 80%;
  margin-top: -20px;
}
.footnotes p {
  text-align: left;
}
</style>

---

## Why Using Threshold?

1. **Adaptive**: PoWER-BERT, Length-Adaptive Transformer (LAT) use a fixed config throughout a dataset, **but sequence length varies a lot!**
   - ![dataset](/img/dataset.png)
2. **Efficient**: Above 2 & SpAtten, TR-BERT use top-k for token pruning, **which is much expensive!**
   - ![topk-time](/img/topk-time.png)

<style>
img {
  width: 75%;
}
</style>

---

## Threshold Token Pruning

Attention of token $\mathbf{x}_{i}$ and $\mathbf{x}_{j}$

$$
\mathbf{A}^{(h, l)}\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\operatorname{softmax}\left(\frac{\left(\mathbf{x}^{T} \mathbf{W}_{q}^{T}\right) \left(\mathbf{W}_{k} \mathbf{x}\right)}{\sqrt{d}}\right)_{(i, j)}
$$

The importance score of token $x_i$ in layer $l$

$$
s^{(l)}\left(\mathrm{x}_{i}\right)=\frac{1}{\#head} \frac{1}{\#token} \sum_{h=1}^{\#head} \sum_{j=1}^{\#token} \mathbf{A}^{(h, l)}\left(\mathrm{x}_{i}, \mathrm{x}_{j}\right)
$$

Prune the token if $s^{(l)}\left(\mathrm{x}_{i}\right) < \theta^{(l)}$

<style>
img {
  width: 80%;
}
</style>

---

## Learning the Threshold

Hard mask

$$
M^{(l)}\left(\mathrm{x}_{i}\right)= \begin{cases}1 &, \text{if } s^{(l)}\left(\mathrm{x}_{i}\right)>\theta^{(l)} \\ 0 &, \text{otherwise}\end{cases}
$$

**No** gradient, **Non**-differentiable, **Cannot** estimate gradient ❌

<br>

<v-click>

Soft mask

$$
\tilde{M}^{(l)}\left(\mathrm{x}_{i}\right)=\sigma\left(\frac{s^{(l)}\left(\mathrm{x}_{i}\right)-\theta^{(l)}}{T}\right)
$$

**Has** gradient, **Is** differentiable ✔️

</v-click>

<style>
.katex {
  font-size: 1.2em !important;
}
</style>

---

## Training LTP

1. Finetune
2. Train finetuned model & thresholds with **SOFT** mask
3. Binarize the mask & fix the thresholds
4. Finetune only the model (**HARD** mask cannot train but OK to inference)

<v-click>

- Trick
  - Add L1 regularization to penalize the network if tokens are unpruned
  - $$
    \mathcal{L}_{\text {new }}=\mathcal{L}+\lambda \mathcal{L}_{\text {reg }} \text{ where } \mathcal{L}_{\text {reg }}=\frac{1}{\#layer} \sum_{l=1}^{\#layer}\left\|\tilde{M}^{(l)}(\mathrm{x})\right\|_{1}
    $$
  - If much token unpruned, $\sum_{i=1}^{n} \tilde{M}^{(l)}\left(\mathrm{x}_{i}\right)$ is large, $\mathcal{L}_{\text {reg }}$ is large

</v-click>

<style>
.katex {
  font-size: 1.0em !important;
}
</style>

---

## LTP Results --- Main

![ltp-main](/img/ltp-main.png#w40)

![ltp-remain](/img/ltp-remain.png#w80)

---

## LTP Results --- Comparative

![ltp-comp](/img/ltp-comp.png)

- LAT is good when train set and dev set has similar length distribution (QQP)
- LTP is adaptive, but advantage is not very significant

<style>
img {
  width: 80%;
}
</style>

---

## LTP V.S. LAT

Train on short samples (~Q2), eval on all length

![ltp-lat](/img/ltp-lat.png)

<br>

Rethinking: What about LAT training on long samples (Q3~)?

<style>
img {
  width: 70%;
}
</style>

---

## Method 3 --- Magic Pyramid [^mp]

![mp](/img/mp.png)

- **Token pruning** is good when sequence is **long**
- But, **early exit** is good when sequence is **short**
- Exploit the synergy!

[^mp]: [Magic Pyramid: Accelerating Inference with Early Exiting and Token Pruning<br>[NIPS Workshop 2021] Monash University, Amazon](https://arxiv.org/abs/2111.00230)

<style>
sup {
  background-color: #1f85ad;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  background-clip: text;
  clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
img {
  width: 40%;
}
</style>

<!-- NIPS Workshop Efficient Natrual Language and Speech Processing 2021.12
also found in ACL ARR Anonymous submission 2022.01 -->

---

## Sequence Length

![mp-len](/img/mp-len.png)

| X-Axis |                |
| :----- | -------------: |
| short  |  1 ~ 35 tokens |
| middle | 35 ~ 70 tokens |
| long   |    > 70 tokens |

Recap: correlation between early exit and length - negative but loose! [^right-tool]

[^right-tool]: [The Right Tool for the Job: Matching Model and Instance Complexities<br>[ACL 2020] Allen AI, University of Washington](https://arxiv.org/abs/2004.07453)

<style>
table {
  width: 30% !important;
  float: left;
  margin-right: 8%;
}
img {
  width: 75%;
}
p {
  font-size: 22px;
}
.footnotes-sep {
  margin-top: -2%;
}
</style>

---

## Training MP

1. Finetuning (vanilla **model**)
2. Soft mask training (**model** & thershold)
3. Hard mask training (**model** w/ hard mask)
4. Exit training (exits KLDiv Loss w/ hard mask)

![mp-train](/img/mp-train.png)

Rethinking: training process is complex, is it FAIR? (same question to LTP)

<style>
img {
  width: 90%;
}
img:hover {
  transform: none;
}
</style>

---

## MP Result --- Comparative

![mp-comp](/img/mp-comp.png#w80)

![mp-tal](/img/mp-tal.png#w60)

MP benefits both from token pruning and early exiting

<style>
img {
  margin-bottom: 3%;
}
</style>

---

## Extend --- TOKEE [^tokee]

![tokee](/img/tokee.png)

[^tokee]: [Accelerating BERT Inference for Sequence Labeling via Early-Exit<br>[ACL 2021] FDU](https://arxiv.org/abs/2105.13878)

<style>
img {
  margin-top: -2%;
  width: 60%;
}
img:hover {
  transform: none;
}
sup {
  background-color: #1f85ad;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  background-clip: text;
  clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

## Method 4 --- Efficient Attention [^lra]

| ![lra](/img/lra.png#w80) | <ol><li>X-axis is speed</li><li>Y-axis is performance</li><li>Circle is memory footprint</li></ol> |
| ------------------------ | -------------------------------------------------------------------------------------------------- |

[^lra]: [Long Range Arena: A Benchmark for Efficient Transformers<br>[ICLR 2021] Google](https://arxiv.org/abs/2011.04006)

<style>
img:hover {
  transform: none;
}
table {
  height: 70%;
}
th:first-child {
  width: 55%;
}
sup {
  background-color: #1f85ad;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  background-clip: text;
  clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
.footnotes p {
  text-align: left;
}
</style>

---

## Efficient Attention Illustrated

![longformer](/img/longformer.png)

![bigbird](/img/big-bird.png)

<style>
img {
  width: 90%;
}
img:hover {
  transform: none;
}
</style>

---

## Recap

1. Attention
   1. Redundancy exist, $O(n^2)$ complexity is high
   2. Less effective and efficient on long document
2. Token importance
   1. Attention distribution shows importance
   2. Choosing granularity (QA->block, classification->token)
   3. Supervised learning a mask function
3. Training
   1. Pseudo skiming/pruning when training, real when inferencing
   2. Joint training is generally good

<style>
li {
  padding: 2px 0px 2px 0px !important;
  line-height: 1.5 !important;
}
</style>
