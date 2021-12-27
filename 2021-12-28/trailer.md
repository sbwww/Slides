---
theme: seriph
background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Efficient PLMs
  from the perspective of Early Exiting

  by [Bowen Shen](https://sbwww.github.io)
drawings:
  persist: false
title: Efficient PLMs
---

# Efficient PLMs

## from the perspective of Early Exiting

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
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

## What & Why?

To exit after being _confident_ to the results during inference.

Don't need to execute **ALL** the model layers.

<br><br>

```mermaid {scale: 0.75}
graph LR
I[Input] --> L0[Embedding]
L0 --> L1[Layer 1]
L1 --> C1{Confident?}
C1 -->|N| L2[Layer 2]
C1 -->|Y| R1[Exit 1]
R1 --> O1[Output]
L2[Layer 2] --> C2{Confident?}
C2 -->|N| L3[...]
C2 -->|Y| R2[Exit 2]
R2 --> O2[Output]
L3 --> L12[Layer 12]
L12 --> O12[Output]
```

---

## How?

1. Entropy [^deebert]
2. Patience [^pabee]
3. Learning-based
4. Pretraining [^elue]

[^deebert]: [DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference [ACL 2020]<br>University of Waterloo, Vector Institute of AI](https://arxiv.org/abs/2004.12993)
[^pabee]: [BERT Loses Patience: Fast and Robust Inference with Early Exit [NIPS 2020]<br>Beihang University, University of California, MSRA](https://arxiv.org/abs/2006.04152v3)
[^elue]: [Towards Efficient NLP: A Standard Evaluation and A Strong Baseline [WIP]<br>Fudan University, Huawei Poisson Lab](https://arxiv.org/abs/2110.07038v1)

<style>
.footnotes-sep {
  @apply mt-0 opacity-10;
}
</style>

---

## Method 1 --- Entropy

DeeBERT [^deebert]

1. Entropy as confidence
2. No further pretraining
3. Single model with multiple inference configs

![entropy_illus](/img/entropy_illus.drawio.svg)

[^deebert]: [DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference [ACL 2020]<br>University of Waterloo, Vector Institute of AI](https://arxiv.org/abs/2004.12993)

<style>
img {
  width: 40%;
  margin-left: auto;
  margin-right: auto;
  left: 0;
  right: 0;
  text-align: center;
  transition: all 0.2s;
}
img:hover {
  background: #fff;
  transform: scale(2);
}
.footnotes-sep {
  @apply mt-0 opacity-10;
}
</style>

---

## Method 2 --- Patience

PABEE [^pabee]

1. Patience as confidence
2. One-stage fine-tuning
3. Even better performance with shorter inference time!?

![pabee](/img/pabee.png)

[^pabee]: [BERT Loses Patience: Fast and Robust Inference with Early Exit [NIPS 2020]<br>Beihang University, University of California, MSRA](https://arxiv.org/abs/2006.04152v3)

<style>
img {
  width: 40%;
  margin-left: auto;
  margin-right: auto;
  left: 0;
  right: 0;
  text-align: center;
  transition: all 0.2s;
}
img:hover {
  background: #fff;
  transform: scale(2);
}
.footnotes-sep {
  @apply mt-0 opacity-10;
}
</style>

---

## Method 3 --- Pretraining

ElasticBERT [^elue]

1. Pretrained multi-exit Transformer
2. Static (base 6-layer) **beats** BERT, RoBERTa, ALBERT, MobileBERT, TinyBERT
3. Dynamic (earlt exit backbone) **beats** DeeBERT, PABEE

![elasticBERT](/img/elasticBERT.gif)

[^elue]: [Towards Efficient NLP: A Standard Evaluation and A Strong Baseline [WIP]<br>Fudan University, Huawei Poisson Lab](https://arxiv.org/abs/2110.07038v1)

<style>
img {
  width: 8%;
  margin-left: auto;
  margin-right: auto;
  left: 0;
  right: 0;
  text-align: center;
}
.footnotes-sep {
  @apply mt-0 opacity-10;
}
</style>

---

## Benchmarking

|                  SOTA                   |             Pareto SOTA [^elue]             |
| :-------------------------------------: | :-----------------------------------------: |
| <center>![sota](/img/sota.png)</center> | <center>![pareto](/img/pareto.png)</center> |

[^elue]: [Towards Efficient NLP: A Standard Evaluation and A Strong Baseline [WIP]<br>Fudan University, Huawei Poisson Lab](https://arxiv.org/abs/2110.07038v1)

<style>
.footnotes-sep {
  @apply mt-5 opacity-10;
}
img {
  height: 250px;
  transition: all 0.2s;
}
img:hover {
  background: #fff;
  transform: scale(1.5);
}
</style>
