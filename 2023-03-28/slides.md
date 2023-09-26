---
theme: seriph
background: /img/cover.jpg
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Augmented Language Model (ALM)

  by [Bowen Shen](https://sbwww.github.io)
drawings:
  persist: false
title: Augmented Language Model (ALM)
---

# Augmented Language Model (ALM)

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

## LLM defects

<br>

> **Hallucinations**: LLMs often provide <font color="red">nonfactual</font> but seemingly plausible predictions
>
> <br>
>
> **Emergent Abilities**: Many LLMs groundbreaking capabilities seem to emerge with <font color="red">a large size</font>
>
> <br>
>
> **A fundamental defect**: LLMs are generally trained to perform statistical language modeling given
>
> 1. a <font color="red">single</font> parametric model
> 2. a <font color="red">limited context</font>, typically the $n$ previous or surrounding tokens

---

## Emergent Abilities

![emergent](/img/emergent.png#w70)

---

## Meta - Augmenting the LM

<img src="/img/alm-twi.png" style="float:right;width:65%"/>

1. Reasoning
   1. Prompting
   2. **Recursive prompting**
2. Tools and acts
   1. Use other model / modalities
   2. **Information retrieval**
   3. **Calculator / interpreter**
3. **Learning to reason, use tools, and act**
   1. Supervision
   2. Reinforcement Learning

> **reasoning**: decompose a problem into simpler sub-tasks
>
> **tools**: getting each step right

---

## Recursive prompting

Few-shot prompting is good for **simple tasks**

But, not sufficient for reasoning **complex tasks**

So, we explicitly <font color="red">decomposing</font> problems into sub-problems [^least-to-most]

![recursive_0](/img/recursive_0.png)

[^least-to-most]: [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models<br>Google Research - ICLR'23](https://arxiv.org/abs/2205.10625)

---

## Recursive prompting - Cont

![recursive_12](/img/recursive_12.png#w80)

---

## DecomP - ICLR'23 [^DecomP]

![decomp](/img/decomp.png)

Different with Least-to-most prompting:

|                            **Least-to-most**                            |                          **DecomP**                           |
| :---------------------------------------------------------------------: | :-----------------------------------------------------------: |
| identify all the sub-problems in <font color="blue">**one-shot**</font> | <font color="red">**iterative**</font> top-down decomposition |
|       answer <font color="blue">**complete query**</font> at last       |    <font color="red">**merge**</font> sub-problems results    |

[^DecomP]: [Decomposed Prompting: A Modular Approach for Solving Complex Tasks<br>Stony Brook University, Allen AI - ICLR'23](https://openreview.net/forum?id=_nGgzQjzaRy)

---

## DecomP - Procedure

![split-merge](/img/split-merge.png#w65)

![decomp-case](/img/decomp-case.png#w65)

---

## DecomP - Components

- **controller**
- **decomposer LLM** to generate prompt $P=\left(\left(f_1, Q_1, A_1 \right), \cdots,\left(f_k, Q_k, A_k \right)\right)$
- **sub-taks handlers** $f\in\mathcal{F}$ (LM, symbolic functions, etc)

![decomp](/img/decomp.drawio.svg#w55)

---

## DecomP - Results

![decomp-res1](/img/decomp-res1.png#w70)

Outperforms vanilla CoT and Least-to-most on $k$-th letter concatenation and sequence reversal tasks.

Generalizes when sequence is longer than in-context examples (OOD).

---

## DecomP - QA

1. Long-context QA (CommaQA)
   ![comma-qa](/img/comma-qa.png)
2. Open-domain QA
   ![odqa](/img/odqa.png)

---

## DecomP - QA results

<div id="imgs-left">
<img src="/img/decomp-qa.png#w70" />

<font color="blue">No-Ctxt: internal knowledge only</font>

<font color="red">NoDecomp-Ctxt: retrieve K paragraphs</font>

<font color="purple">Decomp-Ctxt</font>

</div>
<div id="imgs-right">
<img src="/img/decomp-qa1.png" />
<img src="/img/decomp-qa2.png" />
<img src="/img/decomp-qa3.png" />
</div>

<style>
  #imgs-left {
    float: left;
    width: 48%;
  }
  #imgs-right {
    margin-top: -5rem;
    float: right;
    width: 51%;
  }
</style>

---

## Information retrieval

![ircot](/img/ircot.png#w90)

---

## RR [^rr]

- Internal knowledge of LLMs will be **out-of-date** or **incorrect**
- LLMs hallucination will make **wrong inference** with correct knowledge
- Tuning LLMs to incorporate **external knowledge** is costly or impractical

<table>
<tr>
<td>

<font color="red">Rethinking and retrieval (RR)</font>

- CoT+knowledge
- post-processing
- training-free
- no input length limit

</td>
<td>

![RR](/img/rr.png#w80)

</td>
</tr>
</table>

[^rr]: [Rethinking with Retrieval: Faithful Large Language Model Inference<br>University of Pennsylvania](https://arxiv.org/abs/2301.00303)

<style>
td:first-child {
  width: 40%;
}
</style>

---

## RR - Procedure

1. query $\rightarrow$ CoT LLM
   - GPT-3 text-davinci-002
2. sampling multiple reasoning paths $R_1, \cdots, R_N$
   - as a single greedy-based reasoning is not always faithful
3. foreach $R$, **retrieve** $\mathcal{KB}$ and get knowledge $K_1, \cdots, K_M$
   - BM25 (top-10 paragraphs) + MPNet (similarity)
4. **rethink** and inference with faithfulness function $f_{\mathcal{KB}}(R_i)$
   - $\sum$ **similarity** - **contradiction**
   - $\sum$ **entailment** - **contradiction**
   - $\sum$ **similarity** + **entailment**

---

## RR - Results

<table>
<tr>
<td>

- Commonsense
  - dataset: StrategyQA
  - $\mathcal{KB}$: Wikipedia
- Temporal
  - dataset: TempQuestions
  - $\mathcal{KB}$: Wikipedia
- Tabular
  - dataset: INFOTABS
  - $\mathcal{KB}$: WordNet, ConceptNet

</td>
<td>

![rr-res](/img/rr-res.png)

</td>
</tr>
</table>

<style>
td:first-child {
  width: 45%;
}
</style>

---

## RR - Ablation

<table>
<tr>
<td>

1. Decomposed reasoning > original query

</td>
<td>

![rr-abl1](/img/rr-abl1.png#w70)

</td>
</tr>
<tr>
<td>

2. Background knowledge in tables is almost enough

</td>
<td>

![rr-abl2](/img/rr-abl2.png#w50)

</td>
</tr>
<tr>
<td>

3. Model size (on OPT models)

</td>
<td>

![rr-size](/img/rr-size.png#w80)

</td>
</tr>
</table>

<style>
td:first-child {
  width: 40%;
}
</style>

---

## Using calculator / interpreter

![alm-scores](/img/alm-scores.png#w70)

PAL $>>$ code-davinci-002 $>$ PaLM $>$ text-davinci-002

---

## PAL [^pal]

query $\rightarrow$ natural language (NL) + programming language (PL) $\rightarrow$ answer

![pal](/img/pal.png#w60)

[^pal]: [PAL: Program-aided Language Models<br>CMU](https://arxiv.org/abs/2211.10435)

---

## PAL - prompts

backbone model: Codex (code-davinci-002), strong in math tasks.

prompt: CoT with natural language (NL) and programming language (PL)

![pal](/img/pal.drawio.svg)

---

## PAL - Mathematical Reasoning

- dataset: GSM-HARD (large number version of GSM8K)

![pal-res1](/img/pal-res1.png)

> Large Numbers or Incorrect Reasoning?
>
> COT primary failure mode is the inability to perform arithmetic accurately

---

## PAL - Symbolic Reasoning & Algorithmic Tasks

- dataset: BIG-Bench Hard

![pal-res2](/img/pal-res2.png)

> Is PAL sensitive to the complexity of the question?
>
> COT’s is unstable and sensitive, while PAL remains consistent

---

## PAL - Ablation

| weaker code pre-training LMs? | ![pal-abl1](/img/pal-abl1.png#w40) |  ✔  |
| :---------------------------: | :--------------------------------: | :-: |
|    text pre-training LMs?     | ![pal-abl2](/img/pal-abl2.png#w35) |  ✔  |
|     without interpreter?      | ![pal-abl3](/img/pal-abl3.png#w35) | ❌  |
|     random varible name?      | ![pal-abl3](/img/pal-abl4.png#w60) | ❌  |

<style>
img:hover {
  background: #fff;
  transform: scale(1.5);
}
</style>

---

## Learning to reason, use tools, and act

<br>

- Few-shot prompting
  - require LM's size over emergent size
  - amount of supervision is limited by the LM’s context length<br><br>
- Fine-tuning
  - fine-tuning the LM changes and overfits the distribution of the examples<br><br>
- Prompt pre-training
  - exact gains are not verified

---

## Instruct-, FLAN- , and -IML

- [OpenAI] Instruct-GPT [^instruct]
- [Google] FLAN-T5 / PaLM [^flan]
  > On the MMLU benchmark, Flan-PaLM 540B achieves 75.2%. This is a wide margin over prior models (PaLM = 69.3%, code-davinci-002 = 68.3%, Chinchilla = 67.6%)
- [MetaAI] OPT-IML [^iml]
  > OPT-IML-Max is competitive with FLAN-T5 11B on RAFT, its performance lags behind FLAN-T5, FLAN-PaLM and the family of instruction-tuned GPT-3 models (_-davinci-_) on MMLU and BBH.

[^instruct]: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
[^flan]: [Scaling Instruction-**F**inetuned **Lan**guage Models](https://arxiv.org/abs/2210.11416)
[^iml]: [OPT-IML: Scaling Language Model **I**nstruction **M**eta **L**earning through the Lens of Generalization](https://arxiv.org/abs/2212.12017)

---

## GPT V.S. OPT V.S. T5 / PaLM

| ![opt-score](/img/opt-scores.png#w85) | ![flan-score](/img/flan-scores.png) | ![gpt4-score](/img/gpt4-scores.png) |
| :-----------------------------------: | :---------------------------------: | :---------------------------------: |
|                 Meta                  |               Google                |               OpenAI                |

GPT-4 $>>$ FLAN-PaLM $\approx$ GPT-3.5 $>$ FLAN-T5 $>$ PaLM $>$ OPT-IML $>$ OPT $>$ T5

PaLM has $3\times$ size of GPT-3

T5 is encoder-decoder

<style>
img:hover {
  background: #fff;
  transform: scale(1.5);
}
</style>

---

## Recap & future directions

- Fancy prompts
  - decomposition $\rightarrow$ easy to solve sub-tasks
  - current works _seems_ emperical
- Using tools
  - retrieval $\rightarrow$ faithful generation
  - code interpreter $\rightarrow$ accurate in math tasks
  - current works _seems_ emperical
- Learning
  - Instruction fine-tuning
  - **continual learning** for LLMs remains an open question
