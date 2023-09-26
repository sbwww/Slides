---
theme: seriph
background: /img/cover.jpg
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Fancy Attention and Decoding

  by [Bowen Shen](https://sbwww.github.io)
drawings:
  persist: false
title: Fancy Attention and Decoding
---

# Fancy Attention and Decoding

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

## What is the real bottle-neck of LLM?

Run Llama-7B inference with llama.cpp

|             |    Nvidia A100    |  Apple M2  |
| :---------: | :---------------: | :--------: |
| computation |  624 TOPS (90X)   |   7 TOPS   |
|  bandwidth  |  1935 GB/s(20X)   |  100 GB/s  |
|    speed    | 277 token/s (17X) | 16 token/s |

**Memory bandwidth** is the bottle-neck, not computation.

---

## Encoding & Decoding

| ![](/img/encode_latency.png) | ![](/img/decode_latency.png) |
| :----------------------------: | :----------------------------: |
|   encode / prefill / prefix    |    decode / auto-regressive    |

---

## Decoding

> Transformer is useful in

| prefix                                                                                                     | decode                                  |
| :--------------------------------------------------------------------------------------------------------- | :-------------------------------------- |
| <font color="#609EFC">Transformer is useful in</font> \_\_\_                                               | <font color="#FF4545">natural</font>    |
| <font color="#609EFC">Transformer is useful in</font> <font color="#FF4545">natural</font> \_\_\_          | <font color="#FF4545">language</font>   |
| <font color="#609EFC">Transformer is useful in</font> <font color="#FF4545">natural language</font> \_\_\_ | <font color="#FF4545">processing</font> |

Notice something redundant?

---

### KV cache

![](/img/kv_cache.drawio.svg)

---

### KV cache

1. KV cache is not _the cache_
2. Q (query) doesn't need cache (always 1 tensor)

---

## vLLM & PagedAttention

KV cache length is unpredictable, fragmentation and over-reservation cause 60% – 80% memory redundency

Adopt page table from OS to manage KV cache

![](/img/vllm.gif#w70)

---

### vLLM throughput

![](/img/vllm_throughput.png#w65)

---

## FlashAttention

Attention computation can be divide-and-conquer

Divide Q K V, compute each block on SRAM

![](/img/flashattention.png)

---

## FlashAttention speed

![](/img/flashattention_time.png)

---

## Early Exiting (Again)

![](/img/calm.gif#w50)

---

### CALM

> Confident Adaptive Language Modeling. NeruIPS 2022

Most tokens can early exit with less than 4 layers!

![](/img/calm_result.png)

---

#### Exit point -- Softmax still reigns

![](/img/softmax.png)

Early exit confidence measure:

Softmax > classifier > hidden states

---

#### But 🤔

Softmax is **expensive**! And,

NLG requires previous **KV cache**!

![](/img/calm.png)

CALM **copies** the KV cache of lower layers to high layers. 🤔

---

#### CALM's drawback

![](/img/calm_skipdecode.png)

---

### SkipDecode

> Confident Adaptive Language Modeling. NeruIPS 2022

![](/img/skipdecode.png#w90)

---

#### Skipped tokens

![](/img/token_loss.png)

Front tokens have large losses, which need more computation

---

#### SkipDecode results

![](/img/skip_result.png)

Works on LM, but fails on summarization, which is a common delima in efficient inference. 🤔

---

#### Fixed exit point

![](/img/exit_point.png#w50)

> SkipDecode pre-define fixed exit point of each token 🤔

Strength: **No Softmax**, batching √

---

#### Rethinking fixed exit point 🤔

SkipDecode:

> The token at front of each **SEQUENCE** is hard. 🤔

Recall CALM's result:

> The token at front of each **SENTENCE** is hard.

It is hard to find **SENTENCE** begining (dynamic) with batching

---

### Review early exiting

`input_tensor` of shape `[bsz, len, hsize]`

`model` with $L$ layer and $H$ hsize

- Early exiting reduces decode $L$ only
- Pruning reduces $H$
- We try to use batching with larger `bsz`

What about `len` 🤔

---

## Speculative Decoding

> Fast Inference from Transformers via Speculative Decoding. ICML 2023

![](/img/speculative.png)

- Using small model to generate multiple tokens
- Using LLM to verify the tokens, <font color="green">accept</font> / <font color="red">reject</font>. Then, and <font color="blue">re-generate rejected token</font> or generate the <font color="blue">last token</font>

---

### Algorithm

![](/img/speculative_algor.png#w45)

---

### Small model $M_q$

Generate $\gamma$ tokens **autoregressively**

![](/img/speculative_small.png#w50)

---

### LLM $M_p$

Generate $\gamma+1$ tokens **in parallel** (same as training)

![](/img/speculative_llm_gen.png#w40)

Validate $\gamma$ tokens, reject **once** LLM don't think the guess is good
$p_i(x)<q_i(x)$ → $r_i>p_i(x)/q_i(x)$

![](/img/speculative_llm_valid.png#w40)

Regenerate last token or rejected token

![](/img/speculative_llm_regen.png#w40)

---

### Output

![](/img/speculative_return.png#w70)

🎉🎉🎉

---

### Model Alignment and Speedup

Expected generated tokens in one run $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$

![](/img/speculative_tokens.png#w60)

---

### Model Alignment Evaluation

| ![](/img/speculative_res1.png) | ![](/img/speculative_res2.png) |
| -------------------------------- | -------------------------------- |

Even **N-gram** can accelerate LLM generation 🤯

---

### Speculative Decoding Speedup

| ![](/img/speculative_speed.png) | ![](/img/speculative_speedup.png) |
| --------------------------------- | ----------------------------------- |

**2-3X** speedup with a small model 🤯

---

### Strength and weakness

Strength

- Don't need to change **any** of the model!
- **Always** has a fallback accuracy to LLM!

Weakness

- Need two (or more) models, much more **memory usage**
- Small model need to be **aligned** with LLM or the efficiency falls back

---

### Block-wise generation

> Blockwise parallel decoding for deep autoregressive models. NeruIPS 2018

![](/img/blockwise.png)

---

## Resources

- [大模型推理性能优化之KV Cache解读](https://zhuanlan.zhihu.com/p/630832593)
- [NLP（十七）：从 FlashAttention 到 PagedAttention, 如何进一步优化 Attention 性能](https://zhuanlan.zhihu.com/p/638468472)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [vLLM. Easy, fast, and cheap LLM serving for everyone](https://vllm.readthedocs.io/en/latest/)
- [Accelerating text generation with Confident Adaptive Language Modeling (CALM)](https://blog.research.google/2022/12/accelerating-text-generation-with.html)
- [SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference](https://arxiv.org/abs/2307.02628)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Efficient Systems for Foundation Models [ES-FoMo](https://es-fomo.com/)@ICML2023 [talks](https://icml.cc/virtual/2023/workshop/21479)
