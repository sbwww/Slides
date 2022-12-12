---
theme: seriph
background: /img/千言-问题匹配鲁棒性评测.png
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## 问题匹配鲁棒性

  by [Bowen Shen](https://sbwww.github.io)
drawings:
  persist: false
title: 问题匹配鲁棒性
---

## 申博文 顾佴彬 刘曦雨 陈梦阳 刘衍涛

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/sbwww" target="_blank" alt="GitHub"
    class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<style>
h2 {
  background-color: #fff;
  text-align: left;
  margin-left: 20px;
  margin-top: 300px;
}
</style>

---

## 方法与分工一览

![](/img/methods.drawio.svg)

<center><u>每个人都学习、实践、分享了新知识，没有人躺平</u></center>

<style>
img {
  width: 55%;
}
</style>

---

## 特征工程（博文）

|                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 特征工程主要包括<br><br><ul><li>无意义词的过滤</li><li>POS-tagging & NER</li><li>Asymmetric 结构交换</li><li>汉字纠错</li><li>其他特征</li></ul> | 无意义词通过写正则过滤，例如<br><br><ul><li>`(我想\|麻烦)*((请\|询\|问\|)问\|咨询)(一*下)\W*`</li><li>`(小(爱\|度\|o\|冰)\|siri)*\W*`</li><li>`(你\|您)们*好\|谢谢\W*`</li></ul><br>同时还将一些同义词、口语词统一，例如<br><br><ul><li>`那么`->`这么`</li><li>`(男\|女)(人\|生\|的\|孩子*)`->`$1性`</li></ul> |

<style>
table {
  border: none !important;
  margin-top: 40px;
}
table tr {
  border: none !important;
}
table td, table th {
  padding-right:2px;
  padding-left:2px;
}

</style>

---

## NER & POS-tagging

这一步是后续工作的基础，做好词性标注和命名体识别之后，可以更方便地寻找特定结构、特定词性的关键字。

这两项内容可以调用 PaddleNLP 的 TaskFlow API，其具体实现是使用 GRU+CRF

![](/img/lac.jpg)

<style>
img {
  width: 80%;
}
</style>

---

## Asymmetric 结构交换

Baseline 在 Asymmetric 和 Negative-Asymmetric 任务上的效果都较差，特别是在 Negative-Asymmetric 任务上，正确率不到 20\%，严重拉低了宏平均后的总分。所以针对这种结构的样本，我们进行**名词**（包括命名实体、以及 POS-tagging 中的多种名词）**的交换**。

![](/img/swap.drawio.svg)

需要注意的是，交换之后还需判断是否要对结果取反。

---

## 结果是否取反

| Text 1                   | Text 2                   | Text 2 swap              |        |
| ------------------------ | ------------------------ | ------------------------ | ------ |
| 拜登比特朗普**大**多少岁 | 特朗普比拜登**小**多少岁 | 拜登比特朗普**小**多少岁 | 取反   |
| 新郑到邯郸**多少公里**   | 邯郸到新郑**多少公里**   | 新郑到邯郸**多少公里**   | 不取反 |

特例：“被”字句和“把”字句

| Text 1                   | Text 2                   |                |
| ------------------------ | ------------------------ | -------------- |
| 老师**被**人们比喻成什么 | 人们**把**老师比喻成什么 | 交换后直接判断 |

存疑：

| Text 1                       | Text 2                       |                    |
| ---------------------------- | ---------------------------- | ------------------ |
| 广州到武汉的飞机票**多少钱** | 广州到武汉的飞机票**多少钱** | 目前认为是不相等的 |

---

## 汉字纠错

使用 macbert4csc 做非生成式的单字纠错。macbert 全称为 MLM as corrector，用 [MASK] 的方式来预测正确的字实现纠错。实际使用时发现纠错性能一般，会“误伤”很多正确的字，所以我们增加了一个**置信度判断**，当预测概率分布的熵较高时，就不进行纠错。同时，**跳过** PER、LOC、TIME、ORG 等**命名实体**的纠错。

![](/img/csc.png)

<style>
img {
  width: 50%;
}
</style>

---

## 其他特征

除上述特征之外，还设计了**针对时间、命名实体、拼音**等的比较

总的来说，上述的特征工程总共能贡献 3-5 点的宏平均正确率提升。

具体到任务上，在 Asymmetric，Negative-Asymmetric，Named-Entity，Temporal，Misspelling 等任务中有比较明显的提升。

---

## Pooling（梦阳）

ERNIE-gram 将 pooled_output[^po] 作为整个文本对的表示输入分类器，在此基础上探究对 sequence_output[^so] 做平均池化和最大池化是否会得到更好的效果。

[^po]: [CLS] token，大小为 [batch_size, hidden_size]
[^so]: 模型最后一层的隐藏状态序列，大小为 [batch_size, sequence_length, hidden_size]

![](/img/pooling.png)

<style>
img {
  width: 50%;
}
</style>

---

## Pooling

|              | Dev Acc | Test Acc (Marco) |
| ------------ | :-----: | :--------------: |
| Mean Pooling |  86.5   |       73.9       |
| Max Pooling  |  86.7   |       74.0       |
| CLS          |  85.8   |       70.0       |

在用 BERT 处理文本任务时，最后一层平均池化和最大池化相比于 [CLS] 效果要好，在 ERNIE Gram 中平均池化和最大池化可以获得稍好的性能但不多。

<style>
table {
  margin-top: 20px;
  width: 60% !important;
}
</style>

---

## TextCNN

TextCNN 对关键词更加敏感，所以考虑对 BERT 输出的表示再用 TextCNN 进行特征提取。

![](/img/textcnn.png)

<style>
img {
  width: 50%;
}
</style>

---

## TextCNN

使用了 BERT+TextCNN+Max Pooling 的结构，对比直接使用 BERT 的结果

|              | Dev Acc | Test Acc (Marco) |
| ------------ | :-----: | :--------------: |
| BERT         |  70.0   |        -         |
| BERT+TextCNN |  78.4   |       59.4       |
| ERNIE Gram   |  85.8   |       70.0       |

由结果可以看出，使用了 TextCNN 的效果显著好于使用 BERT 的 [CLS] 池化。时间原因尚未来得及用到 ERNIE-gram 上。

<style>
table {
  margin-top: 20px;
  width: 60% !important;
}
</style>

---
layout: image-right
image: /img/sbert.png
---

## S-ERNIE（曦雨）

考虑到孪生网络广泛应用于语义相似度计算，并且孪生网络在分类时能够考虑**句子对向量之间的语义相似度之差**，这个特点可能有助于模型关注到问题匹配鲁棒性任务中的相似句子语义的细微差别，所以我尝试了采用孪生网络框架训练问题匹配模型，即将 SentenceBERT 中的预训练模型替换为 ERNIE3-base 得到的 SentenceERNIE 模型。

---

## S-ERNIE 结果和分析

预测结果对大部分的测试都分类为相同，即标签为 1。导致了极端评分结果，即部分任务上评分接近 100 而部分任务上接近 0，宏平均大约 50 分。原因可能是 SentenceBERT 这种孪生网络模型只在网络的顶层进行特征交互，而问题匹配鲁棒性任务中句子的语法和语义都是很相近的。将一对句子分开输入孪生网络，并依靠顶层的特征交互，可能无法充分利用句子间细粒度的词的信息。

| ![](/img/lxy1.png) | ![](/img/lxy2.png) | ![](/img/lxy3.png) |
| ------------------ | ------------------ | ------------------ |

这个结果和比赛的**测试数据构造方法**是吻合的，也就是句子对整体类似，只有少数词不同。另一方面反映了 14 项任务的标签特征不太合理，即**很多任务全正、很多任务全负**，使特征工程能打出很高的分数。

---

## 对抗训练（佴彬）

对抗样本最早在 ICLR'15 上提出，通过梯度下降的反方向构造扰动，叠加到原样本中，产生对抗样本，从而用对抗样本和原样本二者产生更鲁棒的梯度输出。

考虑到测试集和训练集的数据差异大，希望通过对抗训练的方式增加模型的泛化性,尝试了目前广泛应用的 FGM 和 PGD 两种方式。

---

## FGM, ICLR'17

- 原理：在构造对抗扰动中，用梯度 $g$ 的实际方向来构造 [^fgm]
- 实现：在本项目中，在 Ernie 的 Embedding 层加对抗扰动
  1. 输入样本 $x$，得到梯度 $g$
  2. 在 Embedding 层上根据梯度 $g$，计算扰动 $r=e*g/||g||_2$
  3. 将扰动 $r$ 叠加到样本 $x$ 中，重新得到梯度 $g^\prime$，将 $g^\prime$ 叠加到 $g$ 上
  4. 恢复 Embedding 的梯度，进行模型更新

|              | Dev Acc | Test Acc (Marco) |
| ------------ | :-----: | :--------------: |
| Basline      |  86.7   |       74.3       |
| Baseline+FGM |  87.4   |       75.5       |

[^fgm]: [https://arxiv.org/abs/1605.07725](https://arxiv.org/abs/1605.07725)

<style>
table {
  margin-top: 30px !important;
  width: 65% !important;
}
</style>

---

## PGD, ICLR'18

- 原理：在 FGSM 的基础上，将一步扰动更改为多步扰动，即在一次扰动的基础上再对扰动得到的梯行扰动 [^pgd]
- 实现：在本项目中，在 Ernie 的 Embedding 层加对抗扰动
  1. 输入样本 $x$，得到梯度 $g$
  2. 在 Embedding 层上根据梯度 $g$，计算扰动 $r=a*g/||g||_2$
  3. 如果扰动 $r$ 不大于 $e$<br>
     &nbsp;&nbsp;&nbsp;&nbsp;将扰动 $r$ 叠加到样本 $x$ 中，重新得到梯度 $g^\prime$，将 $g^\prime$ 叠加到 $g$ 上
  4. 重复这个过程，而后恢复 Embedding 的梯度，进行模型更新

|              | Dev Acc | Test Acc (Marco) |
| ------------ | :-----: | :--------------: |
| Basline      |  86.7   |       74.3       |
| Baseline+FGM |  87.0   |       75.4       |

[^pgd]: [https://arxiv.org/abs/1706.06083](https://arxiv.org/abs/1706.06083)

<style>
table {
  margin-top: 5px !important;
  width: 60% !important;
}
</style>

---

## ERNIE+(Bi)LSTM

在 baseline 中，因为使用 `[CLS]` 或者 Mean pooling 可能会存在信息缺失的情况，所以尝试将 ERNIE 作为 Encoder 得到的向量输入到一个 LSTM 中做分类

- ERNIE+LSTM
  - 将 ERNIE 输出的 `sequence_output` 输入到 LSTM 中，得到的最后一个 step 的 `hidden_state` 代表整句的信息，输入到分类器中
- ERNIE+BiLSTM
  - 将 ERNIE 输出的 `sequence_output` 输入到 BiLSTM 中，获取句子*从头到尾*和*从尾到头*的上下文句子信息，更好的代表整个句子。而后将 BiLSTM 最后一个 step 的 `hidden_state` 进行拼接，输入到分类器中

---

## ERNIE+(Bi)LSTM

| ![](/img/lstm.png) | ![](/img/bilstm.png) |
| :----------------: | :------------------: |
|     ERNIE+LSTM     |     ERNIE+BiLSTM     |

<style>
img {
  width: 65%;
}
</style>

---

## 结果

|                      | Dev Acc | Test Acc (Marco) |
| -------------------- | :-----: | :--------------: |
| Basline (ERNIE-gram) |  86.7   |       74.3       |
| Baseline+LSTM        |  86.7   |       74.9       |
| Baseline+BiLSTM      |  86.8   |       75.6       |
| Ernie+BiLSTM+FGM     |  88.2   |       78.1       |

<style>
table {
  margin-top: 5px !important;
  width: 65% !important;
}
</style>

---

## 大模型与 Prompt （衍涛）

基于 GLM-130B，测试了 K-Example prompt 和 Chain of Thought 两种 prompt 方式在对于最终结果的影响

![](/img/glm.png)

<style>
img {
  width: 70%;
}
</style>

---

## Naive K-Example

1. 先用 BERT-Base 给训练集、测试集中的每一组数据得出其 sentence embeding
2. 使用 Faiss 从训练集中抽出**与测试句 embedding 最相似的 K 个数据**，作为例子
3. 将这 K 个例子，以 `[CLS] sent1 [SEP] sent2 [SEP] 这两个句子是/不是一个意思` 的格式拼在测试数据前，交给 GLM-130B 进行生成
   ![Introduction to Facebook AI Similarity Search (Faiss) | Pinecone](https://d33wubrfki0l68.cloudfront.net/699c5fedaed4afadd0a45c1151aa3fc9992832df/927dd/images/faiss7.png)

最终效果，宏平均准确率：67.4

<style>
img {
  width: 50%;
}
</style>

---

## Chain Of Thought

Chain of thought (CoT) 是谷歌在最近提出一种新的大模型 prompt 范式，要求模型在给出结果的同时也给出推理过程。同样的这种模型也需要提供好 example 作为 prompt 输入给模型。

![](/img/CoT.png)

<style>
img {
  width: 70%;
}
</style>

---

## Chain Of Thought-1

我们手工构造了 k 个带推理过程的 example

```json
{
  "sent1": "脂肪肝能吃猪肝吗",
  "sent2": "脂肪肝能吃猪肉吗",
  "label": false,
  "rationale": "猪肝和猪肉不是一个东西"
}
```

---

## Chain Of Thought-2

将这 k 个 example 按照之前预定义的模版，构造成 prompt，下面是一个例子

> 句子一:脂肪肝能吃猪肝吗
>
> 句子二:脂肪肝能吃猪肉吗
>
> 问题:句子一和句子二是一个意思吗?
>
> 让我们一步步思考，先找出思路再找答案！！
>
> 思路:猪肝和猪肉不是一个东西
>
> 答案:所以说句子一和句子二不是一个意思

---

## Chain Of Thought-3

用同样的播报测试数据也构造成 prompt 的一部分，喂给大模型，让大模型做之后的补全 (complete)

> 句子一:嚼口香糖会减肥吗？
>
> 句子二: 嚼口香糖能减肥吗？
>
> 问题:句子一和句子二是一个意思吗?
>
> 让我们一步步思考，先找出思路再找答案！！
>
> 思路: _<u>**会和能是同义词**</u>_
>
> _<u>**答案:所以说句子一和句子二是一个意思**</u>_

可以看到，这种 CoT 的方式能够使得模型在给出答案的同时，也给出推理过程，这极大的提升了模型的可解释性

---

## 更多 CoT 示例

```
"sent1": "什么牌子的精油皂好",
"sent2": "什么牌子的精油好？",
"label": 0,
"rationale": "精油皂和精油不是一个东西 答案:所以说句子一和句子二不是一个意思 ",
"predict": 0

"sent1": "孕妇能用护肤品吗",
"sent2": "哪些护肤品孕妇能用？",
"label": 1,
"rationale": "孕妇能用护肤品中孕妇和护肤品的位置可以相互交换 答案:所以说句子一和句子二是一个意思 ",
"predict": 1

"sent1": "为什么坐车玩手机会晕车",
"sent2": "为什么我坐车玩手机不晕车",
"label": 0,
"rationale": "插入了不这个否定词 答案:所以说句子一和句子二不是一个意思 ",
"predict": 0
```

---

## 打榜效果为什么不好？

最终效果的宏平均正确率 60.0

我们这里只有固定的人工构造的 $k=10$ 个 example。也就是说，对于每一个数据的 examples 其实都是一样的，这样不太好。

比较理想的是，人工构造 $k > 1000$ 个 example，然后用 naive K-Example 的同样的方式，为每个测试数据去索引其最近似的 $n < 10$ 个 example 来构成 prompt。这样的 prompt 可能会效果更好。

```json
{
  "sent1": "脂肪肝能吃猪肝吗",
  "sent2": "脂肪肝能吃猪肉吗",
  "label": false,
  "rationale": "猪肝和猪肉不是一个东西"
}
```
