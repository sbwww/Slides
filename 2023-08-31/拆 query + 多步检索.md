---
presentation
  width: 1024
  height: 768
---

<!-- slide -->

# 拆 query + 多步检索

<!-- slide -->

## demo 图谱

![](./imgs/0.png#w40)

<!-- slide -->

## 拆分

- 问题：in-context learning 不能保证同时学会多种拆分方法
  比如==线性==问题 `姚明的老婆的父亲是谁？` 和==交集==问题 `李华和王刚的共同好友有哪些？` 的 prompt 不同
- 现行方案：==分开写==好各种类型的拆分 prompt，先==识别 query 意图==，再选择对应的 prompt 做 in-context learning

<!-- slide -->

### 拆分效果

![](./imgs/1.png#w75)

线性

![](./imgs/4.png#w75)

交集

拆分基本没有问题

<!-- slide -->

## 多步检索

根据拆分结果，逐个 query 单独做检索

- 现行方案：使用基本的 GraphQAChain，包含==实体抽取==模块、==检索==模块、==LLM== 模块
  - ==实体抽取==模块抽出 query 中的实体去做检索。默认效果一般，**可以等抽取工作合进来**
  - ==检索==模块**只检索抽出的实体**，并且还是把三元组当 plain text 检索。默认效果一般。如何加入唯一标识还未知

<!-- slide -->

### good case

> 姚明的老婆的父亲是谁？

![](./imgs/2.png#w75)

<!-- slide -->

![](./imgs/3.png#w75)

<!-- slide -->

现有 pipeline 已能较好地解决==线性 2 跳==问题

<!-- slide -->

### bad case

> 李华和王刚的共同好友有哪些？

![](./imgs/5.png#w75)

<!-- slide -->

![](./imgs/6.png#w75)

抽取出错，导致检索和生成出错

<!-- slide -->

![](./imgs/7.png#w75)

上一步出错导致错误累计，但 LLM 的理解和生成能力还是在线的

<!-- slide -->

## 总结

- 拆 query 基本没问题
- TODO: query 意图识别
- TODO: 实体抽取和检索效果差
- TODO: 改写回答 prompt
