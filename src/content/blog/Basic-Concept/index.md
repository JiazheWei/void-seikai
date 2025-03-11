---
title: Basic ML & DL Concepts
publishDate: 2024-09-28
description: '成为一名基础扎实的ML Researcher，路还有很长。'
tags:
  - ML
  - Dl Concepts
heroImage: { src: 'molisa.jpg', color: '#64574D' }
language: 'Chinese'
---

# AGI

AGI，全名为Artificial General Intelligence, 中文名通用人工智能，为人工智能的最高级形态，学习过程无需人类干预，可以处理任何跨领域任务，具有深层推理与抽象逻辑能力，同时还具有自我改进能力。

与当前的人工智能（弱人工智能）相对，是人工智能的终极目标。

# Transformer结构详解

## 整体结构
正式开始之前不妨先观察transformer是如何完成一个翻译任务的。

如图：
![alt text](transformer.png)

transformer由一组编码器结构与一组解码器结构组成，输入是一个中文句子，期望输出是一个翻译之后的英文句子。

第一步：首先获取每个单词的embedding.这个获取embedding的过程是首先得到每个单词的feature，然后叠加一个单词位置的embedding。表示为：$$x=embedding_{pos}+embedding_{feature}$$

在attention is all you need原文中约定最后得到的特征的维度是512.那么对于n个单词我们就得到了一个$n\times d$的feature matrix.

接着将这个特征矩阵送入编码器部分，最终就能得到句子各个单词的feature map--$c$.

在解码（翻译过程）中，思路是：在预测第$i$个单词的时候只能根据第$1\sim i-1$个单词的特征来预测，盖住（mask）后面从$i+1 \sim n$的单词。

![alt text](decoder.png)

## 细节
先谈如何得到单词的embedding.方法有很多，先前的word2vec, glove算法等都可以用。

而要采用时间步嵌入的原因，是transformer没用RNN结构，导致顺序信息不能显式利用，而这部分信息对于NLP任务是非常重要的。因此直接在embedding中加入这部分信息。方法也有很多，可以通过learnable的函数学，也可以直接按照公式嵌入。原方法使用了后者，采用余弦嵌入。
$$
\begin{aligned}
PE_{(pos, 2i)}   &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

接着是transformer的精华：自注意力机制。
每一个encoder和decoder的结构如下：
![alt text](block.png)

红色框框的部分是多头注意力机制部分，每个multi-head里面有多个self-attention块。

每一个自注意力块对输入特征x做了这件事：
>令特征x依次乘以矩阵$W_{Q},W_{K},W_{V}得到矩阵Q，K，V$。

之后就能得到自注意力输出：

$$attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$$

这里的$d_{k}$是矩阵Q,K,V的向量维度，即列数。主要是为了防止内积过大。其中的Q，K是询问-键对，用于询问一个句子中各个其他的单词对于某个单词的重要性/关联性。V表示注意力权重，对信息进行聚合。

而多头注意力机制则是对一个特征图$x$进行多次self-attention运算（假如说n次），最终就得到了n个输出$z_{x}$.我们选择将n个$z_{x}$concat，再通过线性层改变形状即可得到多头注意力机制的输出$z_{final}$,形状与输入的特征矩阵是一样的。


