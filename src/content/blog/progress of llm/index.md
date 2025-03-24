---
title: Interesting progress of LLM -- A survey
publishDate: 2025-03-21
description: '快速地扫过LLM的下游发展工作'
tags:
  - LLM
heroImage: { src: 'thunderbox.jpg', color: '#64574D' }
language: 'Chinese'
---

## RAG
### Intro
LLM面对知识密集型任务时暴露了几个问题：
- 面对generation task, llm给出的回答信息量不足，描述不够准确，在细粒度操作方面明显不如专门为特定任务制定架构的大模型。
-无法解释清楚知识来源。

问题本质可以归因为原始训练数据集和模型参数记忆能力的局限，要想一次性学完所有的知识，一方面数据集的涵盖范围不可能那么大，另一方面模型的遗忘灾难也使模型很难针对要求任务的细节给出满意的操作。

### Methology
既然大模型的参数记忆能力有限，那么就考虑将模型的记忆分成两部分：参数化记忆与非参数化记忆。
参数化记忆顾名思义，就是在训练过程中嵌入模型参数的知识。参数化记忆这部分保持不动，人为引入非参数化部分，可以理解为针对请求的任务到特定的知识库中调取文档，让模型结合文档内容给出更加专业并具有针对性的回答。

具体方式：RAG分为两大块，检索器与生成器，检索器用于在给定的知识库中调取文档，调取方法为通过bi-encoder将文档和query映射到LLM embedding space，取点积最大者作为附加上下文。

生成方式有两种，按序列生成与逐token生成，前者在生成一个完整序列的时候基于同一个文档，而后者在生成每一个token的时候都可以检索新的文档。

RAG-Sequence:

$$p_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-k}(p(\cdot|x))} p_{\eta}(z|x) p_{\theta}(y|x, z) = \sum_{z \in \text{top-k}(p(\cdot|x))} p_{\eta}(z|x) \prod_{i} p_{\theta}(y_i|x, z, y_{1:i-1})$$

RAG-Token:
$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i}^{N} \sum_{z \in \text{top-k}(p(\cdot|x))} p_{\eta}(z|x) p_{\theta}(y_i|x, z, y_{1:i-1})$$

整体是一种端到端，一体化的训练结构。

## Agent

