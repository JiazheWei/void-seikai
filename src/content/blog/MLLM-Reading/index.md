---
title: Foundation Paper Reading--Multi Modal
publishDate: 2025-03-02
description: '多模态领域基石级别工作的快速一瞥'
tags:
  - Multi-Modal
heroImage: { src: 'star moon night.jpg', color: '#64574D' }
language: 'Chinese'
---

# Blip-2
设计思路：原有的图像-视觉语言模型训练所需考虑的参数量非常庞大，预训练代价很高。因此想到不妨利用现成的视觉编码器和语言模型。但同时需要考虑到, Image Encoder 与 LLM 之间存在模态对齐的问题。为了弥补这个gap, BLIP-2提出了核心思想：

>利用现成的视觉编码器与LLM分别处理视觉信息与文本信息，通过引入一个额外的大模型实现视觉-文本特征空间的转换，并起到信息瓶颈的作用，最终生成文本信息，并高质量完成视觉任务。

BLIP-2结构图：

![alt text](image.png)

使用冻结视觉编码器提取视觉特征，再用预训练好的Q-Former提取对文本信息来说信息最丰富的视觉表示，这是第一阶段。在第二阶段，Q-Former将提取的视觉特征送给下游冻结LLM作为视觉信息提示，从而产生文本信息。Q-Former输出维度远小于视觉编码器，还能起到信息瓶颈强迫提取最有用表示的作用，对抗遗忘灾难。

Q-Former一阶段的训练方式是通过优化图像-文本对比，图像-文本匹配，与基于图像文本生成学习三个目标进行。二阶段是利用冻结的Image Encoder将视觉表示输入Q-Former，将Q-Former给出的视觉特征经过线性层对齐之后与前缀结合再送给LLM，最终给出文本表示并根据模型结构设计损失目标。

