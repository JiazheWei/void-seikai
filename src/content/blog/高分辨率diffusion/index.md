---
title: Image Super-Resolution Generatioon - A Survey
publishDate: 2025-03-10
description: '来谈一谈DM的有趣应用--高分辨率生成'
tags:
  - Diffusion Model
  - Generative AI
heroImage: { src: 'ref and rain.jpg', color: '#64574D' }
language: 'Chinese'
---

## Adversarial Diffusion Compression for Real-World Image Super-Resolution
超高分辨率现实场景图片生成领域当下的主流方法可以分为两大类：

- GAN-based, 但是依然没有克服GAN模型本身生成不稳定的缺陷，并且恢复质量受限于退化建模的准确性。
- diffusion-based，生成的可控性与稳定性更高，但是一方面推理速度慢（需要经历大量timestep），一方面由于参数量大，对计算资源要求较高。
综合来看，加速扩散模型推理是一个在Real-ISR领域相对而言更加可行的研究方向，故先将重心聚焦在diffusion-based方法上。

现有的diffusion-based方法的有这几个：
- SinSR，基于知识蒸馏将原本需要15步的ResShift蒸馏为只需要一步生成的学生模型。但是生成细节不足，有时会导致over-smooth的情况，可能失去高频纹理细节。无法恢复复杂退化场景的细节。
- OSEDiff (原sota)：采用变分分数蒸馏，兼顾了推理速度和生成质量，但是依赖大规模预训练SD模型，参数量很大，需要较多资源。
因此自然可以想到一种解决方案：$$做减法，将SD框架中的冗余部分去除；同时改进训练手段，保持生成质量。$$

解决方法： 
### 做减法，删除冗余部件

如论文中原图Fig.1，删除原VAE encoder，time step embedding，Prompt Extractor等不需要的部件，只保留U-net与VAE decoder。对于单步推理生成ISR，时间步骤可以视作相同的嵌入，因此不需要额外引入模块；此外文本提示对生成结果也意义不大，故可去除；通过其他手段来代替vae encoder，这部分部件也去除。

### 做微调，特化剩余部件

传统的下采样策略（卷积，pooling）会导致LR图像失去例如纹理和边缘细节等对于Real-iSR也许非常重要的特征，因此提出只通过PixelUnshuffle 重构像素排列，将原本$H \times W \times C$的图像按照$s\times s$
的块展开成$\frac{H}{s}\times \frac{W}{s} \times C\times s^2$，减小长宽维度，增加通道数。这部分操作代替了原本的VAE-encoder功能，能最大限度保留LR原始特征，同时不会引入额外计算负担。

针对U-net内部与VAE-decoder，也做了内部优化，以适应额外增多的通道。

### 新的训练框架

1）第一阶段先只预训练vae decoder，采用L1范数计算解码器的重构损失。
2）第二阶段包括对教师模型（OSEDiff）的知识蒸馏与对抗学习。对于教师模型，计算对于一张LR图像教师模型与学生模型（AdcSR）给出的特征图的差的L1范数。对抗学习，将一个冻结的预训练好的SD-Unet作为discriminator，引导学生模型给出的表示能够欺骗discriminator（接近高分辨率图像的表示）。整体损失优化如下：$$C = L_{\text{distill}} + \lambda_{\text{adv}} L_{\text{adv}}$$

