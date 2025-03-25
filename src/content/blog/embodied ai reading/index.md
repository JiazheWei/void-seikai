---
title: Embodied AI探索-- Diffusion Policy
publishDate: 2025-03-25
description: '来谈一谈DM的有趣应用-- Diffusion Policy'
tags:
  - Diffusion Model
  - Embodied AI
heroImage: { src: 'thundermail.jpg', color: '#64574D' }
language: 'Chinese'
---

Diffusion model在2D生成和3D生成上都取得了成功。扩散步骤得天独厚的鲁棒性稳定性让它在视觉动作生成方面也有一席之地。

当前机器人动作序列的生成方法大致如下：

1）给出多种动作方案，选择一种执行；
2）建模动作分布，每次从分布中采样一种。

对于后者又有多种不同的形式，有的像能量分数方法，每次取能量分数最低（也就是执行效果最优）的一种执行。而diffusion policy与此类似，基于当前的观察与动作序列，通过逐步去噪生成新的动作分布，取生成分布的部分窗口执行。好处是在保证了动作速度不慢的情况下，兼顾了需要迅速转换状态的复杂情境，以及动作的一致性。

整体流程与2D diffusion model相同不再赘述。关键在于visual encoder与噪声建模。

Diffusion Policy原文使用CNN作为visual encoder。在不同的角度设置camera，并分别用不同的encoder处理来自不同方向的视觉图像。

噪声方面，构成与2D diffusion 相似：

$$L = \text{MSE}(\varepsilon_k, \varepsilon_\theta (O_t, A_{0 \ t} + \varepsilon_k, k))$$

其中$O_{t}$是当前观察到的周围情况，$A_0$是原始动作，训练流程相当于给初始动作加上不同程度的噪声，将视觉环境和动作作为输入喂给CNN，然后命令其预测噪声，最后用KL散度计算分布差异。


