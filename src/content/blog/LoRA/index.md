---
title: LoRA--LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
publishDate: 2025-08-07
description: '聊一聊LORA'
tags:
  - LLM
heroImage: { src: 'thundermail.webp', color: '#64574D' }
language: 'Chinese'
---

以ChatGPT为代表的大模型涌现出优秀的理解、代码和推理等能力，许多玩家也想微调自己的专属ChatGPT。但是微调大模型非常吃资源，以刚刚达到涌现能力边界的7B模型为例，微调7B模型需要3×28G的显存(SGD+Momentum)，至少需要2张A100的显卡才能满足要求。因此，如何降低微调大模型所需的机器资源，是普通玩家最为关心的问题。

## 高效微调的基本原理
以语言模型为例，在微调过程中模型加载预训练参数进行初始化,并通过最大化条件语言模型概率进行参数更新$\phi$，即：

$$[ \max_s \sum_{(x,y) \in Z} \sum_{t=1}^{|y|} \log(P_\theta(y_t | x, y_{<t})) ]$$

这种方法的缺陷是非常明显的，因为需要变换的参数量$\Delta \phi$与原本的参数总量$\phi_{0}$是一个量级的。这种微调方式需要非常多的资源，被称为full fine-tuning.

一种想法是大模型那么庞大的参数量中不一定所有参数都是有用的，只需要改变其中的某一部分，就能对下游效果产生很大影响。这就是LoRA的核心想法，用一个低秩矩阵来编码$\Delta \phi$。

## Instrisic Dimension
[Aghajanyan](https://arxiv.org/abs/2012.13255)的研究表明：预训练模型拥有极小的内在维度(instrisic dimension)，即存在一个极低维度的参数，微调它和在全参数空间中微调能起到相同的效果。

同时Aghajanyan发现在预训练后，越大的模型有越小的内在维度，这也解释了为何大模型都拥有很好的few-shot能力。

我们用两个低秩向量来分解$\Delta W$, 就得到：

$$W_0 + \Delta W = W_0 + BA \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k} \quad \text{and} \quad r \ll \min(d, k) \quad (3)$$

在实际操作中，应当将可微调参数分配到多种类型权重矩阵中，而不应该用更大的秩单独微调某种类型的权重矩阵。在训练过程中，低秩的适应矩阵仅仅放大了对下游任务有用的特征，而不是预训练模型中的主要特征。

## Example
对一个MM-Dit的layout 输入模态的线性转换层用LoRA进行微调：

```python

import torch
from peft import LoraConfig, get_peft_model

# 定义layout专用的LoRA配置
layout_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["layout_mlp.fc1", "layout_mlp.fc2"],
    lora_dropout=0.1,
    bias="none"
)

# 应用LoRA到模型
def setup_layout_lora_fine_tuning(mm_dit_model):
    # 获取layout处理的MLP
    layout_mlp = mm_dit_model.layout_processor
    
    # 应用LoRA
    layout_mlp_with_lora = get_peft_model(layout_mlp, layout_lora_config)
    
    # 冻结其他参数，只训练LoRA参数
    for param in mm_dit_model.parameters():
        param.requires_grad = False
    
    # 启用LoRA参数训练
    layout_mlp_with_lora.train()
    
    return layout_mlp_with_lora

# 训练循环
def train_layout_lora(model, layout_data_loader, optimizer):
    for batch in layout_data_loader:
        layout_tokens = batch['layout_tokens']
        labels = batch['labels']
        
        # 前向传播（LoRA自动应用）
        outputs = model(layout_tokens)
        loss = compute_loss(outputs, labels)
        
        # 反向传播（只更新LoRA参数）
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

```