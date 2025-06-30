---
title: Paper Reading--Multi-layout image generation and editing
publishDate: 2025-06-29
description: '多图层编辑'
tags:
  - 2D Generation
heroImage: { src: 'thundermail.webp', color: '#64574D' }
language: 'Chinese'
---
## 几个评价指标

图形设计的要求相比普通2D生成多了一些在美观和可编辑性方便的要求，有这几个比较常用的指标：

- GPT-4V:用gpt在几个维度指标方面打分，例如layout（空间布局是否合理），compliance(设计是否符合用户的prompt),color(颜色整体有无违和感)，Graphic Style（元素图形的美观性）

- 人工打分。

## CreatiPoster

图形设计的几个要求：
- 文本准确性。
- 资产保真性，最终生成的图形设计中用到的元素必须和用户所提供的assets保持一致。
- 可编辑性，如果用户想要替换其中的元素或者仅仅修改文字，必须要方便修改。
- 美观，设计要符合审美。 

于是CreatiPoster将生成任务分散到了两个模型上，一个Protocol model和一个background model，前者负责接收用户的assets和prompt，然后对每个layout(一个text或者image算一个layout)生成详细的json描述属性：大小，位置，字体等等，同时还可以给出每个layout所想要的背景的简洁需求。接着将所有json文件送入引擎比如Skia等就可以渲染得到foreground文件，最后将foreground和promt送入background model就能得到所需的背景，background + foreground拼接就可以得到完整的图形设计。

总体来看：
系统工作流程
第一阶段 - 协议生成：
用户输入 → 协议模型 → JSON协议规范

第二阶段 - 前景渲染：
JSON协议 → Skia等渲染引擎 → 完全可编辑的前景图层

第三阶段 - 背景生成：
前景图像 + 背景提示 → 背景模型 → 匹配的背景图像

第四阶段 - 最终合成：
前景 + 背景 → 完整的多层图形设计

前景模型（FM）训练时采用了两种范式，prompt-only和prompt-assets两种，前者提供描述即可，由引擎渲染出前景，送入BM即可得到背景；后者会先捕捉资产特征，然后在渲染的时候注意指定assetes的摆放位置以及其他特征。

[arxiv](https://arxiv.org/abs/2506.10890#:~:text=In%20this%20paper%2C%20we%20introduce%20CreatiPoster%2C%20a%20framework,multi-layer%20compositions%20from%20optional%20natural-language%20instructions%20or%20assets.)



## CreatiDesign
