---
title: Video Generation--A survey 
publishDate: 2025-03-11
description: '视频生成工作一览'
tags:
  - Video-Generation
  - Generative AI
heroImage: { src: 'thundermail.jpg', color: '#64574D' }
language: 'Chinese'
---

## 整体方向

整体而言视频生成领域有三个方向：
- video generation
- video editing
- video understanding

相关的任务有：
- 文生视频
- 无限制视频生成
- 文本引导视频生成

## 数据集和指标
### SSIM（结构相似性指数）
将原图像和合成图像的亮度，对比度，结构特征等进行对比反映生成图像的保真程度.符合人类直观判断,因为人眼优先捕捉视觉信息.

### PSNR
比较像素级别的原图像和合成图像像素的区别,公式$PSNR=10\times log_{10}(\frac{MAX^2}{MSE})$。

以上两种指标都只依赖图像本身，因此单靠他们还无法用于多模态或者conditioned generation任务。

### CLIPSIM
通过对视频内容截取多个关键帧，送入LSTM等模型获得视频级特征，或者索性直接求取图像平均值之后与文本进行对比计算相似度。通常用于多模态生成或者text2video.

以上都是帧级别的评价指标，但是视频不仅仅是图片的集合，评估性能的时候还需要考虑视频的流畅性等整体指标。

### Fréchet Video Distance (FVD)
基于FID思想，使用在Kinetics上预训练好的I3D模型提取视频的时空特征，并计算FVD：
$$\begin{align*}
\mathbf{FVD} = \|\mu_g - \mu_r\|^2 + \mathrm{Tr}(\Sigma_g + \Sigma_r - 2(\Sigma_g\Sigma_r)^{1/2})
\end{align*}$$

### Video Inception Score(Video IS)
基于2D图像计算IS的思想，使用3D ConvNet(C3D)提取视频特征，计算条件分布$p(y|x)$与$p(y)$。

