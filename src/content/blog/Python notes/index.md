---
title: Python notes
publishDate: 2025-06-11
description: '不是torch，也不是dl，最基本的Py操作'
tags:
  - Tech
heroImage: { src: 'spotlight.webp', color: '#64574D' }
language: 'Chinese'
---

准备机试，同时也是因为突然发现做了快两年科研，自己只学了torch，而不是python，Python基础的函数操作和语法等等还是要会的，不然AI的泡沫破掉之后只能出来卖炒粉。

## 乘法

```python
import numpy as np
np.dot(x,y)
x@y
x*y
```

`np.dot()`做点积，也可以胜任一般线代乘法，功能可以被@运算符取代。

`*`乘做每个元素中每个元素一一乘起来，得到一个和原本向量长度相同的向量。对于多维数组，将所有对应位置的元素的值乘起来放到对应位置。所以*运算要求前后参加运算的元素形状必须一模一样。

## argmax

`np.argmax`函数找最大值，如果axis = 0 ,则对每一行找最大值--对于每一个列属性来说，所有行中的最大值所在行的索引；如果axis = 1,对列找最大值，返回所有列的值中最大的索引。

## 掩码

有时候对数组需要特定的筛选，但是通过遍历又太麻烦了。可以试试看布尔索引。例如我们现在有两个ndarray,一个`data`，一个`targets`，现在需要从中选出所有标签是1或者0的元素出来：

```python
import numpy as np
mask = (targets == 0)|(targets ==1)
data_selected = data[mask]
targets_selected = targets[mask]
```

其中的mask就是一连串`True,False`数组，`data[mask]`按照真值选出满足条件的元组。

## np.unique()

`unique`函数找出数组中不重复的元素有哪些，返回一个数组。
