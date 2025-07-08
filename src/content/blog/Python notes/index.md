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

## 关于赋值

使用一个数组的值赋给另一个数组时，要使用`copy`函数。否则python只是创建了指向同一个对象的指针。

```python
A = C.copy()
B = C.copy()
```

## Counter
`Counter`是python 提供了可以计数元组中元素出现次数的库。`Counter`的返回是：

```
[{label, 出现次数}]
```
例如针对data:

```
[1,1,4,5,1,4]
```
这个数组，`Counter(data)`返回：

```
[{1,3},{4,2},{5,1}]
```
Counter提供`most_common(k)`方法提取出出现次数最多的前k个元组。通过`Counter(data).most_common(1)[0][0]`就可以得到出现次数最多的标签是什么。

## ID3决策树

```python
import numpy as np
class DecisionTree(object):
    def __init__(self):
        #决策树模型
        self.tree = {}
    def calcInfoGain(self, feature, label, index):
        '''
        计算信息增益
        :param feature:测试用例中字典里的feature，类型为ndarray
        :param label:测试用例中字典里的label，类型为ndarray
        :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        :return:信息增益，类型float
        '''
        # 计算熵
        def calcInfoEntropy(label):
            '''
            计算信息熵
            :param label:数据集中的标签，类型为ndarray
            :return:信息熵，类型float
            '''
            label_set = set(label)
            result = 0
            for l in label_set:
                count = 0
                for j in range(len(label)):
                    if label[j] == l:
                        count += 1
                # 计算标签在数据集中出现的概率
                p = count / len(label)
                # 计算熵
                result -= p * np.log2(p)
            return result
        # 计算条件熵
        def calcHDA(feature, label, index, value):
            '''
            计算信息熵
            :param feature:数据集中的特征，类型为ndarray
            :param label:数据集中的标签，类型为ndarray
            :param index:需要使用的特征列索引，类型为int
            :param value:index所表示的特征列中需要考察的特征值，类型为int
            :return:信息熵，类型float
            '''
            count = 0
            # sub_feature和sub_label表示根据特征列和特征值分割出的子数据集中的特征和标签
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][index] == value:
                    count += 1
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            pHA = count / len(feature)
            e = calcInfoEntropy(sub_label)
            return pHA * e
        base_e = calcInfoEntropy(label)
        f = np.array(feature)
        # 得到指定特征列的值的集合
        f_set = set(f[:, index])
        sum_HDA = 0
        # 计算条件熵
        for value in f_set:
            sum_HDA += calcHDA(feature, label, index, value)
        # 计算信息增益
        return base_e - sum_HDA
    # 获得信息增益最高的特征
    def getBestFeature(self, feature, label):
        max_infogain = 0
        best_feature = 0
        for i in range(len(feature[0])):
            infogain = self.calcInfoGain(feature, label, i)
            if infogain > max_infogain:
                max_infogain = infogain
                best_feature = i
        return best_feature
    def createTree(self, feature, label):
        # 样本里都是同一个label没必要继续分叉了
        if len(set(label)) == 1:
            return label[0]
        # 样本中只有一个特征或者所有样本的特征都一样的话就看哪个label的票数高
        if len(feature[0]) == 1 or len(np.unique(feature, axis=0)) == 1:
            vote = {}
            for l in label:
                if l in vote.keys():
                    vote[l] += 1
                else:
                    vote[l] = 1
            max_count = 0
            vote_label = None
            for k, v in vote.items():
                if v > max_count:
                    max_count = v
                    vote_label = k
            return vote_label
        # 根据信息增益拿到特征的索引
        best_feature = self.getBestFeature(feature, label)
        tree = {best_feature: {}}
        f = np.array(feature)
        # 拿到bestfeature的所有特征值
        f_set = set(f[:, best_feature])
        # 构建对应特征值的子样本集sub_feature, sub_label
        for v in f_set:
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][best_feature] == v:
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            # 递归构建决策树
            tree[best_feature][v] = self.createTree(sub_feature, sub_label)
        return tree
    def fit(self, feature, label):
        '''
        :param feature: 训练集数据，类型为ndarray
        :param label:训练集标签，类型为ndarray
        :return: None
        '''
        #************* Begin ************#
        self.tree=self.createTree(feature,label)
        #************* End **************#
    def predict(self, feature):
        '''
        :param feature:测试集数据，类型为ndarray
        :return:预测结果，如np.array([0, 1, 2, 2, 1, 0])
        '''
        #************* Begin ************#

        results=[]
        def predict_one(tree,single_feature):
            if not isinstance(tree,dict):
                return tree

            feature_index=list(tree.keys())[0]
            value=single_feature[feature_index]
            subtree=tree[feature_index][value]
            return predict_one(subtree,single_feature)

        for i in range(len(feature)):
            label=predict_one(self.tree,feature[i])
            results.append(label)

        return   results      

        #************* End **************#
```

