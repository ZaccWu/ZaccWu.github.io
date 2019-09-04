---
layout:     post                    # 使用的布局（不需要改）
title:      XGBoost、LightGBM、CatBoost比较               # 标题 
subtitle:    #副标题
date:       2019-09-04             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Machine Learning
--- 

# XGBoost、LightGBM、CatBoost比较

## 1 概述
在深度学习红极一时的情况下，boosting算法仍然有其用武之地，尤其在训练样本量较少、训练时间较短、缺乏调参先验等情况下，boosting算法仍然保持着其优势。kaggle比赛中boosting算法更是占据了大多数席位。本文总结了多篇文献和博客中对于三种算法的介绍，从多角度对这几种具有代表性的 boosting 算法进行对比，方便加深理解。

## 2 三种算法的共同点
从结构上来说，XGBoost、LightGBM和CatBoost都是boosting算法，其基学习器都为决策树，同时都是使用贪婪的思想来实现决策树的生长。

实际应用中，这几种以决策树为支持的算法也集成了决策树的良好的可解释性，在Kaggle中一些大型的数据集上，这几种boosting算法都能够取得相当好的performance。

就速度而言，LightGBM和CatBoost在XGBoost上做了进一步的优化改进，速度一般也要快于先前的XGBoost，同时调参环节也方便了很多。

## 3 三种算法的区别
### 3.1 树的特征
三种算法基学习器都是决策树，但是树的特征以及生成的过程仍然有很多不同

CatBoost使用对称树，其节点可以是镜像的。CatBoost基于的树模型其实都是完全二叉树。

XGBoost的决策树是Level-wise增长。Level-wise可以同时分裂同一层的叶子，容易进行多线程优化，过拟合风险较小，但是这种分裂方式也有缺陷，Level-wise对待同一层的叶子不加以区分，带来了很多没必要的开销。实际上很多叶子的分裂增益较低，没有搜索和分裂的必要。

LightGBM的决策树是Leaf-wise增长。每次从当前所有叶子中找到分裂增益最大的一个叶子（通常来说是数据最多的一个），其缺陷是容易生长出比较深的决策树，产生过拟合，为了解决这个问题，LightGBM在Leaf-wise之上增加了一个最大深度的限制。

### 3.2 对于类别型变量
调用boost模型时，当遇到类别型变量，xgboost需要先处理好，再输入到模型，而lightgbm可以指定类别型变量的名称，训练过程中自动处理。

具体来讲，CatBoost 可赋予分类变量指标，进而通过独热最大量得到独热编码形式的结果（独热最大量：在所有特征上，对小于等于某个给定参数值的不同的数使用独热编码；同时，在 CatBoost 语句中设置“跳过”，CatBoost 就会将所有列当作数值变量处理）。

LighGBM 也可以通过使用特征名称的输入来处理属性数据；它没有对数据进行独热编码，因此速度比独热编码快得多。LGBM 使用了一个特殊的算法来确定属性特征的分割值。（注：需要将分类变量转化为整型变量；此算法不允许将字符串数据传给分类变量参数）

和 CatBoost 以及 LGBM 算法不同，XGBoost 本身无法处理分类变量，只接受数值数据，这点和RF很相似。实际使用中，在将分类数据传入 XGBoost 之前，必须通过标记编码、均值编码或独热编码等各种编码方式对数据进行处理。

### 3.3 XGBoost的突破
XGBoost算是Boosting算法发展中的一个突破，首先其利用了二阶梯度来对节点进行划分，相对其他GBM来说，精度更加高；第二，利用局部近似算法对分裂节点的贪心算法优化，我们可以调节选取适当的eps，保持算法的性能且提高算法的运算速度；另外XGBoost也在损失函数中加入了L1/L2项，控制模型的复杂度，提高模型的稳定性；提供并行计算能力也是XGBoost的一个特点。

### 3.4 LightGBM所做的优化
LightGBM在算法上做的一些优化值得关注，首先LightGBM使用基于直方图的算法(histogram based algorithm)，将连续的特征进行分箱离散化，这样做的好处是能够提高训练速度并节省存储空间，这种基于histogram算法代替了先前XGBoost用pre-sorted所构建的数据结构。（因此LGBM的运算为$O(bins)$而不是$O(data)$，当然在分箱的过程中（仅求和运算）仍然需要$O(data)$）

此外，LightGBM使用单侧梯度采样（GOSS Gradient-based One-Side Sampling），这种方法保持高梯度的sample，从梯度变化较小的sample中进行随机采样。

## 4 参数上的补充
### 4.1 最基本的几个参数
三种算法在调参上有相通之处，当然也有很多不同的地方，在基础参数调节防止过拟合的过程中，learning_rate是三者所共有的，一般来说会设置为0.01-0.2不等，通常不会超过0.5。

对于树的深度，LightGBM中的max_depth默认为20，然而由于LightBGM是用leaf-wise策略来实现树的生长，因此这个参数的调节尤其重要。在CatBoost中，这个参数一般能调到16，通常我们选取1-10之间的数值。

此外，在三种boosting中有几个参数也需要留意：XGBoost中拥有min_child_weight，然而CatBoost并不支持此参数，在LightBGM中为min_data_in_leaf。此外，CatBoost可以通过l2_leaf_reg来调节过拟合（L2正则化系数）

### 4.2 与类别变量有关
类别变量有关的参数通常在CatBoost和LightGBM中进行调节（因为XGBoost不具有此类参数），cat_features是CatBoost中进行特征indexing的参数。此外，CatBoost可以用one_hot_max_size为所有特征进行one-hot encoding，该参数的取值就是独热编码后特征数量的上限值（less than 255）。

在LightGBM中，一般用categorical_feature参数来选择训练过程中我们想要用到的categorical features，这个参数有利于我们快速构建合适的训练模型。

### 4.3 速度控制参数
三种boosting方法的此类参数有所区别。在XGBoost中，用colsample_bytree控制列采样的二次采样率，用subsample控制training data的二次采样率，用n_estimators控制决策树数量的最大值；在CatBoost中，用rsm选择随机二次采样的方法，即每次分裂时使用特征数量的百分比，用iterations控制决策树的最大数量；在LightGBM中，用feature_fraction控制每次迭代中可以使用特征的部分，用bagging_fraction控制每次迭代使用的数据（有利于提升训练速度），用num_iterations控制boosting迭代的次数。




