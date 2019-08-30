---
layout:     post                    # 使用的布局（不需要改）
title:      Something about XGBoost               # 标题 
subtitle:    #副标题
date:       2019-08-30             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Machine Learning
--- 
# Something about XGBoost

XGBoost is one of the most widely used machine learning algorithm. This passage talks about the main idea of XGBoost and my conprehension about the model.

## 1 Background Knowledge
To understand XGBoost, firstly we need to understand some points about it.
### 1.1 Boosting
Boosting means to improve a set of weak learning algorithms to a strong learning algorithm. This term belongs to the category of Ensemble Learning. To simplify it, boosting can be compared with team members cooperating to solve a problem.
### 1.2 Gradient Boost Machine
Gradient Boost Machine(GBM) refers to the learning algorithm that improves based on gradient. In Gradient Boosting, the negative gradient is regarded as the error measurement of the previous basis learner, and the error made in the last round is corrected by fitting the negative gradient in the next round of learning.
GBDT is a kind of GBM, the basis learner of GBDT is decision tree. And XGBoost makes the improvement in many aspects. We use decision tree as its basis learner due to its high interpretation capability and its fast speed.

## 2 Mathematical Principle
Before referring to its mathematical principle, some basic rules and representations should be mentioned.
### 2.1 Additive model
XGBoost or GBDT can be regarded as an additive model combined with a number of trees:

$$\hat{y_i}=\sum^K_{k=1}f_k(x_i),f_k\in F$$

Usually we construct an additive model by forward stagewise algorithm. From the front to the end, the additive model learns one basis function each time and approximates objective function steps by steps.(This method is called boosting)

$$\begin{aligned} 
\hat{y^0_i}&=0\\
\hat{y^1_i}&=f_1(x_i)=\hat{y^0_i}+f_1(x_i)\\
\hat{y^2_i}&=f_1(x_i)+f_2(x_i)=\hat{y^1_i}+f_2(x_i)\\
...\\
\hat{y^t_i}&=\sum_{k=1}^tf_k(x_i)=\hat{y^{t-1}_i}+f_t(x_i)
\end{aligned}$$

### 2.2 Taylor Expansion
Usually a certain function can be written in this Second order Taylor Expansion form:

$$f(x+\Delta x)\approx f(x)+f'(x)\Delta x +\frac{1}{2}f''(x)\Delta x^2$$

According to the identities of additive model that stated above, the object function of a certain problem can be written as:

$$
\begin{aligned}
F^t&=\sum^n_{i=1}l(y_i,\hat{y^t_i})+\sum_{i=1}^t\Omega (f_i)\\
&=\sum_{i=1}^n l(y_i,\hat{y^{t-1}_i}+f_t(x_i))+\Omega(f_t)+C
\end{aligned}
$$

Here $l$ means loss function and $C$ is constant. $\Omega$ is regularized term. $f_t(x_i)$ is the function that we need to learn this term.

Consequently, we can apply Taylor Expansion Rules to it, the formula can be written as:

$$F^t=\sum_{i=1}^n[l(y_i,\hat{y_i^{t-1}})+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)+C$$

Suppose our loss function is a squared loss function, then we just need to optimize the term:

$$F^t\approx \sum_{i=1}^n[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)$$

Here $g_i=\frac{\partial l(y_i,\hat{y}^{t-1})}{\partial \hat{y}^{t-1}}$ and $h_i=\frac{\partial^2 l(y_i,\hat{y}^{t-1})}{\partial (\hat{y}^{t-1})^2}$.

## 3 Algorithm of XGBoost
Now we suppose there's a decision tree with $T$ leaf nodes. This decision tree is a vector($w\in R^T$)made up of the values of the leaf nodes. So the decision tree can be represented as$f_t(x)=w_q(x)$. The complexity of the decision tree is determined by the regularize term $\Omega=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^T w_j^2$. Define the set $A=\{I_j=i\|q(x_i)=j\}$ as the set of all the training points divided to the leaf node $j$, so the objective function can be rewritten as:

$$\begin{aligned}
F^t&\approx \sum_{i=1}^n[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)\\
&=\sum_{i=1}^n[g_iw_{q(x_i)}+\frac{1}{2}h_iw^2_{q(x_i)}]+\gamma T+\frac{1}{2}\lambda\sum_{j=1}^Tw_j^2\\
&=\sum_{j=1}^T[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)w_j^2]+\gamma T\\
&=\sum_{j=1}^T[G_iw_j+\frac{1}{2}(H_i+\lambda)w_j^2]+\gamma T
\end{aligned}$$

So when the tree has a fixed structure ($q(x)$ fixed),let $\partial F^t=0$, we have

$$w^*_j=-\frac{G_j}{H_j+\lambda}$$

Consequently the value of objective function is
$$F=-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T$$

Normally, we use greedy strategy to generate every node of the decision tree. And here's how we calculate the GAIN of each division:$Gain=F_{current\ node}-F_{left\ child}-F_{right\ child}$, specifically:

$$Gain=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_R+G_L)^2}{H_R+H_L+\lambda}]-\gamma$$

## 4 Differences between GBDT and XGBoost
* GBDT uses CART as basis classifer while XGBoost can hold different kinds of classifers.
* GBDT uses first-order derivative only but XGBoost uses the second order.
* XGBoost enables multithreading when selecting the best sharding point, greatly increasing the speed.
* To avoid over-fitting, XGBoost introduces Shrinkage and column subsampling.
