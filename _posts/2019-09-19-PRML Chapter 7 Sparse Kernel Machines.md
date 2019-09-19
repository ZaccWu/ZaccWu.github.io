---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 7 Sparse Kernel Machines               # 标题 
subtitle:   #副标题
date:       2019-09-19             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 

# PRML Chapter 7 Sparse Kernel Machines

## 7.1 Maximum Margin Classifiers
The two-class classification problem using linear models of the form:

$$y(x) = w^{T}\phi(x) + b$$

The maximum margin solution is found by solving:

$$\arg\max_{w, b} \left\{\frac{1}{||w||}\min_{n}[t_{n}(w^{T}\phi(x_{n}) + b)] \right \}$$

where we have the constraint: $t_n y(x_n)\geq 0$

The direct solution of this optimization problem would be very complex and we can transform the objective function to:

$$\arg\max_{w, b} \frac{1}{2}||w||^2$$

where we have:

$$t(w^{T}\phi(x_{n}) + b) \geq 1$$

To solve the problem, we introduce Lagrange multipliers $a_n\geq 0$, and the Lagrangian function will be:

$$L(w, b, a) = \frac{1}{2}||w||^{2} - \sum_{n=1}^{N}a_{n}\{ t_{n}(w^{T}\phi(x_{n}) - 1) \}$$

Setting the derivatives with respect to $w$ and $b$ equal to zero we will obtain:

$$w = \sum_{n=1}^{N}a_{n}t_{n}\phi(x_{n})$$

$$0 = \sum_{n=1}^{N}a_{n}t_{n}$$

Eliminating $w$ and $b$ from $L(w,b,a)$ using these conditions then gives the dual representation of the maximum margin problem in which we maximize:

$$\tilde{L}(a) = \sum_{n=1}^{N}a_{n} - \frac{1}{2}\sum_{n=1}^{N}\sum_{n=1}^{N}a-{n}a_{m}t_{n}t_{m}k(x_{n}, x_{m})$$

with respect to the constraints:

$$a_{n} \geq 0,~~~~ n = 1, \dots, N$$

$$\sum_{n=1}^{N}a_{n}t_{n} = 0$$

Substituting for $w$, using parameters $\{a_n\}$ and the kernel function:

$$y(x)=\sum_{n=1}^N a_n t_n k(x,x_n)+b$$

A constrained optimization of this form satisfies the KKT conditions:

$$a_{n} \geq 0$$

$$t_{n}y(x_{n}) - 1 \geq 0$$

$$a_{n}\{ t_{n}y(x_{n}) - 1 \} = 0$$

thus for every data point, either $a_n=0$ or $t_n y(x_n)=1$. Solving for $b$ to give:

$$b = \frac{1}{N_{S}}\sum_{n\in S}\left( t_{n} - \sum_{m\in S}a_{m}t_{m}k(x_{n}, x_{m})  \right)$$

### 7.1.1 Overlapping class distributions
When we are going to maximize the margin while softy penalizing points that lie on the wrong sie of the margin boundary, we therefore minimize:

$$C\sum_{n=1}^{N}\xi_{n} + \frac{1}{2}||w||^{2}$$

The constraints will be replaced with:

$$t_{n}y(x_{n}) \geq 1-\xi_{n} , ~~~ n=1, \dots, N$$

$$\xi_{n} \geq 0$$

The corresponding Lagrangian is given by:

$$L(w, b, \xi, a, \mu) = \frac{1}{2}||w||^{2} + C\sum_{n=1}^{N}\xi_{n}-\sum_{n=1}^{N}a_{n}\{t_{n}y(x_{n}) - 1 + \xi_{n}\} - \sum_{n=1}^{N}\mu_{n}\xi_{n}$$

The corresponding set of KKT conditions are given by:

$$a_{n} \geq 0$$

$$t_{n}y(x_{n}) - 1 + \xi_{n} \geq 0$$

$$a_{n} (t_{n}y(x_{n}) - 1 + \xi_{n}) = 0$$

$$\mu_{n} \geq 0$$

$$\xi_{n} \geq 0$$

$$\mu_{n}\xi_{n} = 0$$

After we eliminate $w$,$b$ and $\xi_n$ from the Lagrangian we obtain the dual Lagrangian in the form:

$$\tilde{L}(a) = \sum_{n=1}^{N}a_{n} - \frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N}a_{n}a_{m}t_{n}t_{m}k(x_{n}, x_{m})$$

with respect to the subjection:

$$0 \leq a_{n} \leq C$$

$$\sum_{n=1}^{N} a_{n}t_{n} = 0$$

Similarly, a numerically stable solution is obtained by:

$$b = \frac{1}{N_{M}}\sum_{n\in M}\left( t_{n} - \sum_{m\in S}a_{m}t_{m}k(x_{n}, x_{m})  \right)$$

### 7.1.2 Relation to logistic regression
The objective function can be written in the form:

$$\sum_{n=1}^N E_{SV}(y_n t_n)+\lambda ||w||^2$$

Also, we can construct an error function by taking the negative logarithm of the likelihood function that, with a quadratic regularizer, takes the form:

$$\sum_{n=1}^N E_{LR}(y_n t_n)+\lambda ||w||^2$$

### 7.1.3 Multiclass SVMs
### 7.1.4 SVMs for regression
To obtain sparse solutions, we therefore minimizing a regularized error function:

$$C\sum_{n=1}^{N}E_{\epsilon}(y(x) - t_{n}) + \frac{1}{2}||w||^{2}$$

Introducing slack variables, the error function for support vector regression can then be written as:

$$C\sum_{n=1}^{N}(\xi_{n} + \hat{\xi_{n}}) + \frac{1}{2}||w||^{2}$$

with the corresponding conditions:

$$y_{n} - \epsilon \leq t_{n} \leq y_{n} + \epsilon$$

$$\xi_{n} \geq 0,\ \ \hat{\xi_{n}} \geq 0$$

The dual problem involves maximizing:

$$\tilde{L}(a, \hat{a}) = -\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N}(a_{n} - \hat{a_{n}}) (a_{m} -\hat{a_{m}})k(x_{n}, x_{m}) - \epsilon\sum_{n=1}^{N}(a_{n} + \hat{a_{n}}) + \sum_{n=1}^{N}(a_{n} - \hat{a_{n}})t_{n}$$

with respect to the condition:

$$0 \leq a_{n} \leq C$$

$$0 \leq \hat{a}_{n} \leq C$$

Solve the dual problem and we can see the predictions for new inputs can be made using:

$$y(x) = \sum_{n=1}^{N}(a_{n} - \hat{a}_{n})k(x, x_{n}) + b$$

which is again expressed in terms of the Kernel function. The corresponding KKT conditions:

$$a_{n}(\epsilon + \xi_{n} + y_{n} - t_{n}) = 0$$

$$\hat{a}_{n}(\epsilon + \hat{\xi}_{n} - y_{n} + t_{n}) = 0$$

$$(C - a_{n})\xi_{n} = 0$$

$$(C - \hat{a}_{n})\hat{\xi}_{n} = 0$$

Solving for $b$ we obtain:

$$b = t_{n} - \epsilon - \sum_{m=1}^{N}(a_{m} - \hat{a}_{m})k(x_{n}, x_{m})$$

### 7.1.5 Computational learning theory

## 7.2 Relevance Vector Machines
### 7.2.1 RVM for regression
The relevance vector machine for regression is a linear model of the form discussed in Chapter 3 but with a modified prior that results in sparse solutions. The condition distribution is:

$$p(t| \mathbf{x}, \mathbf{w}, \beta) = N(t | y(x), \beta^{-1})$$

where $\beta=\sigma^{-2}$ is the noise precision and the mean is given by a linear model of the form:

$$y(x) = \sum_{i=1}^{M}w_{i}\phi_{i}(x) = w^{T}\phi(x)$$

this general expression takes the SVM-like form:

$$y(x) = \sum_{n=1}^{N}w_{n}k(x, x_{n}) + b$$

Suppose we have N observations, the likelihood function is given by:

$$p(\mathbf{t} | \mathbf{X}, w, \beta) = \prod_{n=1}^{N}p(t_{n} | x_{n}, w, \beta)$$

The prior distribution over the parameter $w$:

$$p(w| \alpha) = \prod_{i=1}^{M}N(w_{i} | 0, \alpha_{i}^{-1})$$

It is easy to infer that the posterior distribution for $w$ takes the form:

$$p(w|\mathbf{t},\mathbf{X},\alpha,\beta)=N(w|m,\Sigma)$$

where $m=\beta\Sigma\mathbf{\Phi^T t}$ and $\Sigma=(\mathbf{A}+\beta\mathbf{\Phi^T\Phi})^{-1}$.

We should maximize the marginal likelihood function:

$$p(\mathbf{t}|\mathbf{X},\alpha,\beta)=\int p(\mathbf{t}|\mathbf{X},w,\beta)p(w|\alpha)dw$$

The log marginal likelihood will be the form:

$$p(\mathbf{t} | \mathbf{X}, \alpha, \beta) = \ln N(\mathbf{t} | \mathbf{0, \mathbf{C}})$$

$$C = \beta^{-1}\mathbf{I} + \mathbf{\Phi A^{-1}\Phi^{T}}$$

The predictive distribution will be:

$$p(t| x, \mathbf{X}, \mathbf{t}, \alpha^{*}, \beta^{*}) = N(t | m^{T}\phi(x), \sigma^{2}(x))$$

### 7.2.2 Analysis of sparsity
### 7.2.3 RVM for classification
We can extend the relevance vector machine framework to classification problems by applying the ARD prior over weights to a probabilistic linear classification model of that in Chapter 4.
