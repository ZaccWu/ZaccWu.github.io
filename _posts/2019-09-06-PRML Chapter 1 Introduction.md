---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 1 Introduction              # 标题 
subtitle:    #副标题
date:       2019-09-06             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 
# PRML Chapter 1 Introduction

## 1.1 Example：Polynomial Curve Fitting
For a simple regression problem, our goal is to use the training set to predict new value $\hat{t}$ for input variable $\hat{x}$. The simplest way is to fit the curve with polynomial.

$$y(x,w) = w_{0} + w_{1} x + w_{2} x^{2} + \dots + w_{M}x^{M} = \sum_{j=0}^{M}w_{j}x^{j}$$

To determine the coefficients, we need to minimize the loss function. A general type of loss function could be written as:

$$E(w) = \frac{1}{2}\sum_{n=1}^{N}\{y(x_{n},w) - t_{n}\}^{2}$$

To avoid the coefficients reaching a huge value, regularization is what we need. By adding the penalty term, we can avoid overfitting:

$$\hat{E}(w) = \frac{1}{2}\sum_{n=1}^{N}{y(x_{n},w) - t_{n}}^{2} + \frac{\lambda}{2} ||w||^{2}$$

Where:
$$||w||^{2} = w^{T}w = w_{0}^2 + w_{1}^{2} + \dots + w_{M}^{2}$$

## 1.2 Probability Theory
Two important rules in probability theory:
* **sum rule**: $p(X)=\sum_{Y}p(X,Y)$
* **product rule**:$p(X,Y)=p(Y\|X)p(X)$
Base on the rules we could derive the Bayes' theorem:

$$P(Y | X) = \frac{p(X | Y)p(Y)}{p(X)} = \frac{p(X | Y)p(Y)}{\sum_{Y}p(X|Y)p(Y)}$$

### 1.2.1 Probability densities
For continuous variables, we define the probability density over x as:

$$p(x \in (a,b)) = \int_{a}^{b} p(x)dx$$

Also we have transformative rules between variables:

$$p_y(y)=p_x(g(y))|g'(y)|$$

### 1.2.2 Expectations and covariances
In this chapter we define the expectation, variance and covariance:

$$E[f] = \sum_{x}p(x)f(x)$$

$$var[f] = E[(f(x) - E[f(x)]^{2})] = E[f(x)^2] - E[f(x)]^2$$

$$cov[x, y] = E_{x,y}[{x - E[x]}{y-E[y]}] = E_{x,y}{xy} - E[x]E[y]$$

### 1.2.3 Bayesian probabilities
Here we use bayesian perspective to give a description of uncertainty:

$$p(w | D) = \frac{p(D | w)p(w)}{p(D)} = \frac{p(D | w)p(w)}{\int p(D | w)p(w)dw}$$

which allow us to value the uncertainty of $w$ after we observe the dataset $D$.
### 1.2.4 The Gaussian distribution
The Gaussian distribution is defined as:

$$N(x | \mu,\sigma^{2}) = \frac{1}{(2\pi\sigma^{2})^{\frac{1}{2}}}exp\{-\frac{1}{2\sigma^{2}}(x-\mu)^{2}\}$$

For independent identically distributed dataset, the likelihood function could be written as:

$$p(x | \mu, \sigma^{2}) = \prod_{n=1}^{N}N(x_{n} | \mu, \sigma^{2})$$

it can also be written in the log form:

$$ln p(x| \mu, \sigma^{2}) = -\frac{1}{2\sigma^{2}}\sum_{n=1}^{N}(x_{n} - \mu)^{2} - \frac{N}{2}ln \sigma^{2} - \frac{N}{2}ln(2\pi)$$

Thus we can calculate the partial derivative to get the MLE solution:

$$\mu_{ML} = \frac{1}{N}\sum_{n=1}^{N}x_{n}$$

$$\sigma_{ML}^{2} = \frac{1}{N}\sum_{n=1}{N}(x_{n} - \mu_{ML})^{2}$$

The expectation of the estimation is unbiased, but the variance is underestimated.

$$E[\mu_{ML}] = \mu$$

$$E[\sigma_{ML}^{2}] = \frac{N-1}{N}\sigma^{2}$$

### 1.2.5 Curve fitting re-visited
We assume that given the value of $x$, the corresponding value of $t$ has a Guassian distribution with a mean equal to the value $y(x,w)$. Thus we have

$$p(t | x,w,\beta) = N(t| y(x,w,\beta^{-1}))$$

The likelihood function and log likelihood function is:

$$p(t | x, w, \beta) = \prod_{n=1}^{N}N(t_{n} | y(x_{n},w), \beta^{-1})$$

$$ln p(t | x, w, \beta) = -\frac{\beta}{2}\sum_{n=1}^{N}\{y(x_{n},w) - t_{n}^{2}\} + \frac{N}{2}ln\beta - \frac{N}{2}ln(2\pi)$$

When we calculate $w_{MLE}$, we can see that maximize the likelihood function is equivalent to minimize the sum-of-square error function.

Similarily, for the precision parameter $\beta$, we have:

$$\frac{1}{\beta_{ML}}=\frac{1}{N}\sum_{n=1}^N\{y(x_n,w_{ML})-t_n\}^2$$

To predict the distribution, we substitute the MLE solution into the function:

$$p(t | x, w_{ML}, \beta_{ML}) = N(t | y(x, w_{ML}), \beta_{ML}^{-1})$$

Introduce the prior distribution of $w$ (assume Guassian):

$$p(w | \alpha) = N(w | 0, \alpha^{-1}I) = (\frac{\alpha}{2\pi})^{\frac{M+1}{2}}exp\{-\frac{\alpha}{2}w^{T}w \}$$

We can apply bayes' theorem and maximize the posterior:(MAP techique)

$$p(w | x,t, \alpha, \beta) \propto p(t | x, w, \beta)p(w | \alpha)$$

and it is equivalent to minimize regularized sum-of-square error function:

$$\frac{\beta}{2}\sum_{n=1}^{N}\{y(x_{n},w) - t_{n} \}^{2} + \frac{\alpha}{2}w^{T}w$$

### 1.2.6 Bayesian curve fitting
The pure Bayesian treatment simply corresponds to a consistent application of the sum and product rules.

$$p(t | x,X,t) = \int p(t | x, w)p(w | X, t)dw$$

## 1.3 Model Selection
When the training and testing data are limited, we can use  cross validation to improve the performance. However, it may be time costly.

## 1.4 The Curse of Dimensionality
In a high dimension space the data will be really sparse. Take the example of sphere(see what happen when $\epsilon\rightarrow 0$):

$$V_D(r)=K_D r^D$$

$$\frac{V_D(1)-V_D(1-\epsilon)}{V_D(1)}=1-(1-\epsilon)^D$$

## 1.5 Decision Theory
When we predict the category of a new data point $x$, we can use bayes' theorem:

$$p(C_{k} | x) = \frac{p(x | C_{k})p(C_{k})}{p(x)}$$

Intuitively we would choose the class having the higher posterior probability.

### 1.5.1 Minimizing the misclassification rate
We should minimize the term:
$$p(mistake) = p(x \in R_{1},C_{2}) + p(x \in R_{2},C_{1})$$

More generally, for k classes, we can maximize:

$$p(mistake)=\sum_{k=1}^Kp(x\in R_k,C_k)=\sum_{k=1}^K\int_{R_k}p(x,C_k)dx$$

### 1.5.2 Minimizing the expected loss
The uncertainty of real category could be represented by joint distribution $p(x,C_k)$, so we need to minimize the average loss:

$$E[L]=\sum_k\sum_j\int_{R_j}L_{kj}p(x,C_k)dx$$
### 1.5.3 The reject option
Set the threshold value $\theta$ to reject the $x$ whose maximum posterior $p(C_k\|x)$ value lower then $\theta$.

### 1.5.4 Inference and decision
* generative models: model the distribution of inputs and outputs $p(x,C_k)$.
* discriminative models: model the posterior probabilities directly $p(C_k\|x)$.

### 1.5.5 Loss functions for regression
The expected loss is given by:

$$E[L] = \int\int L(t,y(x))p(x, t) dxdt$$

For example when we use the squared loss:

$$E[L] = \int\int \{y(x) - t\}^{2}p(x, t) dxdt$$

By making $\frac{\partial E[L]}{\partial y(x)}$equals to 0, we can get the regression function, the optimal solution is conditional average:

$$y(x)=\frac{\int tp(x,t)dt}{p(x)}=\int tp(t|x)dt=E_t[t|x]$$

From another point of view, the expected loss function can be written as (where the second part can be seen as noise):

$$E[L] = \int \{y(x) - E[t | x]\}^{2}p(x)dx + \int var[t | x]p(x)dx$$

## 1.6 Information Theory
The amount of information can be viewed as the 'degree of surprise' on learning the value of $x$. Also the amount should be non-negative:

$$h(x) = -log_{2} p(x)$$

And the entropy can be seen as the average amount of information:

$$H[x] = -\sum_{x}p(x)log_{2}p(x)$$

### 1.6.1 Relative entropy and mutual information.
Suppose that we are modeling an unknown distribution $p(x)$ using $q(x)$. Then the average additional amount of information required to specify $x$ as a result of using $q(x)$ instead of true $p(x)$ is given by:

$$KL(p || q) = -\int p(x)ln q(x)dx - (-\int p(x)lnp(x)dx) = -\int p(x)ln\{\frac{q(x)}{p(x)}\}dx$$

which is call relative entropy or KL divergence.

If we have observed a finite set of training points(from $p(x)$), then the expectation with respect to $p(x)$ can by approximated, so that:

$$KL(p || q)  \simeq \frac{1}{N}\{-ln q(x_{n} | \theta) + ln p(x_{n}) \}$$

which shows that minimizing KL divergence is equivalent to maximizing the likelihood function.

We can also use mutual information to judge whether two distributions are closed to independent:

$$I[x, y] = KL( p(x, y) | p(x)p(y) ) = -\int\int p(x,y)ln (\frac{p(x)p(y)}{p(x,y)})dxdy$$

The relationship between mutual information and conditional entropy:

$$I[x,y] = H[x] - H[x | y] = H[y] - H[y | x]$$
