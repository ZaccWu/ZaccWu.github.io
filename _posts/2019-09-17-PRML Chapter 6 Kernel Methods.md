---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 6 Kernel Methods               # 标题 
subtitle:    #副标题
date:       2019-09-17             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 

# PRML Chapter 6 Kernel Methods
For models which are based on a fixed nonlinear feature space mapping $\phi(x)$, the kernel function is given by the relation:

$$k(x, x^{'}) = \phi(x)^{T}\phi(x^{'})$$

Forms of kernel functions:

* linear kernels: $k(x,x')=x^T x$
* stationary kernels: $k(x,x')=k(x-x')$
* homogeneous kernels: $k(x,x')=k(\|\|x-x'\|\|)$

## 6.1 Dual Representations
Consider a linear regression model whose parameters are determined by minimizing a regularized sum-of-squares error funciton given by:

$$J(w) = \frac{1}{2}\sum_{n=1}^{N}\{ w^{T}\phi(x_{n}) - t_{n} \}^{2} + \frac{\lambda}{2}w^{T}w$$

Set the gradient of $J(w)$ with respect to $w$ equals to zero, we can get the solution:

$$w = -\frac{1}{\lambda}\sum_{n=1}^N \{ w^{T}\phi(x_{n}) - t_{n} \}\phi(x_n)=\sum_{n=1}^N a_n\phi(x_n)= \Phi^{T}a$$

$$a_{n} = -\frac{1}{\lambda}\{ w^{T}\phi(x_{n}) - t_{n} \}$$

We now define the Gram matrix $K=\Phi\Phi^T$, which is an $N*N$ symmetric matrix with elements:

$$K_{nm} = \phi(x_{n})^{T}\phi(x_{m}) = k(x_{n}, x_{m})$$

In terms of the Gram matrix, the sum-of-squares error function can be written as:

$$J(a) = \frac{1}{2}a^{T}\mathbf{KK}a - a^{T}\mathbf{Kt} + \frac{1}{2}\mathbf{t^{T}t} + \frac{\lambda}{2}a^{T}\mathbf{K}a$$

Setting the gradient $J(a)$ with respect to $a$ to zero, we obtain the following solution:

$$a = (K + \lambda\mathbf{I}_{N})^{-1}\mathbf{t}$$

If we substitute this back into the linear regression model, we obtain the following prediction for a new $x$.

$$y(x) = w^T\phi(x)=a^T\Phi\phi(x)=k(x)^{T}(K + \lambda\mathbf{I}_{N})^{-1}\mathbf{t}$$

where we have defined the vector $k(x)$ with elements: $k_{n}(x) = k(x_{n}, x)$.

## 6.2 Constructing Kernels
Here the kernel function is defined for a one-dimensional input space by 

$$k(x,x')=\phi(x)^T\phi(x')=\sum_{i=1}^M\phi_i(x)\phi_i(x')$$

A necessary  and sufficient condition for a function $k(x,x')$ to be a valid kernel is that the Gram matrix $K$ should be positive semidefintie for all possible choices of the set $\{x_n\}$.

Given valid kernels $k_1(x,x')$ and $k_2(x,x')$, the following new kernels will also be valid:

$$
\left\{
    \begin{aligned}
    k(x, x^{'}) & = & ck_{1}(x, x^{'}) \\
    k(x, x^{'}) & = & f(x)k_{1}(x, x^{'})f(x^{'}) \\
    k(x, x^{'}) & = & q(k_{1}(x, x^{'})) \\
    k(x, x^{'}) & = & exp(k_{1}(x, x^{'})) \\
    k(x, x^{'}) & = & k_{1}(x, x^{'}) + k_{2}(x, x^{'})\\
    k(x, x^{'}) & = & k_{1}(x, x^{'})k_{2}(x, x^{'}) \\
    k(x, x^{'}) & = & k_{3}(\phi(x), \phi(x^{'}))\\
    k(x, x^{'}) & = & x^{T}Ax^{'} \\
    k(x, x^{'}) & = & k_{a}(x_{a}, x_{a}^{'}) + k_{b}(x_{b}, x_{b}^{'}) \\
    k(x, x^{'}) & = & k_{a}(x_{a}, x_{a}^{'})k_{b}(x_{b}, x_{b}^{'})
    \end{aligned}
    \right.
$$

Another commonly used kernel takes the form:

$$k(x, x^{'}) = exp\left( -\frac{||x - x^{'}||^2}{2\sigma^{2}}  \right)$$

which is called Gaussian kernel.

One powerful approach to the construction of kernels starts from a probabilistic generative model, which allows us to apply generative models in a discriminative setting. Given a generative model $p(x)$ we can define a kernel by:

$$k(x,x')=p(x)p(x')$$

To extend this class of kernels by considering sums over products of different probability distributions:

$$k(x,x')=\sum_i p(x|i)p(x'|i)p(i)$$

Taking the limit of an infinite sum, we can also consider kernels of the form:

$$k(x,x')=\int p(x|z)p(x'|z)p(z)dz$$

Also, by extending the mixture representation we can define a kernel function measuring the similarity of two sequences $X$ and $X'$:

$$k(X,X')=\sum_Z p(X|Z)p(X'|Z)p(Z)$$

Another models is known as Fisher kernel defined by:

$$k(x,x')=g(\theta,x)^T F^{-1} g(\theta,x')$$

here the Fisher score is $g(\theta,x)=\nabla_{\theta}lnp(x\|\theta)$, $F$ is the Fisher information matrix, given by $F=E_x[g(\theta,x)g(\theta,x)^T]$.

## 6.3 Radial Basis Function Networks
Radial basis functions have the property that each basis function depends only on the radial distance (typically Euclidean) from a centre $\mu_j$, so that

$$\phi_j(x)=h(||x-\mu_j||)$$

### 6.3.1 Nadaraya-Watson model
Starting with kernel density estimation. Suppose we have a training set $\{x_n,t_n\}$ and we use a Parzen density estimator to model the joint distribution $p(x,t)$, so that

$$p(x,t)=\frac{1}{N}\sum_{n=1}^N f(x-x_n,t-t_n)$$

We now find an expression for the regression function $y(x)$, corresponding to the conditional average of the target variable conditioned on the input variable, which is given by:

$$\begin{aligned}
y(x) &= E[t | x] =\int_{-\infty}^{\infty}tp(t|x)\\
&=\frac{\int tp(x,t)dt}{\int p(x,t)dt}\\
&= \frac{\sum_{n}\int t f(x - x_{n}, t - t_{n}) dt} { \sum_{m}\int  f(x - x_{m}, t - t_{m}) dt }
\end{aligned}$$

We now assume for simplicity that the component density functions have zero mean so that:

$$\int_{-\infty}^{\infty} f(x, t)t dt = 0$$

for all values of $x$, using a simple change of variable, we then obtain:

$$y(x) = k(x, x_{n}) = \frac{g(x - x_{n})t_{n}}{\sum_{m}g(x - x_{m})} =\sum_n k(x,x_n) t_n$$

and we have defined:

$$g(x) = \int_{-\infty}^{\infty} f(x, t) dt$$

The result is known as the Nadaraya-Watson model or kernel regression.

## 6.4 Gaussian Processes
### 6.4.1 Linear regression revisited
Consider a model defined in terms of a linear combination of fixed basis functions:

$$y(x) = w^{T}\phi(x)$$

The prior distribution over $w$:

$$p(w) = N(w | 0, \alpha^{-1}\mathbf{I})$$

Evaluate the function at specific values of $x$, the vector is given by:

$$\mathbf{y} = \Phi w$$

And the mean and covariance:

$$E[\mathbf{y}] = \Phi E[w] = 0$$

$$cov[\mathbf{y}] = E[\mathbf{y}\mathbf{y}^{T}] = \mathbf{K}$$

where $K$ is the Gram matrix with elements:

$$K_{nm} = k(x_{n}, x_{m}) = \frac{1}{\alpha}\phi(x_{n})^{T}\phi(x_{m})$$

### 6.4.2 Gaussian processes for regression
In order to apply Gaussian process models to the problem of regression, we need to take account of the noise on the observed target values:

$$t_{n} = y_{n} + \epsilon_{n}$$

Consider noise processes that have a Gaussian distribution, so that:

$$p(t_{n} | y_{n}) = N(t_{n}| y_{n}, \beta^{-1})$$

The conditional distribution is given by an isotropic Gaussian of the form:

$$p(\mathbf{t} | \mathbf{y}) = N(\mathbf{t}| \mathbf{y}, \beta^{-1}\mathbf{I}_{N})$$

From the definition, the marginal distribution $p(y)$ is given by a Gaussian:

$$p(\mathbf{y}) = N(\mathbf{y} | 0, \mathbf{K})$$

We see that the marginal distribution of $\textbf{t}$ is given by:

$$p(\mathbf{t}) = \int p(\mathbf{t} | \mathbf{y})p(\mathbf{y}) d\mathbf{y} = N(\mathbf{t} | 0, \mathbf{C})$$

where the covariance matrix $C$ has elements:

$$C(x_{n}, x_{m}) = k(x_{n}, x_{m}) + \beta^{-1}\delta_{nm}$$

One widely used kernel function for Gaussian process regression is given by the exponential of a quadratic form, with the addition of constant and linear terms to give:
\
$$k(x_{n}, x_{m}) = \theta_{0}exp\left\{ -\frac{\theta_{1}}{2}||x_{n} - x_{m}||^{2} \right\} + \theta_{2} + \theta_{3}x_{n}^{T}x_{m}$$

The joint distribution over $t_1,...,t_{N+1}$ will be given by:

$$p(t_{N+1}) = N(t_{N+1} | 0, C_{N+1})$$

So the conditional distribution $p(t_{N+1}\|\textbf{t})$ is a Gaussian distribution with mean and covariance given by:

$$m(x_{N+1}) = K^{t}c_{N}^{-1}\mathbf{t}$$

$$\sigma^{2}(x_{N+1}) = c - k^{T}C_{N}^{-1}\mathbf{k}$$

### 6.4.3 Learning the hyperparameters
Techniques are based on the evaluation of the likelihood function and maximize the log likelihood.

### 6.4.4 Automatic relevance determination
The example of a Gaussian process with a two-dimensional input space $x=(x_1,x_2)$, having a kernel function of the form:

$$k(x,x')=\theta_0 exp\{-\frac{1}{2}\sum_{i=1}^2 \eta_i(x_i-x_i')^2\}$$

The ARD framework is easily incorporated into the exponential-quadratic kernel to give the following form of kernel function:

$$k(x_n,x_m)=\theta_0 exp\{-\frac{1}{2}\sum_{i=1}^D\eta_i(x_{ni}-x_{mi})^2\}+\theta_2+\theta_3\sum_{i=1}^D x_{ni}x_{mi}$$

### 6.4.5 Gaussian processes for classification
For the two-class problem if we define a Gaussian process over a function $a(x)$ and then transform the function using a logistic sigmoid $y=\sigma(a)$, then we obtain a non-Gaussian stochastic process over functions $y(x)$ where $y\in (0,1)$. The distribution of target variable $t$ is given by the Bernoulli distribution:

$$p(t|a) = \sigma(a)^{t}(1-\sigma(a))^{1-t}$$

Our goal is to determine the predictive distribution $p(t_{N+1}\|\textbf{t})$, so we introduce a Gaussian process prior over the vector $a_{N+1}$:

$$p(a_{N+1}) = N(a_{N+1} | 0, C_{N+1})$$

For numerical reasons it is convenient to introduce a noise-like term governed by a parameter $v$ that ensures the covariance matrix is positive definite. Thus the covariance matrix $C_{N+1}$ has elements given by:

$$C(x_{n}, x_{m}) = k(x_{n}, x_{m}) + \nu\delta_{nm}$$

The required predictive distribution is given by:

$$p(t_{N+1} = 1 | t_{N}) = \int p(t_{N+1} = 1 | a_{N+1})p(a_{N+1} | t_{N}) da_{N+1}$$

where $p(t_{N+1}=1\|a_{N+1})=\sigma(a_{N+1})$

### 6.4.6 Laplace approximation
In order to evaluate the predictive distribution, we seek a Gaussian approximation to the posterior distribution over $a_{N+1}$:

$$p(a_{N+1}|\textbf{t}_N)=\int p(a_{N+1},a_N|\textbf{t}_N)da_N=\int p(a_{N+1}|a_N)p(a_N|\textbf{t}_N)da_N$$

We also need to determine the parameters $\theta$ of the covariance function. The likelihood funciton is defined by:

$$p(\textbf{t}_N|\theta)=\int p(\textbf{t}_N|a_N)p(a_N|\theta)da_N$$

### 6.4.7 Connection to neural networks
