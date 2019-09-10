---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 3 Linear Models for Regression               # 标题 
subtitle:    #副标题
date:       2019-09-10             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 

# PRML Chapter 3 Linear Models for Regression

## 3.1 Linear Basis Function Models
The simplest linear model for regression is the form:

$$y(x, w) = w_{0} + \sum_{j=1}^{M-1}w_{j}\phi_{j}(x)$$

and when we define $\phi_{0}(x) = 1$, we will have:

$$y(x, w) = \sum_{j=1}^{M-1}w_{j}\phi_{j}(x) = w^{T}\phi(x)$$

There are many possible choices for the basis functions:

* powers of $x$: $\phi_{j}(x)=x^{j}$
* Gaussian: $\phi_{j}(x) = exp \{ -\frac{(x-\mu_{j})^{2}}{2s^{2}}\}$
* sigmoid: $\phi_{j}(x) = \sigma\left(\frac{x-\mu_{j}}{s}  \right)$
* logistic sigmoid: $\sigma_{a} = \frac{1}{1 + exp(-a)}$
* tanh: $tanh(a) = 2\sigma(2a) - 1$

### 3.1.1 Maximum likelihood and least squares
We assume that the target variable $t$ is given by a deterministic function $y(x,w)$ with additive Gaussian noise:

$$t = y(x,w) + \epsilon$$

So, we have

$$p(t | x, w, \beta) = N(t | y(x,w), \beta^{-1})$$

In the case of this Gaussian conditional distribution, the conditional mean will be

$$E[t|x] = \int tp(t | x)dt = y(x, w)$$

We assume that the data points are independent, and we can get the likelihood function and log likelihood function.

$$p(t|X, w, \beta) = \prod_{n=1}^{N}N(t_{n} | w^{T}\phi(x_{n}), \beta^{-1})$$

$$ln p(t | w, \beta) = \sum_{n=1}^{N} ln N(t_{n} | w^{T}\phi(x_{n}), \beta^{-1})$$

where the sum-of-squares error is $E_{D}(w) = \frac{1}{2}\sum_{n=1}^{N} \{ t_{n} - w^{T}\phi(x_{n}) \}^{2}$.

The MLE solution is:

$$w_{ML} = (\Phi^{T}\Phi)^{-1}\Phi^{T}t$$

this is called normal equations for the least squares problem. $\Phi$ is called the design matrix:

$$% <![CDATA[
\Phi = \left(
    \begin{aligned}
    \phi_{0}(x_{1}) && \phi_{1}(x_{1}) && \dots && \phi_{M-1}(x_{1}) \\
    \phi_{0}(x_{2}) && \phi_{1}(x_{2}) && \dots && \phi_{M-1}(x_{2}) \\
    \vdots          && \vdots          && \ddots&& \vdots            \\
    \phi_{0}(x_{N}) && \phi_{1}(x_{N}) && \dots && \phi_{M-1}(x_{N})
\end{aligned}
    \right) %]]>$$

and the $\Phi^{\dag} \equiv (\Phi^{T}\Phi)^{-1}\Phi^{T}$ is known as the Moore-Penrose pseudo-inverse of the matrix $\Phi$.

If we make the bias parameter explicit, then the error function above becomes:

$$E_{D}(w) = \frac{1}{2}\sum_{n=1}^{N} \{ t_{n} - w_{0} - \sum_{j=1}^{M-1}w_{j}\phi_{j}(x_{n}) \}^{2}$$

Setting the derivative with respect to $w_0$ equal to zero, we obtain:

$$w_{0} = \bar{t} - \sum_{j=1}^{M-1}w_{j}\bar{\phi}_{j}$$

where $\bar{t} = \frac{1}{N}\sum_{n=1}^{N}t_{n},\ \bar{\phi}_{j} = \frac{1}{N}\sum_{n=1}^{N}\phi_{j}(x_{n})$, thus the bias $w_0$ compensates for the difference between the averages of the target values and the weighted sum of the averages of the basis function values.

Also we can get the MLE solution of precision parameter.

$$\frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^{N}\{ t_{n} - w_{ML}^{T}\phi(x_{n}) \}^{2}$$

### 3.1.2 Geometry of least squares
### 3.1.3 Sequential learning
Using gradient descent algorithm updates the parameter vector:

$$w^{(\tau+1)}=w^{(\tau)}-\eta\nabla E_n$$

For the case of the sum-of-squares error function:

$$w^{(\tau+1)}=w^{(\tau)}+\eta(t_n-w^{(\tau)T}\phi_n)\phi_n$$

### 3.1.4 Regularized least squares
The total error function to be minimized takes the form:

$$E_D(w)+\lambda E_W(w)$$

Solving for $w$ as before, we obtain: $w=(\lambda I+\Phi^T \Phi)^{-1}\Phi^Tt$
Also we can consider the general regularized error:

$$\frac{1}{2}\sum_{n=1}^{N}\{ t_{n} - w^{T}\Phi(x_{n}) \}^{2} + \frac{\lambda}{2}\sum_{j=1}^{M}|w_{j}|^{q}$$

### 3.1.5 Multiple outputs
For multiple target variables, use the same set of basis functions to model all of the components of the target vector:

$$y(x,w) = W^{T}\phi(x)$$

Suppose the conditional distribution of the target vector is isotropic Gaussian: $p(t \| x, W, \beta) = N(t \| W^{T}\phi(x), \beta^{-1}I)$, we can get the MLE solution:

$$W_{ML} = (\Phi^{T}\Phi)^{-1}\Phi^{T}T$$

## 3.2 The Bias-Variance Decomposition
The conditional expectation can be written as:

$$h(x)=E[t|x]=\int tp(t|x)dt$$

The expected squared loss in 1.5.5 is

$$E[L] = \int \{y(x) - h(x)\}^{2}p(x)dx + \int \{h(x)-t\}^2p(x,t)dxdt$$

The expected squared difference between $y(x,D)$ and the regression function $h(x)$ can be expressed as the sum of two terms:

$$E_{D}[\{ y(x;D) - h(x) \}^{2} ] = \{ E_{D}[y(x;D)] - h(x) \}^{2} + E_{D}[\{y(x;D) - E_{D}[y(x;D)] \}^{2} ]$$

Generalize from the case of a single data point to the entire data set, we have:

$$expected\ loss=(bias)^2+variance+noise$$

where:

$$bias^{2} = \int \{E_{D}[y(x;D)] - h(x)\}^{2}p(x) dx$$

$$variance = \int E_{D}[\{y(x;D) - E_{D}[y(x;D)]\}^{2}]p(x) dx$$

$$noise = \int\int\{ h(x) - t\}^{2}p(x,t) dxdt$$

## 3.3 Bayesian Linear Regression
### 3.3.1 Parameter distribution
First we introduce the conjugate prior:

$$p(w) = N(w | m_{0}, S_{0})$$

It is easy to calculate the posterior distribution:

$$p(w|t) = N(w | m_{N}, S_{N})$$

where $m_{N} = S_{N}(S_{0}^{-1}m_{0} + \beta\Phi^{T}t)$ and
$S_{N}^{-1} = S_{0}^{-1} + \beta\Phi^{T}\Phi$

For the remainder of this chapter, we consider a zero-mean isotropic Gaussian prior in order to simplify the treatment.

$$p(w | \alpha) = N(w | 0, \alpha^{-1}I)$$

and the parameters in the posterior will be $m_{N} = \beta S_{N}\Phi^{T}t$ and $S_{N}^{-1} = \alpha I + \beta \Phi^{T}\Phi$.

### 3.3.2 Predictive distribution
The predictive distribution is defined by

$$p(t | \mathbf{t}, \alpha, \beta) = \int p(t | w, \beta)p(w | \mathbf{t}, \alpha, \beta) dw$$

It is the convolution of two Gaussian distributions, and the predictive distributions takes the form:

$$p(t | \mathbf{x}, \mathbf{t}, \alpha, \beta) = N(t | m_{N}^{T}\phi(x), \sigma_{N}^{2}(x))$$

where

$$\sigma_{N}^{2}(x) = \frac{1}{\beta} + \phi(x)^{T}S_{N}\phi(x)$$

### 3.3.3 Equivalent kernel
We can transform the predicted mean:

$$y(x, m_{N}) = m_{N}^{T}\phi(x) = \beta\phi(x)^{T}S_{N}\Phi^{T}t = \sum_{n=1}^{N}\beta\phi(x)^{T}S_{N}\phi(x_{n})t_{n}$$

Thus the mean of the predictive distribution at a point $x$ is given by a linear combination of the training set target variables $t_n$:

$$y(x,m_N)=\sum_{n=1}^N k(x,x_n)t_n$$

where the function $k(x, x^{'}) = \beta\phi(x)^{T}S_{N}\phi(x^{'})$ is known as the smoother matrix or the equivalent kernel.

The equivalent kernel shows the connection with the covariance:

$$cov[y(x),y'(x)]=cov[\phi(x)^Tw,w^T\phi(x')]=\phi(x)^TS_N\phi(x')=\beta^{-1}k(x,x')$$

Also, for all values of $x$, it satisfies:

$$\sum_{n=1}^N k(x,x_n)=1$$

What's more, it can be expressed in inner product form:

$$k(x,z)=\psi(x)^T\psi(z)$$

where $\psi(x)=\beta^{1/2}S_N^{1/2}\phi(x)$.

## 3.4 Bayesian Model Comparison
Suppose we wish to compare a set of L models $\{M_i\}$, here a model refers to a probability distribution over the observed data $D$. Given a training set $D$, we then wish to evaluate the posterior distribution:

$$p(M_{i} | D) \propto p(M_{i})p(D | M_{i})$$

$p(M_{i})p(D \| M_{i})$ means model evidence and the ratio of model evidences $\frac{p(D\|M_i)}{p(D \| M_j)}$ for two models is known as a Bayes factor.

The model evidence can be calculated through

$$p( D | M_{i} ) = \int p(D|w, M_{i})p(w|M_{i})dw$$

Assume that $p(w) = \frac{1}{\Delta w_{prior}}$, we will have:

$$p(D) = \int p(D |w)p(w)dw \simeq p(D|w_{MAP})\frac{\Delta w_{posterior}}{\Delta w_{prior}}$$

take the logarithm:

$$\ln p(D) \simeq \ln p(D | w_{MAP}) + \ln(\frac{\Delta w_{posterior}}{\Delta w_{prior}})$$

For model with M parameters, and their $\frac{\Delta w_{posterior}}{\Delta w_{prior}}$ are the same, we will have:

$$\ln p(D) \simeq p(D|w_{MAP}) + M\ln(\frac{\Delta w_{posterior}}{\Delta w_{prior}})$$

## 3.5 The Evidence Approximation
Introduce hyperpriors over $\alpha$ and $\beta$, the predictive distribution is obtained by marginalizing over $w,\alpha,\beta$.

$$p(t | \mathbf{t}) = \int\int\int p(t | w, \beta)p(w | \mathbf{t},\alpha,\beta)p(\alpha,\beta | \mathbf{t}) dw d\alpha d\beta$$

If the posterior distribution $p(\alpha,\beta \| \textbf{t})$ is sharply peaked around $\hat{\alpha}$ and $\hat{\beta}$, then the predictive distribution is obtained simply by marginalizing over $w$:

$$p(t | \mathbf{t}) \simeq p(t | \mathbf{t}, \hat{\alpha}, \hat{\beta}) = \int p(t|w,\hat{\beta})p(w | \mathbf{t}, \hat{\alpha}, \hat{\beta}))dw$$

From Bayes' theorem, the posterior distribution for $\alpha$ and $\beta$ is given by:

$$p(\alpha, \beta | \mathbf{t}) \propto p(\mathbf{t} | \alpha, \beta)p(\alpha, \beta)$$

### 3.5.1 Evaluation of the evidence function
The marginal likelihood function $p(\mathbf{t} \| \alpha, \beta)$ is obtained by integrating over the weight parameters $w$, so that

$$p(\mathbf{t} | \alpha, \beta) = \int p(\mathbf{t} | w, \beta)p(w| \alpha) dw$$

By completing the square in the exponent and making use of the standard form for the normalization coefficient of a Gaussian, we can get the log of the marginal likelihood in the form:

$$\ln p(\mathbf{t} | \alpha, \beta) = \frac{M}{2}\ln\alpha + \frac{N}{2}\ln\beta - E(m_{N}) - \frac{1}{2}\ln|A| - \frac{N}{2}\ln(2\pi)$$

### 3.5.2 Maximizing the evidence function
Defining the eigenvector equation

$$(\beta\Phi^{T}\Phi)u_{i} = \lambda_{i}u_{i}$$

It then follows that $A$ has eigenvalues $\alpha+\lambda_i$. Now consider the derivative of the term involving $ln\|A\|$ with respect to $\alpha$.

$$\frac{d}{d\alpha}\ln|A| = \frac{d}{d\alpha}\ln\prod_{i}(\lambda_{i} + \alpha) = \frac{d}{d\alpha}\sum_{i}\ln(\lambda_{i} + \alpha) = \sum_{i}\frac{1}{\lambda_{i} + \alpha}$$

Let the derivative be zero, we will have:

$$\alpha m_{N}^{T}m_{N} = M - \alpha\sum_{i}\frac{1}{\lambda_{i} + \alpha} = \gamma$$   

So it is easy to get that

$$\gamma = \sum_{i} \frac{\lambda_{i}}{\alpha + \lambda_{i}}\ \ and\ \ \alpha = \frac{\gamma}{m_{N}^{T}m_{N}}$$

As for $\beta$, we can follow the same idea to get

$$\frac{1}{\beta} = \frac{1}{N-\gamma}\sum_{n=1}^{N}\{ t_{n}-m_{N}^PT \phi(x_{n})\}$$

### 3.5.3 Effective number of parameters
When the number of data points is far more larger than parameters, it will be easy to compute that:

$$\alpha=\frac{M}{2E_W(m_N)}$$

$$\beta=\frac{N}{2E_D(m_N)}$$

## 3.6 Limitations of Fixed Basis Functions
