---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 2 Probability Distributions              # 标题 
subtitle:    #副标题
date:       2019-09-09          # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 

# PRML Chapter 2 Probability Distributions

## 2.1 Binary Variables
* bernoulli distribution: $Bern(x \| \mu) = \mu^{x}(1-\mu)^{1-x}$
* binomial distribution: $Bin(m \| N,\mu) = \frac{N!}{(N-m)!m!}\mu^{m}(1-\mu)^{N-m}$

### 2.1.1 The beta distribution
To get the MLE solution in a bayesian perspective, we need a prior distribution. Beta is a common one:

$$Beta(\mu | a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$

Where $E[\mu] = \frac{a}{a + b}$ and $var[\mu] = \frac{ab}{(a+b)^{2}(a + b + 1)}$. We can get the posterior distribution:

$$p(\mu | m,l,a,b) = \frac{\Gamma(m+a+l+b)}{\Gamma(m+a)\Gamma(l+b)}\mu^{m+a-1}(1-\mu)^{l+b-1}$$

For prediction, we need to estimate the distribution of $x$ given the condition that the training data has known.

$$p(x = 1 | D) = \int_{0}^{1}p(x=1 | \mu)p(\mu | D)d\mu = \int_{0}^{1}\mu p(\mu | D)d\mu = E[\mu | D]$$

So we have:

$$p(x = 1 | D) = \frac{m+a}{m+a+l+b}$$

On average, the more data we observe, the uncertainty of the posterior possibility will continuous decreasing.

$$E_{\theta}(\theta) = E_{D}[E_{\theta}[\theta | D]]$$

$$var_{\theta}[\theta] = E_{D}[var_{\theta}[\theta | D]] + var_{D}[E_{\theta}[\theta | D]]$$

## 2.2 Multinomial Variables
If we want to discribe a variable more than two states with binary variables, we can use the form like $x = (0, 0, \dots, 1, \dots, 0)^{T}$. If we use $\mu_k$ to represent the probability of $x_k=1$ and we have N independent dataset, the likelihood function will be:

$$p(D | \mu) = \prod_{k=1}^{K}\mu_{k}^{m_{k}},\ m_k=\sum_n x_{nk}$$

Using a Lagrange multiplier and let the partial derivative to find the MLE solution for $\mu$:

$$\sum_{k=1}^{K}m_{k}ln\mu_{k} + \lambda(\sum_{k=1}^{K}\mu_{k}-1)$$

$$\mu_{k}^{ML} = \frac{m_{k}}{N}$$

Now consider the conditional joint distribution, it is called multinomial distribution (here $\sum_{k=1}^{K}m_{k} = N$):

$$Mult(m_{1}, m_{2}, \dots, m_{K} |\mu, N) = \frac{N!}{m_{1}!m_{2}!\dots m_{K}!}$$

### 2.2.1 The Dirichlet Distribution
The prior distribution of parameter $\mu_k$ is Dirichlet distribution:

$$DIr(\mu | \alpha) = \frac{\Gamma(\alpha_{0})}{\Gamma(\alpha_{1})\dots\Gamma(\alpha_{K})} \prod_{k=1}^{K}\mu_{k}^{\alpha_{k-1}}$$

where $\alpha_{0} = \sum_{k=1}^{K}\alpha_{k}$ and it is easy to get the posterior distribution:

$$p(\mu | D,\alpha) = Dir(\mu | \alpha + m)$$

## 2.3 The Gaussian Distribution
The geometric form of Guassian distribution: The exponent is a quadratic form:

$$\Delta^{2} = (x-\mu)^{T}\Sigma^{-1}(x-\mu)$$

$\Sigma$ can be symmetric and from the eigenvector equation $\Sigma\mu_{i} = \lambda_{i}\mu_{i}$, when we choose the eigenvertor to be orthogonal, we will have:

$$\Sigma = \sum_{i=1}^{D}\lambda_{i}\mu_{i}\mu_{i}^{T}$$

So the quadratic form can be written as:

$$\Delta^{2}  = \sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}},\ y_{i} = \mu_{i}^{T}(x-\mu)$$

Consider the newly defined coordinate system by $y_i$, the form of Guassian distribution will be:

$$p(y) = p(x)|J| = \prod_{j=1}^{D}\frac{1}{(2\pi\lambda_{j})^{\frac{1}{2}}}exp\{-\frac{y_{j}^{2}}{2\lambda_{j}}\},\ \ J_{ij} = \frac{\partial x_{i}}{\partial y_{j}} = U_{ij}$$

### 2.3.1 Conditional Gaussian distributions
Consider multivariate normal distribution, suppose we have:

$$
x = \left(
    \begin{aligned}
    x_{a} \\
    x_{b} 
    \end{aligned}
    \right)
    ,\mu = \left(
    \begin{aligned}
    \mu_{a} \\
    \mu_{b} 
    \end{aligned}
    \right)
    ,% <![CDATA[
\Sigma = \left(
    \begin{aligned}
    \Sigma_{aa} ~&~ \Sigma_{ab}\\
    \Sigma_{ba} ~&~ \Sigma_{bb}
    \end{aligned}
    \right) %]]>
    $$

and we introduce the precision matrix:

$$\Sigma^{-1}=
\Lambda = \left(
    \begin{aligned}
    \Lambda_{aa} ~&~ \Lambda_{ab}\\
    \Lambda_{ba} ~&~ \Lambda_{bb}
    \end{aligned}
    \right) %]]>$$

To find an expression for the conditional distribution $p(x_a\|x_b)$, we obtain:

$$-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu) = 
-\frac{1}{2}(x_{a} - \mu_{a})^{T}\Lambda_{aa}(x_{a} - \mu_{a}) - \frac{1}{2}(x_{a} - \mu_{a})^{T}\Lambda_{ab}(x_{b} - \mu_{b}) \\
-\frac{1}{2}(x_{b} - \mu_{b})^{T}\Lambda_{ba}(x_{a} - \mu_{a}) - \frac{1}{2}(x_{b} - \mu_{b})^{T}\Lambda_{bb}(x_{b} - \mu_{b})$$

which is the exponential term of conditional Gaussian. Also, it is easy to know that:

$$-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu) = -\frac{1}{2}x^{T}\Sigma^{-1}x + x^{T}\Sigma^{-1}\mu + const$$

Apply this method to $p(x_a\|x_b)$ and $x_b$ is regarded as a constant. Compare the second order term $-\frac{1}{2}x_a^T\Lambda_{aa}x_a$ and the linear terms $x_a^T(\Lambda_{aa}\mu_{a} - \Lambda_{ab}(x_{b} - \mu_{b}))$ in $x_a$, we can get the variance and mean:

$$\Sigma_{a|b} = \Lambda_{aa}^{-1}$$

$$\mu_{a|b} = \Sigma_{a|b}(\Lambda_{aa}\mu_{a} - \Lambda_{ab}(x_{b} - \mu_{b}))\\
        = \mu_{a} - \Lambda_{aa}^{-1}\Lambda_{ab}(x_{b} - \mu_{b})$$

### 2.3.2 Marginal Gaussian distributions
To prove the marginal distribution $p(x_{a}) = \int p(x_{a}, x_{b})dx_{b}$ is Gaussian, the method is similar to 2.3.1, the results are:

$$\Sigma_{a} = (\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1}$$

$$\mu_{a} = \Sigma_{a}(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_{a}$$

### 2.3.3 Bayes' theorem for Gaussian variables
Suppose we are given a Gaussian marginal distribution $p(x)$ and a Gaussian conditional distribution $p(y\|x)$ (linear):
$$p(x) = N(x | \mu, \Lambda^{-1})$$

$$p(y | x) = N(y | Ax + b, L^{-1})$$

we wish to find the marginal distribution $p(y)$ and the conditional distribution $p(x\|y)$. Consider the log of the joint distribution:

$$
\begin{aligned}
ln p(z) & = ln p(x) + ln p(y | x) \\
        & = -\frac{1}{2}(x-\mu)^{T}\Lambda(x-\mu) - \frac{1}{2}(y-Ax-b)^{T}L(y-Ax-b) + C 
\end{aligned} %]]>$$

By comparing quadratic forms we can get the mean and covariance of the joint distribution:

$$
cov[z] = R^{-1} = \left(
    \begin{aligned}
    \Lambda^{-1} ~&~ \Lambda^{-1}A^{T} \\
    A\Lambda^{-1} ~&~ L^{-1} + A\Lambda^{-1}A^{T}
    \end{aligned}
    \right) %]]>$$

$$E[z] = \left(
    \begin{aligned}
    \mu \\
    A\mu + b
    \end{aligned}
    \right)$$

And for the marginal distribution $p(y)$, it is easy to observe that

$$E[y] = A\mu + b$$

$$cov[y] = L^{-1} + A\Lambda^{-1}A^{T}$$

Similarly, for conditional distribution, we have:

$$E[x | y] = (\Lambda + A^{T}LA)^{-1}\{A^{T}L(y-b) + a\mu \}$$

$$cov[x | y] = (\Lambda + A^{T}LA)^{-1}$$

### 2.3.4 Maximum likelihood for the Gaussian
This part we only show the result:

$$\mu_{ML}=\frac{1}{N}\sum_{n=1}^N x_n$$

$$\Sigma_{ML}=\frac{1}{N}\sum_{n=1}^N(x_n-\mu_{ML})(x_n-\mu_{ML})^T$$

### 2.3.5 Sequential estimation
Sequential methods allow data points to be processed one at a time and then discarded. Refer to 2.3.4, the contribution from the final data point $x_N$ is:

$$% <![CDATA[
\begin{aligned}
\mu_{ML}^{(N)}  & = \frac{1}{N}\sum_{n=1}^{N}x_{n} \\
  & = \mu_{ML}^{(N-1)} + \frac{1}{N}(x_{N} - \mu_{ML}^{(N-1)}) 
\end{aligned} %]]>$$

A more general formulation pair of sequential learning is called Robbins-Monro algorithm. For random variables $\theta$ and $z$ governed by joint distribution $p(z,\theta)$. The conditional expenctation of $z$ given $\theta$ is

$$f(\theta) = E[z | \theta] = \int zp(z | \theta) dz$$

$f(\theta)$ is called regression functions. Our goal is to find $\theta^*$ that $f(\theta^*)=0$. Suppose we observe values of $z$ and we with to find a corresponding sequential estimation scheme for $\theta^*$. Assume that $
E[(z-f)^{2}][\theta] < \infty %]]>$, the sequence of successive estimation will be:

$$\theta^{(N)} = \theta^{(N-1)} - \alpha_{N-1}z(\theta^{(N-1)})$$

### 2.3.6 Bayesian inference for the Gaussian

**Estimate $\mu$ ($\sigma$ known):**
The likelihood function of Gaussian distribution:

$$p(x | \mu) = \prod_{n=1}^{N}p(x_{n} | \mu) = \frac{1}{2\pi\sigma^{2}}exp\left\{-\frac{1}{2\sigma^{2}}\sum_{n=1}^{N}(x_{n} - \mu^{2})\right\}$$

Choose $p(\mu)$ as Gaussian distribution: $p(\mu) = N(\mu \| \mu_{0},\sigma_{0}^{2})$, and the posterior distribution is given by:

$$p(\mu | x) \propto p(x | \mu)p(\mu)$$

Consequently, the posterior distribution will be:

$$p(\mu | x) = N(\mu | \mu_{N}, \sigma_{N}^{2})$$

where

$$\mu_{N} = \frac{\sigma^{2}}{N\sigma_{0}^{2} + \sigma^{2}}\mu_{0} + \frac{N\sigma_{0}^{2}}{N\sigma_{0}^{2} + \sigma^{2}}\mu_{ML},\ \ \mu_{ML} = \frac{1}{N}\sum_{n=1}^{N}x_{n}$$

$$\frac{1}{\sigma_{N}^{2}} = \frac{1}{\sigma_{0}^{2}} + \frac{N}{\sigma^{2}}$$

The Bayesian paradigm leads very naturally to a sequential view of the inference problem.

$$p(\mu | x) \propto \left[ p(\mu)\prod_{n=1}^{N}p(x_{n} | \mu) \right] p(x_{N} | \mu)$$

**Estimate $\sigma$ ($\mu$ known)**:
The likelihood function for $\lambda$ takes the form:

$$p(x | \lambda) = \prod_{n=1}^{N}N(x_{n} | \mu, \lambda^{-1}) \propto \lambda^{\frac{1}{2}}exp\left\{ -\frac{\lambda}{2}\sum_{n=1}^{N}(x_{n} - \mu)^{2} \right\}$$

The prior distribution that we choose is Gamma distribution:

$$Gam(\lambda | a, ,b) = \frac{1}{\Gamma(a)}b^{a}\lambda^{(a-1)}exp(-b\lambda)$$

The posterior distribution is given by:

$$p(\lambda | x) \propto \lambda^{a_{0} - 1}\lambda^{\frac{N}{2}}exp\left\{ -b_{0}\lambda - \frac{\lambda}{2}\sum_{n=1}^{N}(x_{n}-\mu)^{2}\right\}$$

Consequently, the posterior distribution will be $Gam(\lambda \| a_{N}, b_{N})$, where

$$a_{N} = a_{0} + \frac{N}{2}$$

$$b_{N} = b_{0} + \frac{1}{2}\sum_{n=1}^{N}(x_{n} - \mu)^{2} = b_{0} + \frac{N}{2}\sigma_{ML}^{2}$$

**$\mu$ and $\sigma$ are unknown**:

The form of prior distribution we choose is:

$$p(\mu, \lambda) = N(\mu | \mu_{0}, (\beta\lambda)^{-1})Gam(\lambda | a, b)$$

Where $\mu_{0} = \frac{c}{\beta}$, $a = \frac{1+\beta}{2}$ and $b = d - \frac{c^{2}}{2\beta}$.

### 2.3.7 Student's t-distribution
If we have a gaussian distribution $N(x | \mu, \tau^{-1})$ and a Gamma Prior $Gam(\tau | a, b)$. Calculate the integral of $\tau$, we can get the marginal distribution of $x$:

$$p(x | \mu, a, b) = \int_{0}^{\infty} N(x | \mu, \tau^{-1})Gam(\tau | a, b)d\tau$$

Replace the parameters by  $v=2a$ and $\lambda=\frac{a}{b}$, we can get the Student's t-distribution:

$$St(x | \mu,\lambda, \nu) = \frac{\Gamma(\frac{\nu}{2} + \frac{1}{2})}{\Gamma(\frac{\nu}{2})}\left( \frac{\lambda}{\pi\nu}  \right)^{\frac{1}{2}}\left[ 1 + \frac{\lambda(x-\mu)^{2}}{\nu} \right]^{-\frac{\nu}{2}-\frac{1}{2}}$$

### 2.3.8 Periodic variables
### 2.3.9 Mixtures of Gaussians
Taking linear combinations of basic distributions such as Gaussian, almost any continuous density can be approximated to arbitrary accuracy.

$$p(x) = \sum_{k=1}^{K}\pi_{k}N(x | \mu_{k}, \Sigma_{k})$$

## 2.4 The Exponential Family
The distribution of variable $x$ with parameter $\eta$ can be defined as:

$$p(x | \eta) = h(x)g(\eta)exp\{\eta^{T}u(x)\}$$

And the formula satisfies:

$$g(\eta)\int h(x)exp\{\eta^{T}u(x)\} dx = 1$$

### 2.4.1 Maximum likelihood and sufficient statistics
### 2.4.2 Conjugate priors
For any member of the exponential family, there exists a conjugate a conjugate prior that can be written in the form:

$$p(\eta | \chi, \nu) = f(\chi, \nu)g(\eta)^{\nu}exp(\nu\eta^{T}\chi)$$

### 2.4.3 Noninformative priors
Two simple examples of noninformative priors:

The location parameter: $p(x\| \mu) = f(x-\mu)$

The scale parameter: $p(x \| \sigma) = \frac{1}{\sigma}f(\frac{x}{\sigma})$

## 2.5 Nonparametric Methods
