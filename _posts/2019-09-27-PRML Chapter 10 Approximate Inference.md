---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 10 Approximate Inference               # 标题 
subtitle:   #副标题
date:       2019-09-27             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - PRML
--- 

# PRML Chapter 10 Approximate Inference
## 10.1 Variational Inference
For observed variable $X=\{x_1,...,x_N\}$ and latent $Z=\{z_1,...,z_N\}$. Our probabilistic model specifies the joint distribution $p(Z\|X)$ and our goal is to find an approximation for the posterior distribution $p(Z\|X)$ as well as for the model evidence $p(X)$. As in our discussion of EM, we can decompose the log marginal probability using:

$$\ln p(X) = \mathcal{L}(q) + KL(q || p)$$

where

$$\mathcal{L}(q) = \int q(Z) \ln\left\{ \frac{p(X, Z)}{q(Z)} \right\} dZ$$

$$KL(q || p) = -\int q(Z)\ln \left\{p(Z | X)q(Z)  \right\} dZ$$

### 10.1.1 Factorized distributions
Suppose we partition the elements of $Z$ into disjoint groups and the q factorizes with respect to there groups:

$$q(Z) = \prod_{i=1}^{M}q_{i}(Z_{i})$$

substitute it to the fomula above and denoting $q_j(Z_j)$ by simply $q_j$ we obtain:

$$\mathcal{L}(q) = \int q_{j}\ln \tilde{p}(X, Z_{j}) - \int q_{j}\ln q_{j}dZ_{j} + const$$

where we have defined a new distribution $\tilde{p}(X, Z_{j})$ by the relation:

$$\ln \tilde{p}(X, Z_{j}) = E_{i\neq j}[\ln p(X, Z)] + const$$

$$E_{i\neq j}[\ln p(X, Z)] = \int \ln p(X,Z)\prod_{i\neq j}q_{i}dZ_{i}$$

maximizing $\mathcal{L}(q)$ is equivalent to minimizing the Kullback-Leibler divergence, and the minimum occurs when $q_j(Z_j)=\tilde{p}(X,Z_j)$. Thus we obtain the generl expression for the optimal solution $q_j^*(Z_j)$ given by:

$$\ln q^{*}_{j}(Z_{j}) = E_{i\neq j}[\ln p(X, Z)] + const$$

### 10.1.2 Properties of factorized approximations
Two forms of Kullback-Leibler divergence are members of the alpha family of divergences defined by:

$$D_\alpha (p||q)=\frac{4}{1-\alpha^2}(1-\int p(x)^{(1+\alpha)/2}q(x)^{(1-\alpha)/2}dx)$$

### 10.1.3 Example: The univariate Gaussian
### 10.1.4 Model comparison
We can readily verify the following decomposition based on this variational distribution:

$$lnp(X)=\mathcal{L}_m-\sum_m\sum_Z q(Z|m)q(m)ln(\frac{p(Z,m|X)}{q(Z|m)q(m)})$$

where $\mathcal{L}_m$ is a lower bound on $lnp(X)$ and is given by

$$\mathcal{L}_m=\sum_m\sum_Z q(Z|m)q(m)ln(\frac{p(Z,X,m)}{q(Z|m)q(m)})$$

We can maximizing $\mathcal{L}$ with respect to the distribution $q(m)$ using a Lagrange multiplier, with the result:

$$q(m)\propto p(m)exp(\mathcal{L}_m)$$

## 10.2 Illustration: Variational Mixture of Gaussians
The conditional distribution of $Z$, given the mixing coefficients $\pi$, in the form:

$$p(Z | \pi) = \prod_{n=1}^{N} \prod_{k=1}^{K} \pi_{k}^{z_{nk}}$$

Similarly, the conditional distribution of the observed data vectors, given the latent variables and the component parameters:

$$p(X | Z, \mu, \Lambda) = \prod_{n=1}^{N} \prod_{k=1}^{K}\mathcal{N}(x_{n} | \mu_{k}, \Lambda_{k}^{-1})^{z_{nk}}$$

Choose a Dirichlet distribution over the mixing coefficients $\pi$:

$$p(\pi)= Dir(\pi | \alpha_{0})= C(\alpha_{0})\prod_{k=1}^{K}\pi_{k}^{\alpha_{0} - 1}$$

Introduce an independent Gaussian-Wishart prior governing the mean and precision of each Gaussian component, give by:

$$p(\mu, \Lambda) = p(\mu | \Lambda)p(\Lambda) = \prod_{k=1}^{K}\mathcal{N}(\mu_{k} | m_{0}, (\beta_{0}\Lambda)^{-1})\mathcal{W}(\Lambda_{k} | W_{0}, \nu_{0})$$

### 10.2.1 Variational distribution
In order to formulate a variational treatment of this model, we write down the joint distribution of all of the random variables:

$$p(X, Z, \pi, \mu, \Lambda) = p(X | Z, \mu, \Lambda)p(Z | \pi)p(\pi)p(\mu | \Lambda)p(\Lambda)$$

Consider a variational distribution which factorizes between the latent variables and the parameters as that:

$$q(Z, \pi, \mu, \Lambda) = q(Z)q(\pi, \mu, \Lambda)$$

Let us consider the derivation of the update equation for the factor $q(Z)$. The log of the optimized factor is given by:

$$\ln q^{*}(Z) = E_{\pi, \mu,\Lambda}[\ln p(X, Z, \pi, \mu, \Lambda)] + const$$

Absorb terms do not depend on $Z$ into the constant:

$$\ln q^{*}(Z) = E_\pi[\ln p(Z | \pi)] + E_{\mu, \Lambda}[\ln p(X | Z,\mu, \Lambda)] + const$$

Substituting for the two conditional distributions on the right-hand side and absorb terms independent of $Z$ into the constant, we have:

$$ln q^*(Z)=\sum_{n=1}^N\sum_{k=1}^K z_{nk}ln\rho_{nk}+const$$

where we have defined:

$$\ln \rho_{nk} = E[\ln \pi_{k}] + \frac{1}{2}E[\ln|\Lambda_{k}|] - \frac{D}{2}\ln(2\pi) - \frac{1}{2}E_{\mu_{k}, \Lambda_{k}}[(x_{n} - \mu_{k})^{T}\Lambda_{k}(x_{n} - \mu_{k})]$$

Taking the exponential of both sides we obtain:

$$q^{*}(Z) \propto \prod_{n=1}^{N}\prod_{k=1}^{K}\rho_{nk}^{z_{nk}}$$

Normalized the distribution we obtain:

$$q^{*}(Z) = \prod_{n=1}^{N}\prod_{k=1}^{K} r_{nk}^{z_{nk}}$$

where:

$$r_{nk} = \frac{\rho_{nk}}{\sum_{j=1}^{K}\rho_{nj}}$$

For the distribution $q^*(Z)$ we have the standard result:

$$E[z_{nk}] = r_{nk}$$

Let us consider the factor $q(\pi, \mu, \Lambda)$ in the variational posterior distribution. We have:

$$\ln q^{*}(\pi, \mu, \Lambda) = \ln p(\pi) \sum_{k=1}^{K}\ln p(\mu_{k}, \Lambda_{k}) + E_{Z}[\ln p(Z | \pi)] + \sum_{k=1}^{K}\sum_{n=1}^{N}E[z_{nk}]\ln \mathcal{N}(x_{n} | \mu_{k}, \Lambda_{k}^{-1}) + const$$

The variational approximation will be:

$$q(\pi,\mu,\Lambda)=q(\pi)\prod_{k=1}^kq(\mu_k,\Lambda_k)$$

The results are given by:

$$q^{*}(\pi) = Dir(\pi | \alpha)$$

where $\alpha$ has components $\alpha_k$ given by:

$$\alpha_{k} = \alpha_{0} + \sum_{n=1}^{N}r_{nk}$$

Using $q^* (\mu_{k}, \Lambda_{k}) = q^* (\mu_{k} \| \Lambda_{k})q^* (\Lambda_{k})$, the posterior distribution is a Gaussian-Wishart distribution and is given by:

$$q^{*}(\mu_{k}, \Lambda_{k}) = \mathcal{N}(\mu_{k} | m_{k}, (\beta_{k}\Lambda_{k})^{-1})\mathcal{W}(\Lambda_{k} | W_{k}, \nu_{k})$$

where

$$\beta_{k} = \beta_{0} + N_{0}$$

$$m_{k} = \frac{1}{\beta_{k}}(\beta_{0}m_{0} + N_{k}\hat{x}_{k})$$

$$W_{k}^{-1} = W_{0}^{-1} + N_{k}S_{k} + \frac{\beta_{0}N_{k}}{\beta_{0} + N_{k}}(\hat{x}_{k} - m_{0})(\hat{x}_{k} - m_{0})^{T}$$

$$\nu_{k} = \nu_{0} + N_{k}$$

### 10.2.2 Variational lower bound
For the variational mixture of Gaussians, the lower bound is given by:

$$\mathcal{L} = \sum_{Z}q(Z, \pi, \mu, \Lambda)\ln\left\{\frac{p(X, Z, \pi, \mu, \Lambda)}{q(Z, \pi, \mu, \Lambda)}\right\} d\pi d\mu d\Lambda$$

### 10.2.3 Predictive density
Wihe a new value $\hat{x}$ the predictive density is given by:

$$p(\hat{x} | X) = \sum_{\hat{Z}}\int\int\int p(\hat{X} | \hat{z},\mu, \Lambda)p(\hat{z} | \pi)p(\pi, \mu, \Lambda | X)d\pi d\mu d\Lambda$$

### 10.2.4 Determining the number of components
### 10.2.5 Induced factorizations

## 10.3 Variational Linear Regression
The joint distribution of all the variables is given by:

$$p(t,w,\alpha)=p(t|w)p(w|\alpha)p(\alpha)$$

where

$$p(t|w)=\prod_{n=1}^N N(t_n|w^T\phi_n,\beta^{-1})$$

$$p(w|\alpha)=N(w|0,\alpha^{-1}I)$$

$$p(\alpha)=Gam(\alpha|a_0,b_0)$$

### 10.3.1 Variational distribution
Our first goal is to find an approximation to the posterior distribution $p(w,\alpha\|\mathbf{t})$. The variational posterior distribution is given by the factorized expression:

$$q(w,\alpha)=q(w)q(\alpha)$$

The result will be:

$$q^*(\alpha)=Gam(\alpha|a_N,b_N)$$

where

$$a_N=a_0+\frac{M}{2}$$

$$b_N=b_0+\frac{1}{2}E[w^Tw]$$

and the distribution $q^*(w)$ is Gaussian:

$$q^*(w)=N(w|m_N,S_N)$$

where

$$m_N=\beta S_N\Phi^T t$$

$$S_N=(E[\alpha]I+\beta\Phi^T\Phi)^{-1}$$

### 10.3.2 Predictive distribution
The predictive distribution over $t$, given a new input $x$ is evaluated using the Gaussian variational posterior:

$$p(t|x,\mathbf{t})\simeq N(t|m^T_N\phi(x),\sigma^2(x))$$

where the input-dependent variance is given by

$$\sigma^2(x)=\frac{1}{\beta}+\phi(x)^TS_N\phi(x)$$

### 10.3.3 Lower bound
## 10.4 Exponential Family Distributions
Suppose the joint distribution of observed and latent variables is a member of the exponential family, parameterized by natural parameters $\eta$ so that:

$$p(X,Z|\eta)=\prod_{n=1}^N h(x_n,z_n)g(\eta)exp\{\eta^Tu(x_n,z_n)\}$$

we shall also use a conjugate prior for $eta$, which can be written as:

$$p(\eta|v_0)=f(v_0,x_0)g(\eta)^{v_0}exp\{v_0\eta^Tx_0\}$$

Now consider a variational distribution that factorizes between the latent variables and the parameters, so that $q(Z,\eta)=q(Z)q(\eta)$. The result will be

$$q^*(z_n)=h(x_n,z_n)g(E[\eta])exp\{E[\eta^T]u(x_n,z_n)\}$$

$$q^*(\eta)=f(v_N,x_N)g(\eta)^{v_N}exp\{\eta^Tx_N\}$$

where

$$v_N=v_o+N$$

$$x_N=x_0+\sum_{n=1}^N E_{z_n}[u(x_n,z_n)]$$

### 10.4.1 Variational message passing
The joint distribution corresponding to a directed graph can be written using the decomposition:

$$p(x)=\prod_i p(x_i|pa_i)$$

Now consider a variational approximation in which the distribution $q(x)$ is assumed to factorize with respect to the $x_i$ so that:

$$q(x)=\prod_i q_i(x_i)$$

Substitute the formula above into the general result we will get:

$$lnq_j^*(x_j)=E_{i\not =j}[\sum_i lnp(x_i|pa_i)]+const$$

## 10.5 Local Variational Methods
For convex funcitons, we can obtain upper bounds by:

$$g(\eta) = - \min_{x}\{ f(x) - \eta x \} = \max_{x}\{ \eta x - f(x) \}$$

$$f(x) = \max_{\eta}\{ \eta x - g(\eta) \}$$

And for concave functions:

$$f(x) = \min_{\eta}\{ \eta x- g(\eta) \}$$

$$g(\eta) = \min_{x}\{ \eta x - f(x) \}$$

If the function is not convex or concave, then we need the invertible transformations. An example will be logistic sigmoid function:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

The result with that of the logistic sigmoid will be:

$$\sigma(x) \leq exp(\eta x - g(\eta))$$

$$\sigma(x) \geq \sigma(\xi)exp\left\{ \frac{x-\xi}{2} - \lambda(\xi)(x^{2} - \xi^{2}) \right\}$$

where:

$$\lambda(\xi) = -\frac{1}{2\xi}\left[ \sigma(\xi) - \frac{1}{2} \right]$$

We cna see how the bounds can be used, suppose we wish to evaluate an integral of the form:

$$I = \int \sigma(a)p(a) da$$

We can employ the variational bound and we will get:

$$I \geq \int f(a, \xi)p(a) da = F(\xi)$$

## 10.6 Variational Logistic Regression
### 10.6.1 Variational posterior distribution
In the variational framework, we seek to maximize a lower bound on the marginal likelihood. For the Bayesian logistic regression model, the marginal likelihood takes the form:

$$p(t) = \int p(t | w)p(w)dw = \int \left[\prod_{n=1}^{N}p(t_{n} | w) \right]p(w) dw$$

The conditional distribution for $t$ can be written as

$$p(t | w) = e^{at}\sigma(-a)$$

where $a = w^{T}\phi$. Using the variational lower bound on the logistic sigmoid function we can get:

$$p(t | w) = e^{at}\sigma(-a)\geq e^{at}\sigma(\xi)exp\left\{ -\frac{a + \xi}{2} - \lambda(\xi)(a^{2} - \xi^{2})  \right\}$$

Using $a=w^T\phi$, and multiplying by the prior distribution, we obtain the bound on the joint distribution of $t$ and $w$:

$$p(t | w) = p(t | w)p(w) \geq h(w, \xi)p(w)$$

where

$$h(w, \xi) = \prod_{n=1}^{N}\sigma(\xi_{n})exp\{w^{T}\phi_{n}t_{n} - (w^{T}\phi_{n} + \xi_{n})/2  - \lambda(\xi_{n})([w^{T}\phi_{n}] - \xi^{2}_{n}) \}$$

The Gaussian variational posterior will be the form:

$$q(w) = \mathcal{N}(w | m_{N}, S_{N})$$

where

$$m_{N} = S_{N}\left(  S_{0}^{-1}m_{0} + \sum_{n=1}^{N}(t_{n} - \frac{1}{2}) \phi_{n} \right)$$

$$S_{N}^{-1} = S_{0}^{-1} + 2\sum_{n=1}^{N}\lambda(\xi_{n})\phi_{n}\phi_{n}^{T}$$

### 10.6.2 Optimizing the variational parameters
Substitute the inequality above back into the marginal likelihood to give:

$$\ln p(t) = \ln\int p(t | w)p(w)dw \geq \ln \int h(w, \xi)p(w) dw = \mathcal{L}(\xi)$$

There are two approaches to determining the $\xi_n$: EM algorithm and integrate over $w$.

### 10.6.3 Inference of hyperparameters
## 10.7 Expectation Propagation
