---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 9 Mixture Models and EM               # 标题 
subtitle:    #副标题
date:       2019-09-25             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 

# PRML Chapter 9 Mixture Models and EM

## 9.1 K-means Clustering
Consider the problem of identifying groups or clusters of data points in a multidimensional space.

To describe the assignment of data points to clusters, we define an objective function(distortion measure):

$$J = \sum_{n=1}^{N}\sum_{k=1}^{K}r_{nk}||x_{n}-\mu_{k}||^{2}$$

To find $r_{nk}$ and $\mu_k$ so as to minimize $J$, we go through an iterative procedure in which each iteration involves two successive steps corresponding to successive optimizations. The solution will be:

$$\mu_{k} = \frac{\sum_{n}r_{nk}x_{n}}{\sum_{n}r_{nk}}$$

We can also derive an on-line stochastic algorithm by applying the Robbins-Monro procedure to the problem

$$\mu_{k}^{new} = \mu_{k}^{old} + \eta_{n}(x_{n} - x_{k}^{old})$$

We can generalize the K-means algorithm by introducing a more general dissimilarity measure:

$$\tilde{J} = \sum_{n=1}^{N}\sum_{k=1}^{K}r_{nk}\nu(x_{n}, \mu_{k})$$

which gives the K-medoids algorithm.

### 9.1.1 Image segmentation and compression
## 9.2 Mixtures of Guassians
We have mentioned the Gaussian mixture model as a simple linear superposition of Gaussian components, we now turn to a formulation of Gaussian mixtures in terms of discrete latent variables.

$$p(x) = \sum_{k=1}^{K}\pi_{k}N(x | \mu_{k}, \Sigma_{k})$$

The marginal distribution over $z$ is specified in terms of mixing coefficients $\pi_k$

$$p(z_{k} = 1) = \pi_{k}$$

$$\sum_{k} z_{k} = 1$$

where the parameters $\{\pi_k\}$ must satisfy:

$$0 \leq \pi_{k} \leq 1$$

$$\sum_{k=1}^{K}\pi_k = 1$$

in order to be valid probabilities, because $z$ uses a 1-of-K representation, we can also write this distribution in the form:

$$p(z) = \prod_{k=1}^{K}\pi_{k}^{z_{k}}$$

Similarly, the conditional distribution of x is a Gaussian:

$$p(x | z) = \prod_{k=1}^{K}\mathcal{N}(x | \mu_{k}, \Sigma_{k})^{z_{k}}$$

The joint distribution is given by $p(z)p(x\|z)$ and the marginal distribution of $x$ is then obtained by summing the joint distribution over all possible states of $z$ to give:

$$p(x) = \sum_{z}p(z)p(x | z) = \sum_{k=1}^{K}\pi_{k}\mathcal{N}(x | \mu_{k}, \Sigma_{k})$$

The corresponding posterior probability will be:

$$\gamma(z_{k}) \equiv p(z_{k} = 1 | x) = \frac{p(z_{k}=1)p(x | z_{k}=1)}{\sum_{j=1}^{K}p(z_{j}=1)p(x | z_{j} = 1)} = \frac{\pi_{k}\mathcal{N}(x | \mu_{k}, \Sigma_{k})}{\sum_{j=1}^{K}\pi_{j}\mathcal{N}(x | \mu_{j},\Sigma_{k})}$$


### 9.2.1 Maximum likelihood 
The log of the likelihood function is given by:

$$\ln p(\mathbf{X} | \pi, \mu, \Sigma) = \sum_{n=1}^{N}\ln\left\{\sum_{k=1}^{K}\pi_{k}\mathcal{N}(x_{n} | \mu_{k}, \Sigma_{k}) \right\}$$

### 9.2.2 EM for Gaussian mixtures
An elegant and powerful method for finding maximum likelihood solutions for models with latent variables is called the espectation-maximization algorithm (EM algorithm).

Maximize the likelihood function, we obtain:

$$\mu_{k} = \frac{1}{N_{k}}\sum_{n=1}^{N}\gamma(z_{nk})x_{n}$$

where $N_{k} = \sum_{n=1}^{N}\gamma(z_{nk})$.

And for the covariance matrix of a single Gaussian, we obtain:

$$\Sigma_{k} = \frac{1}{N_{k}}\sum_{n=1}^{N}\gamma(z_{nk})(x_{n} - \mu_{k})^{T}$$

For the mixing coefficients (using Lagrange multiplier) we obtain:

$$\pi_{k} = \frac{N_{k}}{N}$$

**EM for Gaussian Mixtures**

So, given a Gaussian mixture model, the goal is to maximize the likelihood function with respect to the parameters (mean & covariances of the components & mixing coefficients).

* 1. Initialize $\mu_k$, $\Sigma$, $\pi_k$. Evaluate the initial value of the log likelihood.

* 2. **E step**. Evaluate the responsibilities using the current parameter values.

$$\gamma(z_{nk}) = \frac{\pi_{k}\mathcal{N}(x_{n} | \mu_{k}, \Sigma_{k})}{\sum_{j=1}^{K}\pi_{j}\mathcal{N}(x_{n} | \mu_{j}, \Sigma_{j})}$$

* 3. **M step**. Re-estimate the parameters using the current responsibilities.

$$\mu_{k}^{new} = \frac{1}{N_{k}}\sum_{n=1}^{N}\gamma(z_{nk})x_{n}$$

$$\sum_{k}^{new} = \frac{1}{N_{k}}\sum_{n=1}^{N}\gamma(z_{nk})(x_{n} - \mu_{k}^{new})(x_{n} - \mu_{k}^{new})^{T}$$

$$\pi_{k}^{new} = \frac{N_{k}}{N}$$

where

$$N_{k} = \sum_{n=1}^{N}\gamma(z_{nk})$$

4. Evaluate the log likelihood

$$\ln p(\mathbf{X} | \mu, \Sigma, \pi) = \sum_{n=1}^{N}\ln\left\{\sum_{k=1}^{K}\pi_{k}\mathcal{N}(x_{n} | \mu_{k}, \Sigma_{k})  \right\}$$

and check for convergence of either the parameters of the log likelihood. If the convergence criterion is not satisfied return to step 2.

## 9.3 An Alternative View of EM
The goal of the EM algorithm is to find maximum likelihood solutions for models having latent variables. Set the parameters $\theta$, and the likelihood function is given by:

$$\ln p(\mathbf{X}| \theta) = \ln\left\{\sum_{Z}p(\mathbf{X,Z} | \theta)  \right\}$$

In the E step, we use current $\theta^{old}$ to find the posterior by which we can find the expectation of the complete-data log likelihood evaluated for some general parameter value $\theta$:

$$\theta^{new} = \arg\max_{\theta}\mathcal{Q}(\theta, \theta^{old})$$

In the M step, we determine the revised parameter estimate $\theta^{new}$ by maximizing this function:

$$\mathcal{Q}(\theta, \theta^{old}) = \sum_{Z}p(\mathcal{Z} | \mathcal{X}, \theta^{old})\ln p(\mathcal{X, Z} | \theta)$$

### 9.3.1 Guassian mixtures revisited
The log likelihood function takes the form:

$$\ln p(\mathbf{X, Z} | \mu, \Sigma. \pi) = \sum_{n=1}^{N}\sum_{k=1}^{K}z_{nk}\{\ln \pi_{k} + \ln\mathcal{N}(x_{n} | \mu_{k}, \Sigma_{k}) \}$$

Using Bayes' theorem, we see that the posterior distribution takes the form:

$$p(Z | X, \mu, \Sigma, \pi) \propto \prod_{n=1}^{N}\prod_{k=1}^{K}[\pi_{k}\mathcal{N}(x_{n} | \mu_{k}, \Sigma_{k}) ]^{z_{nk}}$$

The expected value of the complete-data log likelihood function is therefore given by:

$$E_{Z}[\ln p(X, Z | \mu, \Sigma, \pi)] = \sum_{n=1}^{N}\sum_{k=1}^{K}\gamma(z_{nk})\{\ln \pi_{k} + \ln\mathcal{N}(x_{n} | \mu_{k}, \Sigma_{k})  \}$$

### 9.3.2 Relation to K-means
The posterior probabilities for a particular data point $x_n$ are given by:

$$\gamma(z_{nk})=\frac{\pi_k exp(-||x_n-\mu_k||^2/2\varepsilon)}{\sum_j \pi_j exp(-||x_n-\mu_j||^2/2\varepsilon)}$$

### 9.3.3 Mixtures of Bernoulli distributions
In order to derive the EM algorithm, we first write down the complete-data log likelihood function, which is given by:

$$\ln p(\mathbf{X, Z} | \mu, \pi) = \sum_{n=1}^{N}\sum_{k=1}^{K}z_{nk} \left\{\ln \pi_{k} + \sum_{i=1}^{D}[x_{ni}\ln\mu_{ki} + (1-x_{ni})\ln(1-\mu_{ki})]   \right\}$$

where $\mathbf{X}=\{x_n\}$ and $\mathbf{Z}=\{z_n\}$. Next we take the expectation of the complete-data log likelihood with respect to the posterior distribution of the latent variables to give:

$$E_{Z}[\ln p(\mathbf{X, Z} | \mu, \pi)] = \sum_{n=1}^{N}\sum_{k=1}^{K}\gamma(z_{nk})\left\{\ln\pi_{k} + \sum_{i=1}^{D}[x_{ni}\ln\mu_{ki} + (1-x_{ni})\ln(1-\mu_{ki})]  \right\}$$

### 9.3.4 EM for Bayesian linear regression
The complete-data log likelihood function is then given by:

$$ln p(\mathbf{t},w|\alpha,\beta)=lnp(\mathbf{t}|w,\beta)+lnp(w|\alpha)$$

Taking the expectation with respect to the posterior distribution of $w$ gives:

$$E[ln p(\mathbf{t},w|\alpha,\beta)]=\frac{M}{2}ln(\frac{\alpha}{2\pi})-\frac{\alpha}{2}E[w^T w]+\frac{N}{2}ln(\frac{\beta}{2\pi})-\frac{\beta}{2}\sum_{n=1}^NE[(t_n-w^T\phi_n)^2]$$

Setting the derivatives with respect to $\alpha$ or $\beta$ to zero, we obtain the M step re-estimation equation:

$$\alpha=\frac{M}{E[w^T w]}=\frac{M}{m^T_N m_N+Tr(S_N)}$$

$$\frac{1}{\beta}=\frac{1}{N}(||\mathbf{t}-\mathbf{\Phi_{m_N}}||^2+Tr[\mathbf{\Phi^T\Phi}S_N])$$

## 9.4 The EM Algorithm in General
Our goal is to maximize the likelihood function:

$$p(X|\theta)=\sum_Z p(X,Z|\theta)$$

Next we introduce a distribution $q(Z)$ defined over the latent variables. For any choice of $q(Z)$, the following decomposition holds:

$$\ln p(X | \theta) = \mathcal{L}(q, \theta) + KL(q||p)$$

where we have defined

$$\mathcal{L}(q, \theta) = \sum_{Z}q(Z)\ln\left\{\frac{p(X,Z | \theta)}{q(Z)}  \right\}$$

$$KL(q||p) = -\sum_{Z}q(Z)\ln \left\{\frac{p(Z | X, \theta)}{q(Z)}  \right\}$$
