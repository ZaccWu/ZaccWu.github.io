---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 8 Graphical Models              # 标题 
subtitle:    #副标题
date:       2019-10-01             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png   #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - PRML
--- 

# PRML Chapter 8 Graphical Models

## 8.1 Bayesian Networks
A specific graph can make probabilistic statements for a broad class of distributions. We can writh the joint distribution in the form:

$$p(a, b, c) = p(c|a,b)p(a,b)=p(c|a, b)p(b | a)p(a)$$

For a graph with K nodes, the joint distribution is given by:

$$p(x) = \prod_{k=1}^{K} p(x_{k} | pa_{k})$$

### 8.1.1 Example: Polynomial regression
The joint distribution can be written as:

$$p(\mathbf{t},w)=p(w)\prod_{n=1}^Np(t_n|w)$$

It may be useful to make the parameters and stochastic variables explicit:

$$p(\mathbf{t},w | \mathbf{x}, \alpha, \sigma^{2}) = p(w|\alpha)\prod_{n=1}^{N}p(t_{n} | w, x_{n}, \sigma^{2})$$

To calculate the posterior distribution of $w$, we note that:

$$p(w|\textbf{T})\propto p(w)\prod_{n=1}^Np(t_n|w)$$

To predict $\hat{t}$ for a new input value $\hat{x}$ we write down the joint distribution of all the random variables conditioned on the deterministic parameters:

$$p(\hat{t},\mathbf{t},w|\hat{x},\mathbf{x},\alpha,\sigma^2)=[\prod_{n=1}^N p(t_n|x_n,w,\sigma^2)]p(w|\alpha)p(\hat{t}|\hat{x},w,\sigma^2)$$

The required predictive distribution for $\hat{t}$ is then obtained:

$$p(\hat{t}|\hat{x},\mathbf{x},\mathbf{t},\alpha,\sigma^2)\propto\int p(\hat{t},\mathbf{t},w|\hat{x},\mathbf{x},\alpha,\sigma^2)dw$$

### 8.1.2 Generative models
### 8.1.3 Discrete variables
The probability distribution $p(x\|\mu)$ for a single discrete variable $x$ having K possible states is given by:

$$p(x|\mu)=\prod_{k=1}^K \mu_k^{x_k}$$

Suppose we have two discrete variables and each of them has K states:

$$p(x_1,x_2|\mu)=\prod_{k=1}^K\prod_{l=1}^K \mu_{kl}^{x_{1k}x_{2l}}$$

We can use parameterized models for the conditional distributions, a more parsimonious form for the conditional distribution can be obtained by using logistic sigmoid function:

$$p(y=1|x_1,...,x_M)=\sigma (w_0+\sum_{i=1}^Mw_i x_i)=\sigma(w^T x)$$

### 8.1.4 Linear-Gaussian models
Suppose a single continuous random variable $x_i$ having a Gaussian distribution:

$$p(x_{i} | pa_{i}) = N(x_{i} | \sum_{j\in pa_{i}} w_{ij}x_{j} + b_{i}, v_{i})$$

The log of the distribution is the log of the product of conditionals over all nodes:

$$\ln p(x) = \sum_{i=1}^{D}p(x_{i} | pa_{i}) = -\sum_{i=1}^{D}\frac{1}{2v_{i}}\left( x_{i} - \sum_{j\in pa_{i}} w_{ij}x_{j} - b_{i}  \right)^{2} + Const$$

Each variable $x_i$ has a Gaussian distribution and we can get:

$$x_i=\sum_{j\in pa_i}w_{ij}x_j+b_i+\sqrt{v_i}\epsilon_i$$

where $\epsilon_i$ satisfies: $E[\epsilon_i]=0$ and $E[\epsilon_i\epsilon_j]=I_{ij}$. We can then get the mean and covariance of the joint distribution:

$$E[x_{i}] = \sum_{j\in pa_{i}} w_{ij}E[x_{j}] + b_{i}$$

$$cov[x_{i}, x_{j}] = \sum_{k\in pa_{j}} w_{jk}cov[x_{i}, x_{k}] + I_{ij}v_{j}$$

## 8.2 Conditional Independence
We say that $a$ is conditionally independent of $b$ given $c$ if we have:

$$p(a | b, c) = p(a | c)$$

We use a shorthand notation for conditional independence in which:

$$a ⊥⊥ b | c$$

### 8.2.1 Three example graphs

* Tail-to-Tail Node
* Head-to-Tail Node
* Head-to-Head Node

### 8.2.2 D-separation
Suppose we condition on $\mu$ and the joint distribution of the observations can be written as:

$$p(D|\mu)=\prod_{n=1}^N p(x_n|\mu)$$

The conditional independence properties can also explored by Markov blanked or Markov boundary, a conditional distribution can be expressed in the form:

$$p(x_i|x_{\{j\not =i\}})=\frac{p(x_1,...,x_D)}{\int p(x_1,...,x_D)dx_i}=\frac{\prod_k p(x_k|pa_k)}{\int \prod_k p(x_k|pa_k)dx_i}$$

## 8.3 Markov Random Fields
### 8.3.1 Conditional independence properties
### 8.3.2 Factorization properties
The joint distribution can be written as a product of potential functions $psi_C(x_C)$ over the maximal cliques of the graph:

$$p(x) = \frac{1}{Z}\prod_{C}\psi_{C}(x_{C})$$

$Z$ is called the partition function and it acts as a normalization constant and is given by:

$$Z = \sum_{x}\prod_{C}\psi_{C}(x_{C})$$

The potential functions can be expressed by energy function:

$$\psi_{C}(x_{C}) = exp\{-E(x_{C})\}$$

### 8.3.3 Illustration: Image de-noising
The complete energy function for the model takes the form:

$$E(x,y)=h\sum_ix_i-\beta\sum_{\{i,j\}}x_ix_j-\eta\sum_ix_iy_i$$

which defines a joint distribution over $x$ and $y$ given by:

$$p(x,y)=\frac{1}{Z}exp\{-E(x,y)\}$$

### 8.3.4 Relation to directed graphs

## 8.4 Inference in Graphical Models
### 8.4.1 Inference on a chain
The joint distribution for a graph takes the form:

$$p(x) = \frac{1}{Z}\psi_{1,2}(x_{1}, x_{2})\psi_{x_{2}, x_{3}}(x_{2}, x_{3})\dots\psi_{N-1, N}(x_{N-1}, x_{N})$$

where for the marginal distribution for a specific node $x_n$:

$$p(x_n)=\sum_{x_1}...\sum_{x_{n-1}}\sum_{x_{n+1}}...\sum_{x_N}p(x)$$

A more efficient algorithm by exploiting the conditional independece properties of the graphical model, the marginal expression can be expresed as:

$$
\begin{aligned}
p(x_{n}) = \frac{1}{Z} & \left[ \sum_{x_{n}-1}\psi_{n-1, n}(x_{n-1}, x_{n}) \dots \left[ \sum_{x_{2}}\psi_{2, 3}(x_{2}, x_{3})\left[ \sum_{x_{1}}\psi_{1,2}(x_{1}, x_{2})  \right]  \right] \dots \right] \\
& \left[ \sum_{x_{n+1}}\psi_{n, n+1}(x_{n}, x_{n+1})\dots \left[ \sum_{N}\psi_{N-1, N}(x_{N-1}, x_{N}) \right] \dots  \right] 
\end{aligned}
$$

We can see that the expression for the marginal $p(x_n)$ decomposes into the product of two factors times the normalization constant:

$$p(x_{n}) = \frac{1}{Z}\mu_{\alpha}(x_{n})\mu_{\beta}(x_{n})$$

The message $\mu_{\alpha}(x_{n})$ and $mu_{\beta}(x_{n})$ can be evaluated recursively:

$$\mu_{\alpha}(x_{n}) = \sum_{x_{n-1}}\psi_{n-1,n}(x_{n-1}, x_{n})\mu_{\alpha}(x_{n-1})$$

$$\mu_{\beta(x_{n})} = \sum_{x_{n+1}}\psi_{n,n+1}(x_{n},x_{n+1})\mu_{\beta}(x_{n+1})$$

We can calculate the joint distribution of the neighboring nodes:

$$p(x_{n-1}, x_{n}) = \frac{1}{Z}\mu_{\alpha}(x_{n-1})\psi_{n-1, n}(x_{n-1}, x_{n})\mu_{\beta}(x_{n})$$

### 8.4.2 Trees

* Undirected tree
* Directed tree
* Polytree

### 8.4.3 Factor graphs
Factor graphs introduces additional nodes for the factors themselves in addition to the nodes representing the variables. To achieve this we write the joint distribution over a set of variables in the form of a product of factors:

$$p(x)=\prod_s f_s(x_s)$$

### 8.4.4 The sum-product algorithm
Our goal is to calculate the marginal for variable node $x$ and this marginal is given by the product of incoming messages along all of the links arriving at that node.

If a leaf node is a variable node, then the message taht it sends along its one and only link is given by:

$$\mu_{x\rightarrow f}(x) = 1$$

IF the leaf node is a factor node, the message sent should take the form:

$$\mu_{f\rightarrow x}(x) = f(x)$$

We can compute recursively that:

$$
\mu_{f_{s}\rightarrow x}(x) = \sum_{x_{1}}\dots\sum_{x_{M}}f_{s}(x,x_{1}, \dots, x_{M})\prod_{m\in ne(f_{s}) \\ x}\mu_{x_{m}\rightarrow f_{s}}(x_{m})
$$

$$\mu_{x_{m}\rightarrow f_{s}}(x_{m}) = \sum_{l\in ne(x_{m}) \\f_{s}}\mu_{f_{l}\rightarrow x_{m}}(x_{m})$$

The marginal distribution of $p(x_i)$ will be:

$$p(x) = \prod_{s\in ne(x)}\mu_{f_{s}\rightarrow x}(x)$$

### 8.4.5 The max-sum algorithm
Our goal is to find a setting of the variables that has the largest probability and to find the value of that probability, we shall simply write out the max operator in terms of its components:

$$\max_{x}p(x) = \max_{x_{1}}\dots \max_{x_{M}}p(x)$$

$$p(x) = \prod_{s}f_{s}(x_{s})$$

The max-sum algorithm in terms of message passing can be:

$$\mu_{f\rightarrow x}(x) = \max_{x_{1}, \dots, x_{M}}\left[ \ln f(x, x_{1}, \dots, x_{M}) + \sum_{m\in ne(f)/x} \mu_{x_{m}\rightarrow f}(x_{m}) \right]$$

$$\mu_{x\rightarrow f}(x) = \sum_{l\in ne(x)/f}\mu_{f_{l}\rightarrow x}(x)$$

The initial message sent by the leaf nodes are given by:

$$\mu_{x\rightarrow f}(x) = 0$$

$$\mu_{f\rightarrow x}(x) = \ln f(x)$$

while at the root node the maximum probability can then be computed, ay analogy in max-sum algorithm, with using:

$$p^{max} = \max_{x}\left[ \sum_{s\in ne(x)}\mu_{f_{s}\rightarrow x}(x)  \right]$$

### 8.4.6 Exact inference in general graphs
### 8.4.7 Loopy belief propagation
### 8.4.8 Learning the graph structure
From a Bayesian viewpoint, we would like to compute a posterior distribution over graph structures and to make predictions by averaging with respect to this distribution.

$$p(m|D)\propto p(m)p(D|m)$$
