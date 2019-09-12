---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 4 Linear Models for Classification               # 标题 
subtitle:    #副标题
date:       2019-09-12             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 

# PRML Chapter 4 Linear Models for Classification

## 4.1 Discriminant Functions
### 4.1.1 Two classes
The simplest representation of a linear discriminant function can be expressed as:

$$y(x) = w^{T}x + w_{0}$$

The normal distance from the origin to the decision surface is given by

$$\frac{w^{T}x}{||w||} = - \frac{w_{0}}{||x||}$$

### 4.1.2 Multiple classes
Considering a single K-class discriminant comprising K linear functions of the form

$$y_{k}(x) = w_{k}^{T}x + w_{k0}$$

and we assume a point $x$ to class $C_k$ if $y_k(x)>y_j(x)$. The decision boundary between class $C_k$ and class $C_j$ is given by $y_k(x)=y_j(x)$, and the corresponding D-1 dimensional hyperplane can be

$$(w_{k} - w_{j})^{T}x + (w_{k0} - w_{j0}) = 0$$

### 4.1.3 Least squares for classification
Each class $C_k$ is described by its own linear model so that 

$$y_k(x)=w^T_kx+w_{k0}$$

and we can group these together to get

$$y(x) = \tilde{W}^{T}\tilde{x}$$

The sum-of-squares error function can be written as

$$E_{D}(\tilde{W}) = \frac{1}{2}Tr\left\{ (\tilde{X}\tilde{W} - T)^{T}(\tilde{X}\tilde{W} - T)  \right\}$$

Let the derivative of $\tilde{W}$ equal to zero and we can obtain the solution for $\tilde{W}$:

$$\tilde{W} = (\tilde{X}^{T}\tilde{X})^{-1}\tilde{X}^{T}T = \tilde{X}^{\dag}T$$

We then obtain the discriminant function in the form

$$y(x) = \tilde{W}^{T}\tilde{x} = T^{T}(\tilde{X}^{\dag})^{T}\tilde{x}$$

### 4.1.4 Fisher's linear discriminant
The mean vectors of the two classes are given by

$$m_1=\frac{1}{N_1}\sum_{n\in C_1} x_n,\ m_2=\frac{1}{N_2}\sum_{n\in C_2}x_n$$

To avoid the overlap in the projection space, we might choose $w$ to maximize $m_{2} - m_{1} = w^{T}(\mathbf{m_{2} - m_{1}})$.

The with-in class variance of the transform data from class $C_k$ is given by:

$$s_{k}^{2} = \sum_{n\in C_{k}}(y_{n} - m_{k})^{2}$$

where $y_n=w^T x_n$. The Fisher criterion is defined to be the ratio of the between-class variance to the within clss variance:

$$J(w) = \frac{(m_{2} - m_{1})^{2}}{s_{1}^{2} - s_{2}^{2}}$$

Rewrite the Fisher criterion in the following form, here we have the between-class covariance matirx $S_B$ and within-class covariance matrix $S_W$.

$$S_{B} = (m_{2} - m_{1})(m_{2} - m_{1})^{T}$$

$$S_{W} = \sum_{n\in C_{1}}(x_{n} - m_{1})(x_{n} - m_{1})^{T} + \sum_{n\in C_{2}}(x_{n}-m_{2})(x_{n} - m_{2})^{T}$$

$$J(w) = \frac{w^{T}S_{B}w}{w^{T}S_{W}w}$$

Differentiating the formula above and we can see that $J(w)$ is maximized when:

$$(w^{T}S_{B}w)S_{W}w = (w^{T}S_{W}w)S_{B}w$$

we then obtain:

$$w \propto S_{W}^{-1}(m_{2} - m_{1})$$

### 4.1.5 Relation to least squares
The Fisher solution can be obtained as a special case of least squares for the two class problem. For class $C_1$ we shall take the targets to be $N/N_1$ and for $C_2$ to be $-N/N_2$.

The sum-of-square error function can be written:

$$E=\frac{1}{2}\sum_{n=1}^N(w^Tx_n+w_0-t_n)^2$$

Setting the derivatives of E with respect to $w_0$ and $w$ to zero, we can obtain:

$$\sum_{n=1}^N(w^Tx_n+w_0-t_n)=0$$

$$\sum_{n=1}^N(w^Tx_n+w_0-t_n)x_n=0$$

Thus we can get:

$$w_0=-w^T m$$
$$(S_W+\frac{N_1N_2}{N}S_B)w=N(m_1-m_2)\ \rightarrow w\propto S_W^{-1}(m_2-m_1)$$

### 4.1.6 Fisher's discriminant for multiple classes
For multiple classes problem, similar to the two classes, the input space may contain:

* Mean vector
$$m_k=\frac{1}{N_k}\sum_{n\in C_k}x_n,\ \ m=\frac{1}{N}\sum_{n=1}^Nx_n=\frac{1}{N}\sum_{k=1}^KN_km_k$$

* Within-class covariance matrix
$$S_W=\sum_{k=1}^K S_k,\ \ S_k=\sum_{n\in C_k}(x_n-m_k)(x_n-m_k)^T$$

* Between-class covariance matrix
$$S_B=\sum_{k=1}^K N_k(m_k-m)(m_k-m)^T$$

* The total covariance matrix
$$S_T=\sum_{n=1}^N(x_n-m)(x_n-m)^T,\ \ S_T=S_W+S_B$$

Next we introduce $D'>1$ linear 'features' $y_k=w_k^Tx$ and the feature values can be grouped together: $y=W^Tx$. We can define similar matrices in the projected D'-dimensional y-space.

$$s_W=\sum_{k=1}^K\sum_{n\in C_k}(y_n-\mu_k)(y_n-\mu_k)^T$$

$$s_B=\sum_{k=1}^K N_k(\mu_k-\mu)(\mu_k-\mu)^T$$

$$\mu_k=\frac{1}{N_k}\sum_{n\in C_k}y_n,\ \ \mu=\frac{1}{N}\sum_{k=1}^KN_k \mu_k$$

One of the many choices of criterion is $J(W)=Tr(s_W^{-1}s_B)$ and it is straightforward to see that we should maximize $J(W)=Tr[(WS_WW^T)^{-1}(WS_BW^T)]$
### 4.1.7 The perceptron algorithm
A generalized linear model will be the form:

$$y(x) = f(w^{T}\phi(x))$$

and the nonlinear activation function is given by:

$$f(a) = \left\{
    \begin{aligned}
    +1, ~~~a\geq 0 \\
    -1, ~~~a\lt 0
    \end{aligned}
    \right.$$

Here's an alternative error function known as perceptron criterion:

$$E_{P}(w) = -\sum_{n\in M}w^T\phi_{n}t_{n}$$

And the training process towards this problem will be the stochastic gradient descent algorithm:

$$w^{(\tau+1)}=w^{(\tau)}-\eta\nabla E_P(w)=w^{(\tau)}+\eta\phi_n t_n$$

## 4.2 Probabilistic Generative Models
For the problem of two classes, the posterior probability for class one can be:

$$p(C_{1} | x) = \sigma(a)$$

where $a = \ln\frac{p(x\|C_{1})p(C_{1})}{p(x\|C_{2})p(C_{2})}$ and $\sigma(a) = \frac{1}{1 + \exp(-1)}$ (logistic sigmoid).

For the case of K>2, we have:

$$p(C_{k} | x) = \frac{p(x|C_{k})p(C_{k})}{\sum_{j}p(x|C_{j})p(C_{j})} = \frac{exp(a_{k})}{\sum_{j}\exp(a_{j})}$$

where $a_{k} = \ln p((x \| C_{k})p(C_{k}))$ (equals to softmax function).

### 4.2.1 Continuous inputs
Assume that the class-conditional densities are Gaussian and all classes share the same covariance matrix. So the density for class $C_k$ is given by:

$$p(x | C_{k}) = \frac{1}{2\pi^{\frac{D}{2}}}\frac{1}{|\Sigma|^{\frac{1}{2}}}\exp\left\{ -\frac{1}{2}(x-\mu_{k})^{T}\Sigma^{-1}(x-\mu_{k}) \right\}$$

Consider the first two classes, we have:

$$p(C_{1} | x) = \sigma(w^{T}x + w_{0})$$

Where $w = \Sigma^{-1}(\mu_{1} - \mu_{2})$ and $w_{0} = -\frac{1}{2}\mu_{1}^{T}\Sigma^{-1}\mu_{1} + \frac{1}{2}\mu_{2}^{T}\Sigma^{-1}\mu_{2} + \ln\frac{p(C_{1})}{p(C_{2})}$.

For the general case of K classes we have, we have:

$$a_k(x)=w_k^Tx+w_{k0}$$

where $w_k=\Sigma^{-1}\mu_k$ and $w_{k0}=-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+lnp(C_k)$.

### 4.2.2 Maximum likelihood solution
After we specified the class-conditional densities, we can determine the parameters' value and prior probabilities using maximum likelihood.

For the case of two classes, we have:

$$p(\mathbf{t}, \mathbf{X} | \pi, \mu_{1}, \mu_{2}, \Sigma) = \prod_{n=1}^{N}[\pi N(x_{1} | \mu_{1}, \Sigma)]^{t_{n}}[(1-\pi)N(x_{n} | \mu_{2}, \Sigma)]^{1-t_{n}}$$

The solution will be:

$$\mu_{1} = \frac{1}{N_{1}}\sum_{n=1}^{N}t_{n}x_{n}$$

$$\mu_{2} = \frac{1}{N_{2}}\sum_{n=1}^{N}(1-t_{n})x_{n}$$

$$\Sigma = \frac{N_{1}}{N}S_{1} + \frac{N_{2}}{N}S_{2}$$

where $S_{1} = \frac{1}{N_{1}}\sum_{n\in C_{1}}(x_{n} - \mu_{1})(x_{n} - \mu_{1})^{T}$ and $S_{2} = \frac{1}{N_{2}}\sum_{n\in C_{2}}(x_{n} - \mu_{2})(x_{n} - \mu_{2})^{T}$.

The idea of multi-classes problem will be the same.

### 4.2.3 Discrete features
In this case, we have class-conditional distributions of the form:

$$p(x|C_k)=\prod_{i=1}^D\mu_{ki}^{x_i}(1-\mu_{ki})^{1-x_i}$$ 

The posterior probability density will be:

$$a_k(x)=\sum_{i=1}^D[x_i ln\mu_{ki}+(1-x_i)ln(1-\mu_{ki})]+lnp(C_k)$$

### 4.2.4 Exponential family
For members of the exponential family, the distribution of $x$ can be written in the form:
$$p(x|\lambda_k)=h(x)g(\lambda_k)exp\{\lambda_k^Tu(x)\}$$

if we let $u(x)=x$ and introduce a scaling parameter $s$:

$$p(x | \lambda_{k}, s) = \frac{1}{s}h(\frac{1}{s}x)g(\lambda_{k})\exp\left\{\frac{1}{s}\lambda_{k}^{T}x\right\}$$

Consequently, for two-class problem the posterior class probability is given by a logistic sigmoid acting on a linear function $a(x)$:

$$a(x)=(\lambda_1-\lambda_2)^Tx+lng(\lambda_1)-lng(\lambda_2)+lnp(C_1)-lnp(C_2)$$

And for K-classes problem:

$$a_{k}(x) = \frac{1}{s}\mathbf{\lambda_{k}^{T}x} + \ln g(\lambda_{k}) + \ln p(C_{k})$$

## 4.3 Probabilistic Discriminative Models
### 4.3.1 Fixed basis functions
### 4.3.2 Logistic regression
For two-class classification problem, the posterior probability of class $C_1$ can be written as a logistic sigmoid acting on a linear function of the feature vector:

$$p(C_{1} | \phi) = y(x) = \sigma(w^{T}\phi)$$

We now use MLE to determine the parameters of logistic regression:

$$p(\mathbf{t} | w) = \prod_{n=1}^{N}y_{n}^{t_{n}}\{1-y_{n}\}^{1-t_{n}}$$

where $\mathbf{t}=(t_1,...,t_n)$ and $y_n=p(C_1\|\phi_n)$. We can define an error function by taking the negative logarithm of the likelihood, which gives the cross-entropy error function in the form:

$$E(w) = -\ln p(\mathbf{t} | w) = -\sum_{n=1}^{N}\{ t_{n}\ln y_{n} + (1-t_{n})\ln(1-y_{n}) \}$$

where $y_n=\sigma(a_n)$ and $a_n=w^T\phi_n$. Taking the gradient of the error function with respect to $w$, we obtain:

$$\nabla E(w) = \sum_{n=1}^{N}(y_{n} - t_{n})\phi_{n}$$

### 4.3.3 Iterative reweighted least squares
The Newton-Raphson update, for minimizing a function $E(w)$, takes the form:

$$w^{new} = w^{old} - \mathbf{H}^{-1}\nabla E(w)$$

Apply the Newton-Raphson update to the cross-entropy error function for the logistic regression model, the gradient and Hessian of the error function are given by:

$$\nabla E(w) = \sum_{n=1}^{N}(y_{n} - t_{n})\phi_{n} = \Phi^{T}(\mathbf{y} - t)$$

$$H = \nabla\nabla E(w) = \sum_{n=1}^{N}y_{n}(1-y_{n})\phi_{n}\phi_{n}^{T} = \Phi^{T}R\Phi$$

$$R_{nn} = y_{n}(1-y_{n})$$

The Newton-Raphson update formula for the logistic regression model becomes:

$$w^{new} = w^{old} - (\Phi^{T}R\Phi)^{-1}\Phi^{T}(\mathbf{y} - \mathbf{t}) = (\Phi^{T}R\Phi)^{-1}\Phi^{T}Rz$$

$$z = \Phi w^{old}  - R^{-1}(\mathbf{y} - \mathbf{t})$$

### 4.3.4 Multiclass logistic regression
For this problem, the posterior probabilities are given by:

$$p(C_k|\phi)=y_k(\phi)=\frac{exp(a_k)}{\sum_j exp(a_j)}$$

where $a_k=w^T_k\phi$.

Similarly, we can write down the likelihood function:

$$p(\textbf{T}|w_1,...,w_K)=\prod_{n=1}^N\prod_{k=1}^K p(C_k|\phi_n)^{t_{nk}}=\prod_{n=1}^N\prod_{k=1}^K y_{nk}^{t_{nk}}$$

The cross-entropy error function for the multiclass classification problem:

$$E(w_1,...,w_K)=-lnp(\textbf{T}|w_1,...,w_k)=-\sum_{n=1}^N\sum_{k=1}^K t_{nk}lny_{nk}$$

The derivatives will be

$$\nabla_{w_j}E(w_1,...,w_K)=\sum_{n=1}^N(y_{nj}-t_{nj})\phi_n$$

$$\nabla_{w_k}\nabla_{w_j}E(w_1,...,w_K)=-\sum_{n=1}^N y_{nj}(I_{kj}-y_{nj})\phi_n\phi_n^T$$

### 4.3.5 Probit regression
If the value of $\theta$ is drawn from a probability density $p(\theta)$, then the corresponding activation function will be given by the cumulative distribution function:

$$f(a)=\int^a_{-\infty}p(\theta)d\theta$$

And we suppose the density is given by a zero mean, unit variance Gaussian:

$$\Phi(a) = \int_{-\infty}^{a} N(\theta | 0, 1) d\theta$$

which is known as probit function. Many numerical packages provide for evaluation of a closely related function defined by:

$$erf(a) = \frac{2}{\sqrt{\pi}}\int_{0}^{a}\exp(-\theta^{2})d\theta$$

which is known as erf function. It is related to the probit function by:

$$\Phi(a) = \frac{1}{2}\{ 1 + \frac{1}{\sqrt{2}}erf(a)\}$$

The generalized linear model based on a probit activation function is known as probit regression.

### 4.3.6 Canonical link functions
If we assume that the conditional distribution of the target variable comes from the exponential family distribution, the corresponding activation function is selected as the standard link function (the link function is the inverse of the activation function), then we have:

$$\nabla E(w) = \frac{1}{s}\sum_{n=1}^{N}\{y_{n} - t_{n}\}\phi_{n}$$

For the Gaussian $s=\beta^{-1}$, whereas for the logistic model $s=1$.

## 4.4 The Laplace Approximation
Laplace approximation aims to find a Gaussian approximation to a probability density defined over a set of continuous variables. Suppose the distribution is difined by:

$$p(z) = \frac{1}{z}f(z)$$

Expanding around the stationary point:

$$\ln f(z) \simeq \ln f(z_{0}) - \frac{1}{2}(z-z_{0})^{T}A(z-z_{0})$$

where $A = -\nabla\nabla ln f(z)_{z=z_{0}}$. Taking the exponential of both sides we obtain:

$$f(z)  \simeq f(z_{0})\exp \left\{ -frac{1}{2}(z - z_{0})^{T}A(z-z_{0}) \right\}$$

and we know that $q(z)$ is proportional to $f(z)$ so:

$$q(z) = \frac{|A|^{\frac{1}{2}}}{(2\pi)^{\frac{M}{2}}}\exp \left\{ -\frac{1}{2}(z-z_{0})^{T}A(z-z_{0}) \right\} = N(z | z_{0}, A^{-1})$$

### 4.4.1 Model comparison and BIC

## 4.5 Bayesian Logistic Regression
### 4.5.1 Laplace approximation
Seek a Gaussian representation for the posterior distribution and get the log form:

$$\ln p(w | t) = -\frac{1}{2}(w - w_{0})^{T}S_{0}^{-1}(w - w_{0}) + \sum_{n=1}^{N}\{ t_{n}\ln y_{n} + (1-t_{n})\ln(1-y_{n})\} + const$$

The covariance is then given by the inverse of the matrix of second derivatives of the negative log likelihood:

$$S_{N}^{-1} = -\nabla\nabla \ln p(w | t) = S_{0}^{-1} + \sum_{n=1}^{N}y_{n}(1-y_{n})\phi_{n}\phi_{n}^{T}$$

The Gaussian approximation to the posterior distribution therefore takes the form:

$$q(w) = N(w | w_{MAP}, S_{N})$$

### 4.5.2 Predictive distribution
The variational approximation to the predictive distribution:

$$p(C_{1} | t) = \int \sigma(a)p(a)d a = \int\sigma(a)N(a | \mu_{a}, \sigma_{a}^{2})da$$
