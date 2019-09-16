---
layout:     post                    # 使用的布局（不需要改）
title:      PRML Chapter 5 Neural Networks           # 标题 
subtitle:    #副标题
date:       2019-09-16             # 时间
author:     WZY                      # 作者
header-img: img/vis1.png   #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true
tags:                               #标签
    - Machine Learning
--- 

# PRML Chapter 5 Neural Networks

## 5.1 Feed-forward Network Functions
A network with one hidden layer may be the form like this:

$$y_{k}(x, w) = \sigma\left( \sum_{j=1}^{M}w_{kj}^{2}h(\sum_{i=1}^{D}w_{ji}^{1}x_{i} + w_{j0}^{1}) + w_{k0}^{2} \right)$$


### 5.1.1 Weight-space symmetries
## 5.2 Network Training
The error function that we need to minimize:

$$E(w) = \frac{1}{2}\sum_{n=1}^{N} || y(x_{n}, w) - t_{n} ||^{2}$$

For the regression problem, assume that the output of the network is

$$p(t | x, w) = N(t | y(x,w), \beta^{-1})$$

The corresponding likelihood function will be

$$p(\mathbf{t} | \mathbf{X}, w, \beta) = \prod_{n=1}^{N}p(t_{n} | x_{n}, w, \beta)$$

Taking the negative logarithm, we obtain the error function:

$$\frac{\beta}{2}\sum_{n=1}^{N}\{ y(x, w) - t_{n} \}^{2} - \frac{N}{2}\ln\beta + \frac{N}{2} \ln(2\pi)$$

from which we can learn $w$ and $\beta$. Maximizing the likelihood funciton is equivalent to minimizing the sum-of-squares error function given by:

$$E(w) = \frac{1}{2}\sum_{n=1}^{N}\{y(x_{n}, w) - t_{n}  \}^{2}$$

Having found $w_{ML}$, the value of $\beta$ can be found by minimizing the negative log likelihood to give

$$\frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^{N}\{ y(x_{n}, w_{ML}) - t_{n} \}^{2}$$

In the regression case, we can view the network as having an output activation function that is the identity, so that $y_k=a_k$. The corresponding sum-of-squares error function has the property

$$\frac{\partial E}{\partial a_{k}} = y_{k} - t_{k}$$

In the case of binary classification, the conditional distribution of targets is a Bernoulli distribution:

$$p(t|x,w)=y(x,w)^t\{1-y(x,w)\}^{1-t}$$

Given by negative log likelihood, we have a cross-entropy error function:

$$E(w) = -\sum_{n=1}^{N}\{ t_{n}\ln y_{n} + (1-t_{n})\ln(1-y_{n}) \}$$

And for multiclass classification problem:

$$E(w) = -\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\ln y_{k}(x_{n}, w)$$

### 5.2.1 Parameter optimization
We choose some initial value and moving through weight space in a succession of steps:

$$w^{\tau + 1} = w^{\tau} + \Delta w^{\tau}$$

### 5.2.2 Local quadratic approximation
Consider the Taylor expansion of $E(w)$ around some point $\hat{w}$ in weight space:

$$E(w) \simeq E(\hat{w}) + (w - \hat{w})^{T}b + \frac{1}{2}(w - \hat{w})^{T}H(w - \hat{w})$$

where $b$ is defined to be the gradient of $E$ evaluated at $\hat{w}$: $b \equiv \nabla E\|_{w=\hat{w}}$

When $w^*$ is a minimum of the error function, there is no linear term because $\nabla E=0$, we have:

$$E(w) = E(w^{*}) + \frac{1}{2}(w-w^{*})^{T}H(w-w^{*})$$

### 5.2.3 Use of gradient information
### 5.2.4 Gradient descent optimization
Using gradient information:

$$w^{(\tau + 1)} = w^{(\tau)} - \eta\nabla E(w^{(\tau)})$$

On-line gradient descent(sequential/stochastic gradient descent)

$$w^{(\tau+1)} = w^{(\tau)}- \eta\nabla E_{n}(w^{(\tau)})$$

## 5.3 Error Backpropagation
### 5.3.1 Evaluation of error-function derivatives
We now derive the backpropagation algorithm for a general network:

$$E_n=\frac{1}{2}\sum_k(y_k(x_n,w)-t_{nk})^2\ \rightarrow\ \frac{\partial E_n}{\partial w_{ji}}=(y_{nj}-t_{nj})x_{ni}$$

In a general feed-forward network, each unit computes a weighted sum of its inputs, and the sum is transformed by a nonlinear activation function:

$$a_j=\sum_i w_{ji}z_i,\ \ z_j=h(a_j)$$

Apply the chain rule

$$\frac{\partial E_n}{\partial w_{ji}}=\frac{\partial E_n}{\partial a_j}\frac{\partial a_j}{\partial w_{ji}}=\delta_j z_i$$

We can obtain the backpropagation formula:

$$\delta_j=\frac{\partial E_n}{\partial a_j}=\sum_k\frac{\partial E_n}{\partial a_k}\frac{\partial a_k}{\partial a_j}=h'(a_j)\sum_k w_{kj}\delta_k$$

For all data points, we have:

$$E(w)=\sum_{n=1}^N E_n(w),\ \ \frac{\partial E}{\partial w_{ji}}=\sum_n\frac{\partial E_n}{\partial w_{ji}}$$

### 5.3.2 A simple example
### 5.3.3 Efficiency of backpropagation
### 5.3.4 The Jacobian matrix
Consider the evaluation of the Jacobian matrix, first we write down the backpropagation formula to determine the derivatives $\frac{\partial y_{k}}{\partial a_{j}}$.

$$\frac{\partial y_{k}}{\partial a_{j}} = \sum_{l}\frac{\partial y_{k}}{\partial a_{l}}\frac{\partial a_{l}}{\partial a_{j}} = h^{'}(a_{j})\sum_{l}w_{lj}\frac{\partial y_{k}}{\partial a_{l}}$$


If we have individual sigmoidal activation functions at each output unit, then

$$\frac{\partial y_{k}}{\partial a_{l}} = \delta_{kl}^{'}(a_{l})$$

whereas for softmax outputs we have:

$$\frac{\partial y_{k}}{\partial a_{l}} = \delta_{kl}y_{k} - y_{k}y_{l}$$

Finally we can calculate the element in the Jacobi matrix:

$$J_{ji} = \sum_{j}w_{ji}\frac {\partial y_{k}}{\partial a_{j}}$$

## 5.4 The Hessian Matrix
### 5.4.1 Diagonal approximation
The diagonal elements of the Hessian can be written:

$$\frac{\partial^2 E_n}{\partial w^2_{ji}}=\frac{\partial^2 E_n}{\partial a_j^2} z^2_i$$

Recursively using the chain rule of differential calculus to give a backpropagation equation of the form:

$$\frac{\partial^2 E_n}{\partial a_j^2}=h'(a_j)^2\sum_k\sum_{k'}w_{kj}w_{k'j} \frac{\partial^2 E_n}{\partial a_k \partial a_{k'}}+h''(a_j)\sum_k w_{kj}\frac{\partial E^n}{\partial a_k}$$

### 5.4.2 Outer product approximation
Write teh Hessian matrix in the form:

$$H=\nabla\nabla E=\sum_{n=1}^N\nabla y_n\nabla y_n+\sum_{n=1}^N(y_n-t_n)\nabla\nabla y_n$$

By neglecting the second term, we get the Levenberg-Marquardt approximation (outer product approximation) 

$$H\simeq\sum_{n=1}^N b_n b_n^T$$

where $b_n=\nabla y_n=\nabla a_n$.

In the case of the cross-entropy error function for a network with logistic sigmoid output-unit activation functions, the corresponding approximation is given by:

$$H\simeq\sum_{n=1}^N y_n(1-y_n)b_nb_n^T$$

## 5.5 Regularization in Neural Networks
To control the complexity of a neural network, the simplest regularizer is the quadratic, giving a regularized error:

$$\tilde{E}(w)=E(w)+\frac{\lambda}{2} w^Tw$$

### 5.5.1 Consistent Gaussian priors
A regularizer which is invariant under the linear transformations is given by:

$$\frac{\lambda_{1}}{2}\sum_{w\in W_{1}}w^{2} + \frac{\lambda_{2}}{2}\sum_{w\in W_{2}}w^{2}$$

### 5.5.2 Early stopping
Training can be stopped at the point of smallest error with respect to the validation set in order to obtain a network having good generalization performance.

### 5.5.3 Invariances
### 5.5.4 Tangent propagation
We can use regularization to encourage models to be invariant to transformations of the input through the technique of tangent propagation. Let the vector that results from acting on $x_n$ bu this transformation be denoted by $s(x_{n}, \epsilon)$ and $s(x_n,0)=x$. Then the tangent to the curve $M$ is given by the directional derivative $\tau = \frac{\partial x}{\partial \epsilon}$,and the tangent vector at the point $x_n$ is given by:

$$\tau_n=\frac{\partial s(x_n,\epsilon)}{\partial\epsilon}|_{\epsilon =0}$$

The derivative of output k with respect to $\epsilon$ is given by:

$$\frac{\partial y_{k}}{\partial \epsilon} |_{\epsilon=0} =\sum_{i=1}^D\frac{\partial y_k}{\partial x_i}\frac{x_i}{\partial\epsilon}|_{\epsilon=0} =\sum_{i=1}^{D}J_{ki}\tau_{i}$$

The result can be used to modify the standard error funciton:

$$\tilde{E}=E+\lambda\Omega$$

where $\lambda$ is a regularization coefficient and:

$$\Omega=\frac{1}{2}\sum_n\sum_k(\frac{\partial y_{nk}}{\partial \epsilon}|_{\epsilon=0})^2=\frac{1}{2}\sum_n\sum_k(\sum_{i=1}^D J_{nki}\tau_{ni})^2$$

### 5.5.5 Training with transformed data
Consider a transformation governed by a single parameter $\epsilon$ and describe by the function $s(x,\epsilon)$. Consider a sum-of-squares error function, for untransformed inputs can be written in the form:

$$E = \frac{1}{2}\int\int \{ y(x) - t\}^{2}p(t|x)p(x) dx dt$$

if the parameter $\epsilon$ is drawn from a distribution $p(\epsilon)$, then:

$$\tilde{E} = \frac{1}{2}\int\int \{ y(s(x, \epsilon)) - t\}^{2}p(t|x)p(x)p(\epsilon) dx dt d\epsilon$$

Further assume that $p(\epsilon)$ has zero mean with small variance, after the Taylor expansion and substituting into the mean error function, the average error
$$\tilde{E} = E + \lambda\Omega$$

where E is the original sum-of-squares error, and the regularization term $Omega$ takes the form:

$$\Omega = \frac{1}{2}\int [ \{ y(x) - E[t|x] \} \{ (\tau')^T\nabla y(x) + \tau^{T}\nabla\nabla y(x)\tau \} + (\tau^{T}\nabla y(x))^{2} ]p(x) dx$$

### 5.5.6 Convolutional networks
### 5.5.7 Soft weight sharing
In this part, the hard constraint of equal weights is replaced by a form of regularization in which groups of weights are encouraged to have similar values. Furthermore, the division of weights into groups, the mean weight value for each group, and the spread of values within the groups are all determined as part of the learning process.

## 5.6 Mixture Density Networks
Develop the model explicitly for Gaussian components, so that:

$$p(t|x) = \sum_{k=1}^{K}\pi_{k}(x)N(t | \mu_{k}(x), \sigma_{k}^{2}(x)I)$$

For indenpendent data, the error function takes the form:

$$E(w) = -\sum_{n=1}^{N}\ln \left\{ \sum_{n=1}^{K}\pi_{k}(x_{n}, w)N(t_{n} | \mu_{k}(x_{n},w), \sigma_{k}^{2}(x_{n}, w)\mathbf{I})   \right\}$$

## 5.7 Bayesian Neural Networks
In this part, we will approximate the posterior distribution by a Guassian, centred at a mode of the true posterior. We will also assume that the covariance of this Gaussian is small so that the network function is approximately linear.

### 5.7.1 Posterior parameter distribution
We suppose that the conditional distribution $p(t\|x)$ is Gaussian.

$$p(t|x,w,\beta) = N(t | y(x, w), \beta^{-1})$$

Also, we choose a prior distribution over the weights $w$ that is Guassian of the form.

$$p(w | \alpha) = N(w | 0, \alpha^{-1}\mathbf{I})$$

For an i.i.d. data set of $N$ observations $x_1,...,x_N$, with a corresponding set of target values $D=\{t_1,...,t_N\}$, the likelihood function is given by:

$$p(D | w, \beta) = \prod_{n=1}^{N}N(t_{n} | y(x, w), \beta^{-1})$$

so we can get the posterior distribution:

$$p(w | D, \alpha, \beta) \propto p(w | \alpha)p(D | w, \beta)$$

The Gaussian approximation to the posterior is given by:

$$q(w | D) = N(w | w_{MAP}, \mathbf{A}^{-1})$$

Similarly, the predictive distribution is obtained by marginalizing with respect to this posterior distribution:

$$p(t | x, D) = \int p(t | x, w)q(w | D) dw$$

Make a Taylor series expansion of the network function around $w_{MAP}$ and retain the linear terms, we will get a linear-Gaussian model:

$$p(t| x, w, \beta) \simeq N(t | y(x, w_{MAP}) + g^{T}(w - w_{MAP}), \beta^{-1})$$

we can therefore make use of the general result for the marginal $p(x)$ to give:

$$p(t | x, D, \alpha, \beta) = N(t | y(x, w_{MAP}), \sigma^{2}(x))$$

where

$$\sigma^{2}(x) = \beta^{-1} + g^{T}\mathbf{A}^{-1}g$$

$$g = \nabla_{w}y(x,w)|_{w = w_{MAP}}$$

### 5.7.2 Hyperparameter optimization
### 5.7.3 Bayesian neural networks for classification
The logistic sigmoid output corresponding to a two-class classification problem. The log likelihood function for this model is given by:

$$\ln p(D|w) = \sum_{n=1}^{N}\{t_{n}\ln y_{n} + (1-t_{n})\ln(1-y_{n}) \}$$

Minimizing the regularized error function:

$$E(w) = \ln p(D|w) + \frac{\alpha}{2}w^{T}w$$

The result of the approximate distribution will be

$$p(t=1 | x, D) = \sigma(k(\sigma_{a}^{2})b^T w_{MAP})$$
