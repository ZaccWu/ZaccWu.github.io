---
layout:     post                    # 使用的布局（不需要改）
title:      The Parameters of XGBoost and LightGBM               # 标题 
subtitle:    #副标题
date:       2019-08-31             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Machine Learning
--- 
# The Parameters of XGBoost and LightGBM
***
This passage summarize the most important and frequently used parameters in XGBoost and LightGBM, which can help us find the information more convenient when we are using these two models and frameworks.

## 1 Parameters of XGBoost
There are three types of parameters that we need to focus on: General parameters, Booster parameters as well as Objective parameters.
### 1.1 General Parameters
**booster**
* bdtree: Using tree structure.
* gblinear: Using linear model to operate data.

**silent**: Silent pattern，'1' means do not output when operating the model.

**nthread**: Number of thread. Default -1, means using all the threads.

### 1.2 Booster Parameters
**n_estimator**: Maximum number of iterations/Number of largest trees generated.

**learning rate (eta)**: Steps size in each iteration (default: 0.3)
This parameter is important in the model. If the iteration step size is too large, the accuracy will be reduced, while too small will slow down the running speed. About 0.1 is appropriate.

**gamma**：Set the minimum loss function drop value required for node splitting (default: 0)
The larger gamma means node splitting requires larger value drop of the loss function, thus harder to split the node. If gamma is set too large the algorithm will be conservative.

**subsample**：Control the proportion of random sampling to each tree. (default: 1)
The value range is (0,1], reduce the value can avoid over-fitting. 0.5 means average sampling.

**colsample_bytree**：Control the column proportion of random sampling each time. A column represents one feature, usually the parameter is set 0.8. (default: 1)

**colsample_bylevel**：Column sampling ratio for each node splitting of each tree (default: 1)

**max_depth**：Maximum depth of the tree (default: 6)
Usually this parameter is set 3-10, it can control over-fitting, deeper means learning more accurately.

**max_delta_step**：Limit the maximum stride length of each tree weight change (default: 0)
Setting it to 0 means there are no constraints. Generally, no setting is required, but setting this parameter can help the Logistic regression optimizer when the category sample and its imbalance are involved

**lambda**：L2 regularization term，similar to Ridge Regression (default: 0)

**alpha**：L1 regularization term, similar to Lasso Regression(default: 0)

**scale_pos_weight**：When classes are unbalanced, a positive value is set (usually the ratio of negative sample number to positive sample number) to make the algorithm converge faster (default: 1)

### 1.3 Objective Parameters
**objective**
* reg:linear：linear regression (default)
* reg:logistic: logistic regression
* binary:logistic: binary logistic regression, output the probability.
* binary:logitraw: binary logistic regression, output $w^Tx$
* count:poisson: poisson regression of the counting problem，output a poisson distribution.
* multi:softmax: set XGBoost using softmax objective function for multi-classification.
* multi:softprob: Same as softmax，but output a vector of ndata*nclass, the value are the probabilities that the point being assigned to each class.

**eval_metric**
* rmse: Root mean square error
* mae: Mean absolute value error
* logloss: negative log-likelihood
* error: The binary classification error rate (obtained by the ratio of the number of error classifications to the number of all classifications).
* merror：Multi-classification error rate.
* mlogloss：Multi-classification log loss.
* auc：Area under the curve
* ndcg：Normalized Discounted Cumulative Gain
* map：Average correct ratio.

## 2 Parameters of LightGBM
There exist three kinds of parameters in XGBoost. While for LightGBM, more kinds of parameters exist. Such as Core parameters, learning controling parameters, IO and so on. And here are some parameters that are frequently used.

### 2.1 Core Parameters
**boosting**: Also called boost or boosting_type (default gbdt).
The number of boosting parameters in LGB is larger than XGB, such as traditional gbdt as well as rf, dart, doss and so on. Tradional gbdt may be more stable in some cases.
* gbdt
* rf: Random Forest
* dart: Dropouts meet Multiple Additive Regression Trees
* goss: Gradient-based One-Side Sampling.

**num_thread**: Also called num_thread,nthread. Means the number of threads.
The official documentation here mentions that it is faster to set the number of CPU cores than the number of threads (considering that most cpus are now hyper-threaded). Parallel learning should not be set to all threads, which makes training slow.

**application**：The goal of the task (default regression)
* regression
    * regression_l2: L2 loss, alias=regression, mean_squared_error, mse
    * regression_l1: L1 loss, alias=mean_absolute_error, mae
    * huber: Huber loss
    * fair: Fair loss
    * poisson: Poisson regression
    * quantile: Quantile regression
    * quantile_l2: Similar to quantile, but using L2 loss
* binary, binary log loss classification application
* multi-classification
    * multiclass, softmax objective function, need to set 'num_class'
    * multiclassova, One-vs-All binary objective function, need to set 'num_class'
* cross-entropy application
    * xentropy: cross-entropy objective function (also can choose linear weight), alias=cross_entropy
    * xentlambda: Substitute parameterized cross-entropy, alias=cross_entropy_lambda
* lambdarank, lambdarank application
In lambdarank the label should be int type, larger means more relevant (e.g. 0:bad, 1:fair, 2:good, 3:perfect)

**valid**: Used in validation set，also called test，valid_data, test_data. Supports multi-validation set.

**learning_rate**: also called shrinkage_rate, the length of the step in gradient descent. (default 0.1, always between 0.05 and 0.2)

**num_leaves**: also called num_leaf(default 31), which represents the number of the leaves on a tree.

**num_iterations**: also called num_iteration, num_tree, num_trees, num_round, num_rounds,num_boost_round.

**device**：default=cpu, options=cpu, gpu. 
Choose the device for the learning process, using GPU to gain a faster learning speed.

### 2.2 Learning Controling Parameters
**max_depth**: (default -1, type=int) To limit the maximum depth of the tree model, which can control over-fitting. Trees can grow by leaf-wise, too. Setting under zero means no limitation. 

**feature_fraction**：(default=1.0, type=double, 0.0 < feature_fraction < 1.0) also called sub_feature, colsample_bytree. If the feature_fraction lower than 1.0, LightGBM will choose some of the features in each step randomly. For example, when set the parameter to 0.8, it will choose 80% of the features before training each tree. Also, this parameter can speed up training and deal with over-fitting.

**bagging_fraction**：(default=1.0, type=double, 0.0 < bagging_fraction < 1.0) also called sub_row, subsample. 
Similar to feature_fraction, but it can randomly choose the data with out resampling. It can also speed up training and deal with over-fitting.

**bagging_freq**: (default=0, type=int) also called subsample_freq. This parameter means the frequency of bagging. 0 represents prohibit bagging process while k means process the bagging every k times.

**lambda_l1**: (default 0) also called reg_alpha, L1 regularization.

**lambda_l2**: (default 0) also called reg_lambda, L2 regularization.

**cat_smooth**: (default=10, type=double). This parameter is used for feature classification. It can reduce the impact of the noise in the classify process, especially for the data with fewer classes.

**min_data_in_leaf**: (default 20) also called min_data_per_leaf , min_data, min_child_samples. 

**min_sum_hessian_in_leaf**: (default=1e-3) also called min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight. This parameter represents the minimax hessian sum of a leaf, similar to min_data_in_leaf, which can deal with over-fitting. 

**early_stopping_round**: (default 0, type=int) also called early_stopping_rounds, early_stopping. If early_stopping_round in a validation set doesn't improve in a circulation, stop training.

**min_split_gain**: (default 0, type=double) also called min_gain_to_split.

**max_bin**：The maximum number of histogram(default 255).

### 2.3 Measurement Parameters
**metric**： (default={l2 for regression}, {binary_logloss for binary classification}, {ndcg for lambdarank}, type=multi-enum, options=l1, l2, ndcg, auc, binary_logloss, binary_error …)
* l1: absolute loss, alias=mean_absolute_error, mae
* l2: square loss, alias=mean_squared_error, mse
* l2_root: root square loss, alias=root_mean_squared_error, rmse
* quantile： Quantile regression
* huber： Huber loss
* fair： Fair loss
* poisson： Poisson regression
* ndcg: NDCG
* map: MAP
* auc: AUC
* binary_logloss: log loss
* binary_error
* multi_loglos
* multi_error
* xentropy, cross-entropy (与可选的线性权重), alias=cross_entropy
* xentlambda, “intensity-weighted” 交叉熵, alias=cross_entropy_lambda
* kldiv, Kullback-Leibler divergence, alias=kullback_leibler

## 3 The Most Important in Adjusting Parameters
* Improving the accuracy: num_leaves, max_depth, learning_rate.
* Control Overfit: max_bin, min_data_in_leaf, regularization, sampling.

If we want to deal with overfitting, we should use:
* smaller 'max_bin'
* smaller 'num_leaves'
* use the parameters 'min_data_in_leaf' and 'min_sum_hessian_in_leaf'
* set 'bagging_fraction' and 'bagging _freq' to use bagging
* set 'feature_fraction'<1 to use feature sampling
* set 'lambda_l1', 'lambda_l2' and 'min_gain_to_split' to use regularization.
* use 'max_depth' to avoid large-depth-tree.
