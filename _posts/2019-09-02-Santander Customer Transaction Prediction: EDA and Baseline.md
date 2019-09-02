---
layout:     post                    # 使用的布局（不需要改
title:      Santander Customer Transaction Prediction: EDA# 标题 
subtitle:   Kaggle Competition #副标题
date:       2019-09-02             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Kaggle
--- 

# Santander Customer Transaction Prediction: EDA and Baseline

## 1 Description
At Santander our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.

Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?

In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.

## 2 Prepare The Data
### 2.1 Import and preparation
First we import the packages that we might need in the solution.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import six.moves.urllib as urllib
import sklearn
import scipy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
%matplotlib inline
```


```python
PATH='E:/kaggle/santander-customer-transaction-prediction/'
train=pd.read_csv(PATH+'train.csv')
test=pd.read_csv(PATH+'test.csv')
```

Check the data information

```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200000 entries, 0 to 199999
    Columns: 202 entries, ID_code to var_199
    dtypes: float64(200), int64(1), object(1)
    memory usage: 308.2+ MB
    

Check the dimension of the data

```python
train.shape
```




    (200000, 202)




```python
train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID_code</th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>...</th>
      <th>var_190</th>
      <th>var_191</th>
      <th>var_192</th>
      <th>var_193</th>
      <th>var_194</th>
      <th>var_195</th>
      <th>var_196</th>
      <th>var_197</th>
      <th>var_198</th>
      <th>var_199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>train_0</td>
      <td>0</td>
      <td>8.9255</td>
      <td>-6.7863</td>
      <td>11.9081</td>
      <td>5.0930</td>
      <td>11.4607</td>
      <td>-9.2834</td>
      <td>5.1187</td>
      <td>18.6266</td>
      <td>...</td>
      <td>4.4354</td>
      <td>3.9642</td>
      <td>3.1364</td>
      <td>1.6910</td>
      <td>18.5227</td>
      <td>-2.3978</td>
      <td>7.8784</td>
      <td>8.5635</td>
      <td>12.7803</td>
      <td>-1.0914</td>
    </tr>
    <tr>
      <td>1</td>
      <td>train_1</td>
      <td>0</td>
      <td>11.5006</td>
      <td>-4.1473</td>
      <td>13.8588</td>
      <td>5.3890</td>
      <td>12.3622</td>
      <td>7.0433</td>
      <td>5.6208</td>
      <td>16.5338</td>
      <td>...</td>
      <td>7.6421</td>
      <td>7.7214</td>
      <td>2.5837</td>
      <td>10.9516</td>
      <td>15.4305</td>
      <td>2.0339</td>
      <td>8.1267</td>
      <td>8.7889</td>
      <td>18.3560</td>
      <td>1.9518</td>
    </tr>
    <tr>
      <td>2</td>
      <td>train_2</td>
      <td>0</td>
      <td>8.6093</td>
      <td>-2.7457</td>
      <td>12.0805</td>
      <td>7.8928</td>
      <td>10.5825</td>
      <td>-9.0837</td>
      <td>6.9427</td>
      <td>14.6155</td>
      <td>...</td>
      <td>2.9057</td>
      <td>9.7905</td>
      <td>1.6704</td>
      <td>1.6858</td>
      <td>21.6042</td>
      <td>3.1417</td>
      <td>-6.5213</td>
      <td>8.2675</td>
      <td>14.7222</td>
      <td>0.3965</td>
    </tr>
    <tr>
      <td>3</td>
      <td>train_3</td>
      <td>0</td>
      <td>11.0604</td>
      <td>-2.1518</td>
      <td>8.9522</td>
      <td>7.1957</td>
      <td>12.5846</td>
      <td>-1.8361</td>
      <td>5.8428</td>
      <td>14.9250</td>
      <td>...</td>
      <td>4.4666</td>
      <td>4.7433</td>
      <td>0.7178</td>
      <td>1.4214</td>
      <td>23.0347</td>
      <td>-1.2706</td>
      <td>-2.9275</td>
      <td>10.2922</td>
      <td>17.9697</td>
      <td>-8.9996</td>
    </tr>
    <tr>
      <td>4</td>
      <td>train_4</td>
      <td>0</td>
      <td>9.8369</td>
      <td>-1.4834</td>
      <td>12.8746</td>
      <td>6.6375</td>
      <td>12.2772</td>
      <td>2.4486</td>
      <td>5.9405</td>
      <td>19.2514</td>
      <td>...</td>
      <td>-1.4905</td>
      <td>9.5214</td>
      <td>-0.1508</td>
      <td>9.1942</td>
      <td>13.2876</td>
      <td>-1.5121</td>
      <td>3.9267</td>
      <td>9.5031</td>
      <td>17.9974</td>
      <td>-8.8104</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 202 columns</p>
</div>


We can observe the basic condition of the data here. We can not infer any actual information from the name of the columns and the data, too. So it is better for us to find out more. Before that, first test whether there are missing values.

### 2.2 Check the Data

```python
# check the missing values
data_na=(train.isnull().sum()/len(train))*100
data_na=data_na.drop(data_na[data_na==0].index).sort_values(ascending=False)
missing_data=pd.DataFrame({'MissingRatio':data_na})
print(missing_data)
```

    Empty DataFrame
    Columns: [MissingRatio]
    Index: []
    

We can see there are no missing values.

```python
train.target.value_counts()
```




    0    179902
    1     20098
    Name: target, dtype: int64


The dataset may be quite unbalanced, we can see that almost 90 percent of the items have the target '0' while 10 percent are '1'.

We first extract all the features here.

```python
features=[col for col in train.columns if col not in ['ID_code','target']]
```

## 3 EDA
### 3.1 Check the Train-test Distribution
Before we doing our work, we might be extremely interested in the distribution of the dataset. The division of train set and test set should be as balanced as possible in all kinds of aspects. So we first examine this point.

First we check the mean values per row.
```python
# check the distribution
plt.figure(figsize=(18,10))
plt.title('Distribution of mean values per row in the train and test set')
sns.distplot(train[features].mean(axis=1),color='green',kde=True,bins=120,label='train')
sns.distplot(test[features].mean(axis=1),color='red',kde=True,bins=120,label='test')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_8_0.png)

Then we apply the same operation to the columns.

```python
plt.figure(figsize=(18,10))
plt.title('Distribution of mean values per column in the train and test set')
sns.distplot(train[features].mean(axis=0),color='purple',kde=True,bins=120,label='train')
sns.distplot(test[features].mean(axis=0),color='orange',kde=True,bins=120,label='test')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_9_0.png)


Besides, the standard deviation also worth examining.

```python
plt.figure(figsize=(18,10))
plt.title('Distribution of std values per rows in the train and test set')
sns.distplot(train[features].std(axis=1),color='black',kde=True,bins=120,label='train')
sns.distplot(test[features].std(axis=1),color='yellow',kde=True,bins=120,label='test')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_10_0.png)



```python
plt.figure(figsize=(18,10))
plt.title('Distribution of std values per column in the train and test set')
sns.distplot(train[features].std(axis=0),color='blue',kde=True,bins=120,label='train')
sns.distplot(test[features].std(axis=0),color='green',kde=True,bins=120,label='test')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_11_0.png)

We can see the data distribution of each row and column in the train set and the test set are almost balanced.

### 3.2 Check the Feature Correlation

```python
# check the feature correlation
corrmat=train.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,vmax=0.9,square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x25c953f7358>




![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_12_1.png)


We can see that the correlation between features are barely slight. Also it is worth to check the biggest correlation value.


```python
%%time
correlations=train[features].corr().unstack().sort_values(kind='quicksort').reset_index()
correlations=correlations[correlations['level_0']!=correlations['level_1']]
```

    Wall time: 16.2 s
    


```python
correlations.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level_0</th>
      <th>level_1</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>39790</td>
      <td>var_122</td>
      <td>var_132</td>
      <td>0.008956</td>
    </tr>
    <tr>
      <td>39791</td>
      <td>var_132</td>
      <td>var_122</td>
      <td>0.008956</td>
    </tr>
    <tr>
      <td>39792</td>
      <td>var_146</td>
      <td>var_169</td>
      <td>0.009071</td>
    </tr>
    <tr>
      <td>39793</td>
      <td>var_169</td>
      <td>var_146</td>
      <td>0.009071</td>
    </tr>
    <tr>
      <td>39794</td>
      <td>var_189</td>
      <td>var_183</td>
      <td>0.009359</td>
    </tr>
    <tr>
      <td>39795</td>
      <td>var_183</td>
      <td>var_189</td>
      <td>0.009359</td>
    </tr>
    <tr>
      <td>39796</td>
      <td>var_174</td>
      <td>var_81</td>
      <td>0.009490</td>
    </tr>
    <tr>
      <td>39797</td>
      <td>var_81</td>
      <td>var_174</td>
      <td>0.009490</td>
    </tr>
    <tr>
      <td>39798</td>
      <td>var_165</td>
      <td>var_81</td>
      <td>0.009714</td>
    </tr>
    <tr>
      <td>39799</td>
      <td>var_81</td>
      <td>var_165</td>
      <td>0.009714</td>
    </tr>
  </tbody>
</table>
</div>




```python
correlations.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level_0</th>
      <th>level_1</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>var_26</td>
      <td>var_139</td>
      <td>-0.009844</td>
    </tr>
    <tr>
      <td>1</td>
      <td>var_139</td>
      <td>var_26</td>
      <td>-0.009844</td>
    </tr>
    <tr>
      <td>2</td>
      <td>var_148</td>
      <td>var_53</td>
      <td>-0.009788</td>
    </tr>
    <tr>
      <td>3</td>
      <td>var_53</td>
      <td>var_148</td>
      <td>-0.009788</td>
    </tr>
    <tr>
      <td>4</td>
      <td>var_80</td>
      <td>var_6</td>
      <td>-0.008958</td>
    </tr>
    <tr>
      <td>5</td>
      <td>var_6</td>
      <td>var_80</td>
      <td>-0.008958</td>
    </tr>
    <tr>
      <td>6</td>
      <td>var_1</td>
      <td>var_80</td>
      <td>-0.008855</td>
    </tr>
    <tr>
      <td>7</td>
      <td>var_80</td>
      <td>var_1</td>
      <td>-0.008855</td>
    </tr>
    <tr>
      <td>8</td>
      <td>var_13</td>
      <td>var_2</td>
      <td>-0.008795</td>
    </tr>
    <tr>
      <td>9</td>
      <td>var_2</td>
      <td>var_13</td>
      <td>-0.008795</td>
    </tr>
  </tbody>
</table>
</div>

Well, the maximum absolute value of feature correlation is below 0.01. So we might not get any useful information from here.

### 3.3 Further Exploring

How about the distribution of each feature, here we try to print all the distribution plot on a single graph.

```python
# check the distribution of each feature
def plot_features(df1,df2,label1,label2,features):
    sns.set_style('whitegrid')
    plt.figure()
    fig,ax=plt.subplots(10,20,figsize=(18,22))
    i=0
    for feature in features:
        i+=1
        plt.subplot(10,20,i)
        sns.distplot(df1[feature],hist=False,label=label1)
        sns.distplot(df2[feature],hist=False,label=label2)
        plt.xlabel(feature,fontsize=9)
        locs, labels=plt.xticks()
        plt.tick_params(axis='x',which='major',labelsize=6,pad=-6)
        plt.tick_params(axis='y',which='major',labelsize=6)
    plt.show()
        
t0=train.loc[train['target']==0]
t1=train.loc[train['target']==1]
features=train.columns.values[2:202]
plot_features(t0,t1,'0','1',features)
```


    <Figure size 432x288 with 0 Axes>



![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_16_1.png)



```python
features=train.columns.values[2:202]
plot_features(train,test,'train','test',features)
```


    <Figure size 432x288 with 0 Axes>



![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_17_1.png)


All the features here are nearly balanced, it can make our work really convenient.

### 3.4 Other Statistical Indicators that Worth Checking
In order to have a more comprehensive grasp of the whole data, we can check every statistical indicators that might provide more

```python
# Distribution of min and max
t0=train.loc[train['target']==0]
t1=train.loc[train['target']==1]
plt.figure(figsize=(18,10))
plt.title('Distribution of min values per row in the train set')
sns.distplot(t0[features].min(axis=1),color='orange',kde=True,bins=120,label='0')
sns.distplot(t1[features].min(axis=1),color='red',kde=True,bins=120,label='1')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_18_0.png)



```python
plt.figure(figsize=(18,10))
plt.title('Distribution of min values per column in the train set')
sns.distplot(t0[features].min(axis=0),color='blue',kde=True,bins=120,label='0')
sns.distplot(t1[features].min(axis=0),color='green',kde=True,bins=120,label='1')
plt.legend()
plt.plot()
```




![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_19_1.png)



```python
plt.figure(figsize=(18,10))
plt.title('Distribution of max values per row in the train set')
sns.distplot(t0[features].max(axis=1),color='orange',kde=True,bins=120,label='0')
sns.distplot(t1[features].max(axis=1),color='red',kde=True,bins=120,label='1')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_20_0.png)



```python
plt.figure(figsize=(18,10))
plt.title('Distribution of max values per column in the train set')
sns.distplot(t0[features].max(axis=0),color='blue',kde=True,bins=120,label='0')
sns.distplot(t1[features].max(axis=0),color='green',kde=True,bins=120,label='1')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_21_0.png)



```python
# skewness and kurtosis
plt.figure(figsize=(18,10))
plt.title('Distribution of skew values per row in the train set')
sns.distplot(t0[features].skew(axis=1),color='orange',kde=True,bins=120,label='0')
sns.distplot(t1[features].skew(axis=1),color='red',kde=True,bins=120,label='1')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_22_0.png)



```python
plt.figure(figsize=(18,10))
plt.title('Distribution of skew values per column in the train set')
sns.distplot(t0[features].skew(axis=0),color='blue',kde=True,bins=120,label='0')
sns.distplot(t1[features].skew(axis=0),color='green',kde=True,bins=120,label='1')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_23_0.png)



```python
plt.figure(figsize=(18,10))
plt.title('Distribution of kurtosis values per row in the train set')
sns.distplot(t0[features].kurtosis(axis=1),color='orange',kde=True,bins=120,label='0')
sns.distplot(t1[features].kurtosis(axis=1),color='red',kde=True,bins=120,label='1')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_24_0.png)



```python
plt.figure(figsize=(18,10))
plt.title('Distribution of kurtosis values per column in the train set')
sns.distplot(t0[features].kurtosis(axis=0),color='blue',kde=True,bins=120,label='0')
sns.distplot(t1[features].kurtosis(axis=0),color='green',kde=True,bins=120,label='1')
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_25_0.png)

## 4 Feature Engineering and Modeling
### 4.1 Create New Features
We can add the statistical indicators to the dataset for modeling. They may be useful.

```python
# creating new features
idx=features=train.columns.values[2:202]
for df in [train,test]:
    df['sum']=df[idx].sum(axis=1)
    df['min']=df[idx].min(axis=1)
    df['max']=df[idx].max(axis=1)
    df['mean']=df[idx].mean(axis=1)
    df['std']=df[idx].std(axis=1)
    df['skew']=df[idx].skew(axis=1)
    df['kurt']=df[idx].kurtosis(axis=1)
    df['med']=df[idx].median(axis=1)
train[train.columns[202:]].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>std</th>
      <th>skew</th>
      <th>kurt</th>
      <th>med</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1456.3182</td>
      <td>-21.4494</td>
      <td>43.1127</td>
      <td>7.281591</td>
      <td>9.331540</td>
      <td>0.101580</td>
      <td>1.331023</td>
      <td>6.77040</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1415.3636</td>
      <td>-47.3797</td>
      <td>40.5632</td>
      <td>7.076818</td>
      <td>10.336130</td>
      <td>-0.351734</td>
      <td>4.110215</td>
      <td>7.22315</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1240.8966</td>
      <td>-22.4038</td>
      <td>33.8820</td>
      <td>6.204483</td>
      <td>8.753387</td>
      <td>-0.056957</td>
      <td>0.546438</td>
      <td>5.89940</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1288.2319</td>
      <td>-35.1659</td>
      <td>38.1015</td>
      <td>6.441160</td>
      <td>9.594064</td>
      <td>-0.480116</td>
      <td>2.630499</td>
      <td>6.70260</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1354.2310</td>
      <td>-65.4863</td>
      <td>41.1037</td>
      <td>6.771155</td>
      <td>11.287122</td>
      <td>-1.463426</td>
      <td>9.787399</td>
      <td>6.94735</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1272.3216</td>
      <td>-44.7257</td>
      <td>35.2664</td>
      <td>6.361608</td>
      <td>9.313012</td>
      <td>-0.920439</td>
      <td>4.581343</td>
      <td>6.23790</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1509.4490</td>
      <td>-29.9763</td>
      <td>39.9599</td>
      <td>7.547245</td>
      <td>9.246130</td>
      <td>-0.133489</td>
      <td>1.816453</td>
      <td>7.47605</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1438.5083</td>
      <td>-27.2543</td>
      <td>31.9043</td>
      <td>7.192541</td>
      <td>9.162558</td>
      <td>-0.300415</td>
      <td>1.174273</td>
      <td>6.97300</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1369.7375</td>
      <td>-31.7855</td>
      <td>42.4798</td>
      <td>6.848688</td>
      <td>9.837520</td>
      <td>0.084047</td>
      <td>1.997040</td>
      <td>6.32870</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1303.1155</td>
      <td>-39.3042</td>
      <td>34.4640</td>
      <td>6.515577</td>
      <td>9.943238</td>
      <td>-0.670024</td>
      <td>2.521160</td>
      <td>6.36320</td>
    </tr>
  </tbody>
</table>
</div>




```python
test[test.columns[201:]].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>std</th>
      <th>skew</th>
      <th>kurt</th>
      <th>med</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1416.6404</td>
      <td>-31.9891</td>
      <td>42.0248</td>
      <td>7.083202</td>
      <td>9.910632</td>
      <td>-0.088518</td>
      <td>1.871262</td>
      <td>7.31440</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1249.6860</td>
      <td>-41.1924</td>
      <td>35.6020</td>
      <td>6.248430</td>
      <td>9.541267</td>
      <td>-0.559785</td>
      <td>3.391068</td>
      <td>6.43960</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1430.2599</td>
      <td>-34.3488</td>
      <td>39.3654</td>
      <td>7.151300</td>
      <td>9.967466</td>
      <td>-0.135084</td>
      <td>2.326901</td>
      <td>7.26355</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1411.4447</td>
      <td>-21.4797</td>
      <td>40.3383</td>
      <td>7.057224</td>
      <td>8.257204</td>
      <td>-0.167741</td>
      <td>2.253054</td>
      <td>6.89675</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1423.7364</td>
      <td>-24.8254</td>
      <td>45.5510</td>
      <td>7.118682</td>
      <td>10.043542</td>
      <td>0.293484</td>
      <td>2.044943</td>
      <td>6.83375</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1273.1592</td>
      <td>-19.8952</td>
      <td>30.2647</td>
      <td>6.365796</td>
      <td>8.728466</td>
      <td>-0.031814</td>
      <td>0.113763</td>
      <td>5.83800</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1440.7387</td>
      <td>-18.7481</td>
      <td>37.4611</td>
      <td>7.203693</td>
      <td>8.676615</td>
      <td>-0.045407</td>
      <td>0.653782</td>
      <td>6.66335</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1429.5281</td>
      <td>-22.7363</td>
      <td>33.2387</td>
      <td>7.147640</td>
      <td>9.697687</td>
      <td>-0.017784</td>
      <td>0.713021</td>
      <td>7.44665</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1270.4978</td>
      <td>-17.4719</td>
      <td>28.1225</td>
      <td>6.352489</td>
      <td>8.257376</td>
      <td>-0.138639</td>
      <td>0.342360</td>
      <td>6.55820</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1271.6875</td>
      <td>-32.8776</td>
      <td>38.3319</td>
      <td>6.358437</td>
      <td>9.489171</td>
      <td>-0.354497</td>
      <td>1.934290</td>
      <td>6.83960</td>
    </tr>
  </tbody>
</table>
</div>



Now let's check the distributions of the new features.

```python
def plot_new_features(df1,df2,label1,label2,features):
    sns.set_style('whitegrid')
    plt.figure()
    fig,ax=plt.subplots(2,4,figsize=(18,8))
    i=0
    for feature in features:
        i+=1
        plt.subplot(2,4,i)
        sns.kdeplot(df1[feature],bw=0.5,label=label1)
        sns.kdeplot(df2[feature],bw=0.5,label=label2)
        plt.xlabel(feature,fontsize=11)
        locs,labels=plt.xticks()
        plt.tick_params(axis='x',which='major',labelsize=8)
        plt.tick_params(axis='y',which='major',labelsize=8)
    plt.show()
t0=train.loc[train['target']==0]
t1=train.loc[train['target']==1]
features=train.columns.values[202:]
plot_new_features(t0,t1,'0','1',features)
```


    <Figure size 432x288 with 0 Axes>



![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_28_1.png)



```python
print('Columns in train_set:{} Columns in test_set:{}'.format(len(train.columns),len(test.columns)))
```

    Columns in train_set:210 Columns in test_set:209
    

### 4.2 Training the Model
Here's a baseline model that uses LightGBM.

```python
# training the model
features=[col for col in train.columns if col not in ['ID_code','target']]
target=train['target']
param={
    'bagging_freq':5,
    'bagging_fraction':0.4,
    'boost':'gbdt',
    'boost_from_average':'false',
    'feature_fraction':0.05,
    'learning_rate':0.01,
    'max_depth':-1,
    'metric':'auc',
    'min_data_in_leaf':80,
    'min_sum_hessian_in_leaf':10.0,
    'num_leaves':13,
    'num_threads':8,
    'tree_learner':'serial',
    'objective':'binary',
    'verbosity':1
}
```


```python
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
```

    Fold 0
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.900229	valid_1's auc: 0.881617
    [2000]	training's auc: 0.91128	valid_1's auc: 0.889429
    [3000]	training's auc: 0.918765	valid_1's auc: 0.893439
    [4000]	training's auc: 0.924616	valid_1's auc: 0.895931
    [5000]	training's auc: 0.929592	valid_1's auc: 0.897636
    [6000]	training's auc: 0.933838	valid_1's auc: 0.898786
    [7000]	training's auc: 0.937858	valid_1's auc: 0.899318
    [8000]	training's auc: 0.941557	valid_1's auc: 0.899733
    [9000]	training's auc: 0.94517	valid_1's auc: 0.899901
    [10000]	training's auc: 0.948529	valid_1's auc: 0.900143
    [11000]	training's auc: 0.951807	valid_1's auc: 0.900281
    [12000]	training's auc: 0.954903	valid_1's auc: 0.900269
    [13000]	training's auc: 0.957815	valid_1's auc: 0.900107
    [14000]	training's auc: 0.960655	valid_1's auc: 0.89994
    Early stopping, best iteration is:
    [11603]	training's auc: 0.953681	valid_1's auc: 0.900347
    Fold 1
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.900404	valid_1's auc: 0.882765
    [2000]	training's auc: 0.911307	valid_1's auc: 0.889508
    [3000]	training's auc: 0.918917	valid_1's auc: 0.893254
    [4000]	training's auc: 0.924779	valid_1's auc: 0.895682
    [5000]	training's auc: 0.929704	valid_1's auc: 0.897004
    [6000]	training's auc: 0.933907	valid_1's auc: 0.897785
    [7000]	training's auc: 0.93784	valid_1's auc: 0.89799
    [8000]	training's auc: 0.941511	valid_1's auc: 0.898383
    [9000]	training's auc: 0.945033	valid_1's auc: 0.898701
    [10000]	training's auc: 0.94837	valid_1's auc: 0.898763
    [11000]	training's auc: 0.951605	valid_1's auc: 0.89877
    [12000]	training's auc: 0.954709	valid_1's auc: 0.898751
    [13000]	training's auc: 0.957618	valid_1's auc: 0.898634
    Early stopping, best iteration is:
    [10791]	training's auc: 0.950935	valid_1's auc: 0.89889
    Fold 2
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.90084	valid_1's auc: 0.87531
    [2000]	training's auc: 0.911957	valid_1's auc: 0.883717
    [3000]	training's auc: 0.919463	valid_1's auc: 0.888423
    [4000]	training's auc: 0.925317	valid_1's auc: 0.891101
    [5000]	training's auc: 0.930106	valid_1's auc: 0.892821
    [6000]	training's auc: 0.93436	valid_1's auc: 0.89362
    [7000]	training's auc: 0.938282	valid_1's auc: 0.89429
    [8000]	training's auc: 0.941897	valid_1's auc: 0.894544
    [9000]	training's auc: 0.945462	valid_1's auc: 0.894652
    [10000]	training's auc: 0.948798	valid_1's auc: 0.894821
    [11000]	training's auc: 0.952036	valid_1's auc: 0.894888
    [12000]	training's auc: 0.955136	valid_1's auc: 0.894657
    [13000]	training's auc: 0.958081	valid_1's auc: 0.894511
    [14000]	training's auc: 0.960904	valid_1's auc: 0.894327
    Early stopping, best iteration is:
    [11094]	training's auc: 0.952334	valid_1's auc: 0.894948
    Fold 3
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.900276	valid_1's auc: 0.882173
    [2000]	training's auc: 0.911124	valid_1's auc: 0.889171
    [3000]	training's auc: 0.918758	valid_1's auc: 0.893614
    [4000]	training's auc: 0.92463	valid_1's auc: 0.89627
    [5000]	training's auc: 0.929475	valid_1's auc: 0.897519
    [6000]	training's auc: 0.933971	valid_1's auc: 0.898018
    [7000]	training's auc: 0.937925	valid_1's auc: 0.898396
    [8000]	training's auc: 0.941684	valid_1's auc: 0.898475
    [9000]	training's auc: 0.945229	valid_1's auc: 0.898597
    [10000]	training's auc: 0.948626	valid_1's auc: 0.898725
    [11000]	training's auc: 0.951822	valid_1's auc: 0.898657
    [12000]	training's auc: 0.95488	valid_1's auc: 0.898504
    [13000]	training's auc: 0.957871	valid_1's auc: 0.898503
    Early stopping, best iteration is:
    [10712]	training's auc: 0.950891	valid_1's auc: 0.898759
    Fold 4
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.900213	valid_1's auc: 0.883231
    [2000]	training's auc: 0.911052	valid_1's auc: 0.890297
    [3000]	training's auc: 0.918649	valid_1's auc: 0.894252
    [4000]	training's auc: 0.924548	valid_1's auc: 0.896724
    [5000]	training's auc: 0.92951	valid_1's auc: 0.897923
    [6000]	training's auc: 0.93393	valid_1's auc: 0.898887
    [7000]	training's auc: 0.937896	valid_1's auc: 0.899048
    [8000]	training's auc: 0.941556	valid_1's auc: 0.899335
    [9000]	training's auc: 0.945033	valid_1's auc: 0.899469
    [10000]	training's auc: 0.94841	valid_1's auc: 0.899536
    [11000]	training's auc: 0.951679	valid_1's auc: 0.899371
    [12000]	training's auc: 0.954731	valid_1's auc: 0.899314
    [13000]	training's auc: 0.95771	valid_1's auc: 0.899024
    Early stopping, best iteration is:
    [10307]	training's auc: 0.949415	valid_1's auc: 0.899591
    Fold 5
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.899832	valid_1's auc: 0.887942
    [2000]	training's auc: 0.910762	valid_1's auc: 0.895511
    [3000]	training's auc: 0.918306	valid_1's auc: 0.899303
    [4000]	training's auc: 0.924334	valid_1's auc: 0.901522
    [5000]	training's auc: 0.929353	valid_1's auc: 0.902569
    [6000]	training's auc: 0.933747	valid_1's auc: 0.903396
    [7000]	training's auc: 0.937725	valid_1's auc: 0.903844
    [8000]	training's auc: 0.941422	valid_1's auc: 0.904181
    [9000]	training's auc: 0.944946	valid_1's auc: 0.904167
    [10000]	training's auc: 0.948326	valid_1's auc: 0.903872
    [11000]	training's auc: 0.951534	valid_1's auc: 0.903846
    Early stopping, best iteration is:
    [8408]	training's auc: 0.942866	valid_1's auc: 0.904303
    Fold 6
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.899935	valid_1's auc: 0.884744
    [2000]	training's auc: 0.910967	valid_1's auc: 0.892097
    [3000]	training's auc: 0.918595	valid_1's auc: 0.896277
    [4000]	training's auc: 0.924503	valid_1's auc: 0.898606
    [5000]	training's auc: 0.929414	valid_1's auc: 0.89991
    [6000]	training's auc: 0.933745	valid_1's auc: 0.900743
    [7000]	training's auc: 0.937714	valid_1's auc: 0.901066
    [8000]	training's auc: 0.94139	valid_1's auc: 0.900995
    [9000]	training's auc: 0.944926	valid_1's auc: 0.901016
    Early stopping, best iteration is:
    [6986]	training's auc: 0.937661	valid_1's auc: 0.901085
    Fold 7
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.899968	valid_1's auc: 0.881017
    [2000]	training's auc: 0.910826	valid_1's auc: 0.889131
    [3000]	training's auc: 0.918484	valid_1's auc: 0.893968
    [4000]	training's auc: 0.924432	valid_1's auc: 0.896794
    [5000]	training's auc: 0.929348	valid_1's auc: 0.898531
    [6000]	training's auc: 0.933656	valid_1's auc: 0.899541
    [7000]	training's auc: 0.937572	valid_1's auc: 0.899903
    [8000]	training's auc: 0.941255	valid_1's auc: 0.900259
    [9000]	training's auc: 0.944865	valid_1's auc: 0.900205
    [10000]	training's auc: 0.948314	valid_1's auc: 0.900135
    [11000]	training's auc: 0.951556	valid_1's auc: 0.900281
    [12000]	training's auc: 0.954647	valid_1's auc: 0.900202
    [13000]	training's auc: 0.957629	valid_1's auc: 0.900083
    [14000]	training's auc: 0.960473	valid_1's auc: 0.900019
    Early stopping, best iteration is:
    [11028]	training's auc: 0.951647	valid_1's auc: 0.900328
    Fold 8
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.899642	valid_1's auc: 0.889764
    [2000]	training's auc: 0.91067	valid_1's auc: 0.897589
    [3000]	training's auc: 0.918364	valid_1's auc: 0.901604
    [4000]	training's auc: 0.92421	valid_1's auc: 0.903614
    [5000]	training's auc: 0.929197	valid_1's auc: 0.904601
    [6000]	training's auc: 0.933471	valid_1's auc: 0.905101
    [7000]	training's auc: 0.93741	valid_1's auc: 0.905128
    [8000]	training's auc: 0.941136	valid_1's auc: 0.905215
    [9000]	training's auc: 0.944594	valid_1's auc: 0.905207
    [10000]	training's auc: 0.948042	valid_1's auc: 0.905092
    [11000]	training's auc: 0.951259	valid_1's auc: 0.905037
    Early stopping, best iteration is:
    [8028]	training's auc: 0.941228	valid_1's auc: 0.905247
    Fold 9
    Training until validation scores don't improve for 3000 rounds.
    [1000]	training's auc: 0.900193	valid_1's auc: 0.884426
    [2000]	training's auc: 0.911194	valid_1's auc: 0.891741
    [3000]	training's auc: 0.918785	valid_1's auc: 0.895999
    [4000]	training's auc: 0.924653	valid_1's auc: 0.8984
    [5000]	training's auc: 0.929607	valid_1's auc: 0.899584
    [6000]	training's auc: 0.933898	valid_1's auc: 0.900395
    [7000]	training's auc: 0.937896	valid_1's auc: 0.900785
    [8000]	training's auc: 0.941574	valid_1's auc: 0.900916
    [9000]	training's auc: 0.945132	valid_1's auc: 0.901081
    [10000]	training's auc: 0.948568	valid_1's auc: 0.901075
    [11000]	training's auc: 0.951714	valid_1's auc: 0.901069
    [12000]	training's auc: 0.954815	valid_1's auc: 0.901025
    [13000]	training's auc: 0.957792	valid_1's auc: 0.901129
    Early stopping, best iteration is:
    [10567]	training's auc: 0.950365	valid_1's auc: 0.901193
    CV score: 0.90025 
    

We are also interested in the feature importance. What feature counts most during the prediction process.

```python
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.show()
```


![png](https://github.com/Tinky2013/Kaggle/raw/master/006%20Santander/img/1/output_32_0.png)

## 5 Submission and Final Result

```python
submission=pd.DataFrame({"ID_code":test['ID_code'].values})
submission['target']=predictions
submission.to_csv(PATH+'submission.csv',index=False)
```

The submission's public score here is 0.89889 and the private score is 0.90021.
