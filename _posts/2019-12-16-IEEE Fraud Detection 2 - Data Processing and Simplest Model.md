---
layout:     post                    # 使用的布局（不需要改）
title:      IEEE Fraud Detection 2 - Data Processing and Simplest Model               # 标题 
subtitle:   Dealing with the massive data
date:       2019-12-16             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Kaggle
--- 

# IEEE Fraud Detection 2 - Data Processing and Simplest Model

In this section we will do some data cleaning jobs, which is necessary before we build our models.

We have seen the distributions of all kinds of features in the last passage, so now we can process the data with the knowledge we got from the visual charts.

## 1 Prepare the Data

### 1.1 Import and Merge the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import warnings
import time
warnings.filterwarnings('ignore')
```


```python
PATH='E:/kaggle/ieee-fraud-detection/'
tran_tr=pd.read_csv(PATH+'train_transaction.csv')
iden_tr=pd.read_csv(PATH+'train_identity.csv')
tran_ts=pd.read_csv(PATH+'test_transaction.csv')
iden_ts=pd.read_csv(PATH+'test_identity.csv')
```


```python
train=pd.merge(tran_tr,iden_tr,on='TransactionID',how='left')
test=pd.merge(tran_ts,iden_ts,on='TransactionID',how='left')
print(f'Train: {train.shape[0]} rows {train.shape[1]} columns.')
print(f'Test: {test.shape[0]} rows {test.shape[1]} columns.')
```

    Train: 590540 rows 434 columns.
    Test: 506691 rows 433 columns.
    
We can see that the dataset is really large, so while processing the data we should take the efficiency of the methods into account.

### 1.2 Reduce the Memory

First we delete the dataset that we won't use in the subsequent steps.

```python
del tran_tr, iden_tr, tran_ts, iden_ts
```

This function is used to reduce the memory usage of the dataset.

```python
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    # reduce memory usage
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type

            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
reduce_mem_usage(train)
reduce_mem_usage(test)
```

    Memory usage of dataframe is 1959.88 MB
    Memory usage after optimization is: 1049.21 MB
    Decreased by 46.5%
    Memory usage of dataframe is 1677.73 MB
    Memory usage after optimization is: 899.75 MB
    Decreased by 46.4%
    

We can see that the memory usage has largely reduced and it will process faster when we build the model.

## 2 Data Preprocessing

### 2.1 Label Encoding

In this part we need to convert the categorial features to integers, so we use the **LabelEncoder** in sklearn.

```python
# label encoding

# convert the categorical features to integer code

for col in test.columns:
    if train[col].dtype=='object' or test[col].dtype=='object':
        lb=LabelEncoder()
        lb.fit(list(train[col].values)+list(test[col].values))
        train[col]=lb.transform(list(train[col].values))
        test[col]=lb.transform(list(test[col].values))
print(f'Train: {train.shape[0]} rows {train.shape[1]} columns.')
print(f'Test: {test.shape[0]} rows {test.shape[1]} columns.')
```

    Train: 590540 rows 434 columns.
    Test: 506691 rows 433 columns.
    

### 2.2 Drop Useless Columns and Fill the NaN

We can see that we have too many features in this dataset, however, we have done some work before and observed how those data points distribute. So we may choose some useful columns for our model in this part.

```python
useful_cols=['isFraud','TransactionAmt','ProductCD','card1','card2','card3','card4','card5','card6',
            'addr1','addr2','dist1','P_emaildomain','R_emaildomain','C1','C2','C3','C4','C5','C6',
            'C7','C8','C9','C10','C11','C12','C13','C14','V95','V96','V97','V98','V99','V100','V101',
            'V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114',
            'V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127',
            'V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V279','V280','V281',
            'V282','V283','V284','V285','V286','V287','V288','V289','V290','V291','V292','V293','V294',
            'V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306','V307',
            'V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320',
            'DeviceType','DeviceInfo']
```

Also, for the missing value, we need to fill those columns with mean or mode.

```python
def drop_columns(data):
    for col in data.columns:
        if col not in useful_cols:
            data.drop(col,axis=1,inplace=True)

drop_columns(train)
drop_columns(test)
```


```python
def fill_na_mean(data):
    col=['addr1','card2','card3','card5','V95','V96','V97','V98','V99','V100','V101',
            'V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114',
            'V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127',
            'V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V279','V280','V281',
            'V282','V283','V284','V285','V286','V287','V288','V289','V290','V291','V292','V293','V294',
            'V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306','V307',
            'V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320']
    for c in col:
        data[c].fillna(data[c].mean(),inplace=True)

def fill_na_mean_test(data):
    col=['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']
    for c in col:
        data[c].fillna(data[c].mean(),inplace=True)
    # C has missing values in test set only
    
def fill_na_mode(data):
    col=['addr2','dist1']
    for c in col:
        data[c].fillna(data[c].mode()[0],inplace=True)
    
```


```python
fill_na_mean(train)
fill_na_mean(test)
fill_na_mean_test(test)      
fill_na_mode(train)
fill_na_mode(test)
```

After process the data, we need to check whether there exists any missing value.

```python
# check missing values
def check(data):
    T_na=(data.isnull().sum()/len(data))*100
    T_na=T_na.drop(T_na[T_na==0].index).sort_values(ascending=False)
    T_mis=pd.DataFrame({'MissingRatio':T_na})
    print(T_mis)
check(train)    
check(test)
# there are no missing value now
```

    Empty DataFrame
    Columns: [MissingRatio]
    Index: []
    Empty DataFrame
    Columns: [MissingRatio]
    Index: []
    

We can see that there aren't missing value in our dataset now.

### 2.3 Log Transform the Data

For the column **TransactionAmt**, we can do the log transform to make the distribution of the data closer to a normal distribution.

```python
def data_transform(data):
    log_trans_col=['TransactionAmt']
    for c in log_trans_col:
        data[c]=np.log(data[c]+1)

data_transform(train)
data_transform(test)
```


```python
print(f'Train: {train.shape[0]} rows {train.shape[1]} columns.')
print(f'Test: {test.shape[0]} rows {test.shape[1]} columns.')
```

    Train: 590540 rows 115 columns.
    Test: 506691 rows 114 columns.
    


```python
# you can store the file for safety in this part

# Atrain=train

# Atest=test

# Atrain.to_csv(PATH+'Atrain.csv',index=0)

# Atest.to_csv(PATH+'Atest.csv',index=0)

# train=pd.read_csv(PATH+'Atrain.csv')

# test=pd.read_csv(PATH+'Atest.csv')
```

## 3 Build the Models

In this passage we only try a few simple models in sklearn and observe the output. Further improvement of the models will be discussed in the serial articles.

```python
import sklearn.metrics as metric
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
```

### 3.1 One-hot Encoding

For those categorial features, it is suitable to do one-hot encoding, which will transform those features into vectors. (You can also do this step just after importing the data)

```python
# one-hot
def One_hot(df):
    oh=OneHotEncoder()
    df_col=oh.fit_transform(np.array(df['ProductCD'].astype(str)).reshape(-1,1))
    df_col=pd.DataFrame(df_col.todense())
    df_col.columns=['Pro_1','Pro_2','Pro_3','Pro_4','Pro_5']
    df=pd.concat([df,df_col],axis=1)
    df.drop(['ProductCD'],axis=1,inplace=True)

    oh=OneHotEncoder()
    df_col=oh.fit_transform(np.array(df['card4'].astype(str)).reshape(-1,1))
    df_col=pd.DataFrame(df_col.todense())
    df_col.columns=['c4_1','c4_2','c4_3','c4_4','c4_5']
    df=pd.concat([df,df_col],axis=1)
    df.drop(['card4'],axis=1,inplace=True)

    oh=OneHotEncoder()
    df_col=oh.fit_transform(np.array(df['DeviceType'].astype(str)).reshape(-1,1))
    df_col=pd.DataFrame(df_col.todense())
    df_col.columns=['de_1','de_2','de_3']
    df=pd.concat([df,df_col],axis=1)
    df.drop(['DeviceType'],axis=1,inplace=True)
    print(type(df))
    return df
```


```python
train=One_hot(train)
test=One_hot(test)
print(f'Train: {train.shape[0]} rows {train.shape[1]} columns.')
print(f'Test: {test.shape[0]} rows {test.shape[1]} columns.')
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    Train: 590540 rows 125 columns.
    Test: 506691 rows 124 columns.
    

### 3.2 Build the Models

In this step, we need to split the dataset into training set and testing set.

```python
# build the model
X = train.loc[:,"TransactionAmt":]
y = train.loc[:,'isFraud']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=0)
```

Now we can try several models:

#### 3.2.1 Random Forest


```python
start =time.clock()
rft = RandomForestClassifier(criterion='entropy',max_depth=8,n_estimators=100,verbose=0)
rft.fit(X_train, y_train)
predVal = rft.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[142452    132]
     [  4023   1028]]
    0.9718562671453246
    Running time: 86.99202809999952 Seconds
    

We can see that the accruacy is relatively high. However, don't forget we are dealing with a really unbalanced dataset! So if we just predict all the target values are **Not Fraud**, we will get high accracy score, too. So it is necessary for us to check the confusion matrix. We can see that this algorithm doesn't perform well in this case.

#### 3.2.2 K Nearest Neighbor

```python
start =time.clock()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
predVal = clf.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[141658    926]
     [  3511   1540]]
    0.9699461509804586
    Running time: 1646.5342081 Seconds
    

#### 3.2.3 Logistic Regression

```python
start =time.clock()
log = LogisticRegression()
log.fit(X_train, y_train)
predVal = log.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[142527     57]
     [  4990     61]]
    0.9658143394181596
    Running time: 101.20429169999989 Seconds
    

#### 3.2.4 SVM Classifier

```python
start =time.clock()
svc = LinearSVC()
svc.fit(X_train, y_train)
predVal = svc.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[142418    166]
     [  4792    259]]
    0.9664171774985606
    Running time: 226.08227179999994 Seconds
    

#### 3.2.5 Decision Tree

```python
start =time.clock()
tree = DecisionTreeClassifier(max_depth=8,random_state=0)
tree.fit(X_train, y_train)
predVal = tree.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[142229    355]
     [  3522   1529]]
    0.9737392894638806
    Running time: 6.9810839999991 Seconds
    
#### 3.2.6 Gradient Boosting Tree

```python
start =time.clock()
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
predVal = gbrt.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[142342    242]
     [  3650   1401]]
    0.9736376875402174
    Running time: 290.94794999999976 Seconds
    

#### 3.2.7 MLP

```python
start =time.clock()
mlp = MLPClassifier(solver='lbfgs', random_state=0)
mlp.fit(X_train, y_train)
predVal = mlp.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[140823   1761]
     [  4667    384]]
    0.956460188979578
    Running time: 373.1303016999991 Seconds
    

#### 3.2.8 XGBoost

```python
start =time.clock()
XGB = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9
                       )

XGB.fit(X_train, y_train)
predVal = XGB.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[142354    230]
     [  2339   2712]]
    0.9825989772073018
    Running time: 1444.0721217999999 Seconds
    

#### 3.2.9 LightGBM

```python
start =time.clock()
GBM = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         verbose = 0,
                         learning_rate = 0.05,
                         max_depth=8,
                         n_estimators = 1500,
                         num_leaves = 256,
                         max_bin = 255,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
GBM.fit(X_train, y_train)
predVal = GBM.predict(X_test)
actVal = y_test.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))
end = time.clock()
print('Running time: %s Seconds'%(end-start))
```

    [[142354    230]
     [  2038   3013]]
    0.9846377891421411
    Running time: 280.8494519000001 Seconds
    
### 3.3 Conclusion of the Algorithm

We can see that among these algorithms, XGBoost and LightGBM perform well and we have discussed these two algorithms in other passages. However, if we want to improve the performance, beside adjusting the parameters, we need to do taxing feature engineering as this is a question based on actual commercial environment.

## 4 Make Submissions

```python
# make submissions

# y_preds=rft.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv'index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_rf.csv')



# y_preds=clf.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_KNeighbor.csv')



# y_preds=log.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_log.csv')



# y_preds=svc.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

#submission.to_csv(PATH+'submission11_svc.csv')



# y_preds=tree.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_tree.csv')



# y_preds=gbrt.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_gbrt.csv')



# y_preds=mlp.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_mlp.csv')



# y_preds=XGB.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_xgb.csv')



# y_preds=GBM.predict(Atest)

# submission=pd.read_csv(PATH+'sample_submission.csv',index_col='TransactionID')

# submission['isFraud'] = y_preds

# submission.to_csv(PATH+'submission11_gbm.csv')
```
