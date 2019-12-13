---
layout:     post                    # 使用的布局（不需要改）
title:      IEEE Fraud Detection 1——EDA               # 标题 
subtitle:   Visualization, using Seaborn #副标题
date:       2019-12-02             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Kaggle
--- 

# IEEE Fraud Detection: EDA

## 1 Description

In this competition, you’ll benchmark machine learning models on a challenging large-scale dataset. The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features. You also have the opportunity to create new features to improve your results.

If successful, you’ll improve the efficacy of fraudulent transaction alerts for millions of people around the world, helping hundreds of thousands of businesses reduce their fraud loss and increase their revenue. And of course, you will save party people just like you the hassle of false positives.

We need to predict the probability that an online transaction is fraudulent, as denoted by the binary target isFraud. The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

## 2 Prepare the Data
### 2.1 Import and Preparation

First we import the necessary libraries that we need.

```python
from sklearn import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
```


```python
PATH='E:/kaggle/ieee-fraud-detection/'
tran_tr=pd.read_csv(PATH+'train_transaction.csv')
iden_tr=pd.read_csv(PATH+'train_identity.csv')
tran_ts=pd.read_csv(PATH+'test_transaction.csv')
iden_ts=pd.read_csv(PATH+'test_identity.csv')
```

To make the process simplier, we can merge the two datasets: identities and transaction.

```python
train=pd.merge(tran_tr,iden_tr,on='TransactionID',how='left')
test=pd.merge(tran_ts,iden_ts,on='TransactionID',how='left')
print(f'Train: {train.shape[0]} rows {train.shape[1]} columns.')
print(f'Test: {test.shape[0]} rows {test.shape[1]} columns.')
```

### 2.2 Check the Data

We need to check out the missing value in the new dataset.

```python
# check the missing values

data_na=(train.isnull().sum()/len(train))*100
missing_data=pd.DataFrame({'MissingRatio':data_na})
```


```python
print((missing_data[0:50]))
print((missing_data[50:100]))
print((missing_data[100:150]))
print((missing_data[150:200]))
print((missing_data[200:250]))
print((missing_data[250:300]))
print((missing_data[300:350]))
print((missing_data[350:400]))
print((missing_data[400:434]))
```

                    MissingRatio
    TransactionID       0.000000
    isFraud             0.000000
    TransactionDT       0.000000
    TransactionAmt      0.000000
    ProductCD           0.000000
    card1               0.000000
    card2               1.512683
    card3               0.265012
    card4               0.267044
    card5               0.721204
    card6               0.266028
    addr1              11.126427
    addr2              11.126427
    dist1              59.652352
    dist2              93.628374
    P_emaildomain      15.994852
    R_emaildomain      76.751617
    C1                  0.000000
    C2                  0.000000
    C3                  0.000000
    C4                  0.000000
    C5                  0.000000
    C6                  0.000000
    C7                  0.000000
    C8                  0.000000
    C9                  0.000000
    C10                 0.000000
    C11                 0.000000
    C12                 0.000000
    C13                 0.000000
    C14                 0.000000
    D1                  0.214888
    D2                 47.549192
    D3                 44.514851
    D4                 28.604667
    D5                 52.467403
    D6                 87.606767
    D7                 93.409930
    D8                 87.312290
    D9                 87.312290
    D10                12.873302
    D11                47.293494
    D12                89.041047
    D13                89.509263
    D14                89.469469
    D15                15.090087
    M1                 45.907136
    M2                 45.907136
    M3                 45.907136
    M4                 47.658753
         MissingRatio
    M5      59.349409
    M6      28.678836
    M7      58.635317
    M8      58.633115
    M9      58.633115
    V1      47.293494
    V2      47.293494
    V3      47.293494
    V4      47.293494
    V5      47.293494
    V6      47.293494
    V7      47.293494
    V8      47.293494
    V9      47.293494
    V10     47.293494
    V11     47.293494
    V12     12.881939
    V13     12.881939
    V14     12.881939
    V15     12.881939
    V16     12.881939
    V17     12.881939
    V18     12.881939
    V19     12.881939
    V20     12.881939
    V21     12.881939
    V22     12.881939
    V23     12.881939
    V24     12.881939
    V25     12.881939
    V26     12.881939
    V27     12.881939
    V28     12.881939
    V29     12.881939
    V30     12.881939
    V31     12.881939
    V32     12.881939
    V33     12.881939
    V34     12.881939
    V35     28.612626
    V36     28.612626
    V37     28.612626
    V38     28.612626
    V39     28.612626
    V40     28.612626
    V41     28.612626
    V42     28.612626
    V43     28.612626
    V44     28.612626
    V45     28.612626
         MissingRatio
    V46     28.612626
    V47     28.612626
    V48     28.612626
    V49     28.612626
    V50     28.612626
    V51     28.612626
    V52     28.612626
    V53     13.055170
    V54     13.055170
    V55     13.055170
    V56     13.055170
    V57     13.055170
    V58     13.055170
    V59     13.055170
    V60     13.055170
    V61     13.055170
    V62     13.055170
    V63     13.055170
    V64     13.055170
    V65     13.055170
    V66     13.055170
    V67     13.055170
    V68     13.055170
    V69     13.055170
    V70     13.055170
    V71     13.055170
    V72     13.055170
    V73     13.055170
    V74     13.055170
    V75     15.098723
    V76     15.098723
    V77     15.098723
    V78     15.098723
    V79     15.098723
    V80     15.098723
    V81     15.098723
    V82     15.098723
    V83     15.098723
    V84     15.098723
    V85     15.098723
    V86     15.098723
    V87     15.098723
    V88     15.098723
    V89     15.098723
    V90     15.098723
    V91     15.098723
    V92     15.098723
    V93     15.098723
    V94     15.098723
    V95      0.053172
          MissingRatio
    V96       0.053172
    V97       0.053172
    V98       0.053172
    V99       0.053172
    V100      0.053172
    V101      0.053172
    V102      0.053172
    V103      0.053172
    V104      0.053172
    V105      0.053172
    V106      0.053172
    V107      0.053172
    V108      0.053172
    V109      0.053172
    V110      0.053172
    V111      0.053172
    V112      0.053172
    V113      0.053172
    V114      0.053172
    V115      0.053172
    V116      0.053172
    V117      0.053172
    V118      0.053172
    V119      0.053172
    V120      0.053172
    V121      0.053172
    V122      0.053172
    V123      0.053172
    V124      0.053172
    V125      0.053172
    V126      0.053172
    V127      0.053172
    V128      0.053172
    V129      0.053172
    V130      0.053172
    V131      0.053172
    V132      0.053172
    V133      0.053172
    V134      0.053172
    V135      0.053172
    V136      0.053172
    V137      0.053172
    V138     86.123717
    V139     86.123717
    V140     86.123717
    V141     86.123717
    V142     86.123717
    V143     86.122701
    V144     86.122701
    V145     86.122701
          MissingRatio
    V146     86.123717
    V147     86.123717
    V148     86.123717
    V149     86.123717
    V150     86.122701
    V151     86.122701
    V152     86.122701
    V153     86.123717
    V154     86.123717
    V155     86.123717
    V156     86.123717
    V157     86.123717
    V158     86.123717
    V159     86.122701
    V160     86.122701
    V161     86.123717
    V162     86.123717
    V163     86.123717
    V164     86.122701
    V165     86.122701
    V166     86.122701
    V167     76.355370
    V168     76.355370
    V169     76.323534
    V170     76.323534
    V171     76.323534
    V172     76.355370
    V173     76.355370
    V174     76.323534
    V175     76.323534
    V176     76.355370
    V177     76.355370
    V178     76.355370
    V179     76.355370
    V180     76.323534
    V181     76.355370
    V182     76.355370
    V183     76.355370
    V184     76.323534
    V185     76.323534
    V186     76.355370
    V187     76.355370
    V188     76.323534
    V189     76.323534
    V190     76.355370
    V191     76.355370
    V192     76.355370
    V193     76.355370
    V194     76.323534
    V195     76.323534
          MissingRatio
    V196     76.355370
    V197     76.323534
    V198     76.323534
    V199     76.355370
    V200     76.323534
    V201     76.323534
    V202     76.355370
    V203     76.355370
    V204     76.355370
    V205     76.355370
    V206     76.355370
    V207     76.355370
    V208     76.323534
    V209     76.323534
    V210     76.323534
    V211     76.355370
    V212     76.355370
    V213     76.355370
    V214     76.355370
    V215     76.355370
    V216     76.355370
    V217     77.913435
    V218     77.913435
    V219     77.913435
    V220     76.053104
    V221     76.053104
    V222     76.053104
    V223     77.913435
    V224     77.913435
    V225     77.913435
    V226     77.913435
    V227     76.053104
    V228     77.913435
    V229     77.913435
    V230     77.913435
    V231     77.913435
    V232     77.913435
    V233     77.913435
    V234     76.053104
    V235     77.913435
    V236     77.913435
    V237     77.913435
    V238     76.053104
    V239     76.053104
    V240     77.913435
    V241     77.913435
    V242     77.913435
    V243     77.913435
    V244     77.913435
    V245     76.053104
          MissingRatio
    V246     77.913435
    V247     77.913435
    V248     77.913435
    V249     77.913435
    V250     76.053104
    V251     76.053104
    V252     77.913435
    V253     77.913435
    V254     77.913435
    V255     76.053104
    V256     76.053104
    V257     77.913435
    V258     77.913435
    V259     76.053104
    V260     77.913435
    V261     77.913435
    V262     77.913435
    V263     77.913435
    V264     77.913435
    V265     77.913435
    V266     77.913435
    V267     77.913435
    V268     77.913435
    V269     77.913435
    V270     76.053104
    V271     76.053104
    V272     76.053104
    V273     77.913435
    V274     77.913435
    V275     77.913435
    V276     77.913435
    V277     77.913435
    V278     77.913435
    V279      0.002032
    V280      0.002032
    V281      0.214888
    V282      0.214888
    V283      0.214888
    V284      0.002032
    V285      0.002032
    V286      0.002032
    V287      0.002032
    V288      0.214888
    V289      0.214888
    V290      0.002032
    V291      0.002032
    V292      0.002032
    V293      0.002032
    V294      0.002032
    V295      0.002032
           MissingRatio
    V296       0.214888
    V297       0.002032
    V298       0.002032
    V299       0.002032
    V300       0.214888
    V301       0.214888
    V302       0.002032
    V303       0.002032
    V304       0.002032
    V305       0.002032
    V306       0.002032
    V307       0.002032
    V308       0.002032
    V309       0.002032
    V310       0.002032
    V311       0.002032
    V312       0.002032
    V313       0.214888
    V314       0.214888
    V315       0.214888
    V316       0.002032
    V317       0.002032
    V318       0.002032
    V319       0.002032
    V320       0.002032
    V321       0.002032
    V322      86.054967
    V323      86.054967
    V324      86.054967
    V325      86.054967
    V326      86.054967
    V327      86.054967
    V328      86.054967
    V329      86.054967
    V330      86.054967
    V331      86.054967
    V332      86.054967
    V333      86.054967
    V334      86.054967
    V335      86.054967
    V336      86.054967
    V337      86.054967
    V338      86.054967
    V339      86.054967
    id_01     75.576083
    id_02     76.145223
    id_03     88.768923
    id_04     88.768923
    id_05     76.823755
    id_06     76.823755
                MissingRatio
    id_07          99.127070
    id_08          99.127070
    id_09          87.312290
    id_10          87.312290
    id_11          76.127273
    id_12          75.576083
    id_13          78.440072
    id_14          86.445626
    id_15          76.126088
    id_16          78.098012
    id_17          76.399736
    id_18          92.360721
    id_19          76.408372
    id_20          76.418024
    id_21          99.126393
    id_22          99.124699
    id_23          99.124699
    id_24          99.196159
    id_25          99.130965
    id_26          99.125715
    id_27          99.124699
    id_28          76.127273
    id_29          76.127273
    id_30          86.865411
    id_31          76.245132
    id_32          86.861855
    id_33          87.589494
    id_34          86.824771
    id_35          76.126088
    id_36          76.126088
    id_37          76.126088
    id_38          76.126088
    DeviceType     76.155722
    DeviceInfo     79.905510
    

As we can see, some columns contain massive missing point, even up to 90% percent, which will bring a lot of trouble to our processing. On the contrary, some columns don't contain any missing value.

Then we check the target value.

```python
# target value

train.isFraud.value_counts()
```




    0    569877
    1     20663
    Name: isFraud, dtype: int64


As we can see, this dataset is unbalanced. Although we are dealing with a binary classification problem, we may not evaluate the outcome by accuracy only, as the accuracy will appear to be high if we classify all the data as 'Not fraud'.


```python
'''
# Just for data checking

for col in train.columns:
    def check(mit):
        print(mit.value_counts())
        print(mit.isnull().value_counts())
    check(test.TransactionAmt)
'''
```

### 2.3 Summary

After some simple analytical work, we can see that this dataset contains all kinds of data:

* TransactionDT/TransactionAmt：Continuous variable

* ProductCD：Categorical variable (Five classes)

* card1/2/3/5：Features with integers

* card4/6：Categorical variable (Four classes, very unbalanced)

* addr1/2：Strange features with integers

* dist1/2：Strange features with integers (Many missing points, extremely unbalanced)

* P_emaildomain：Categorical variable

* R_emaildomain：Categorical variable (With lots of missing values)

* C1-14：Strange features with integers (Some are really unbalanced, but without missing values)

* D1-15：Strange features with integers (Some are really unbalanced, some have lots of missing values)

* M1-9：Categorical features (T or F)

* V1-339：Strange features with integers (Some are really unbalanced, some have lots of missing values)

* id_01-38：Contain all kinds of features

* Device Type/Info：Categorical features

## 3 Data Visualization

### 3.1 TransectionDT/TransectionAmt

We need to do some 'Artisitic Work' in this section, first let's find out the distribution of **TransactionDT** in the training set and testing set.

```python
# data visualization
# TransactionDT

plt.figure(figsize=(18,10))
plt.title('Distribution of TransactionDT')
sns.distplot(train.TransactionDT,color='red',kde=True,label='train')
sns.distplot(test.TransactionDT,color='blue',kde=True,label='test')
plt.legend()
plt.show()
# 30 Days gap between train and test
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_8_0.png)


We can see that this feature represents the time and there's a gap around 30 days between the training set and testing set.

Then we visualize the distribution of TransactionAmt, we also plot the distribution in both training set and testing set.

```python
# TransactionAmt & log-TransactionAmt

plt.figure(figsize=(18,10))
plt.suptitle('Distribution of TransactionAmt')
plt.subplot(221)
g1=sns.distplot(train.TransactionAmt,color='red',kde=True,bins=960,label='train')
g1=sns.distplot(test.TransactionAmt,color='blue',kde=True,bins=960,label='test')
g1.set_title("Transaction Amount Distribuition", fontsize=18)
plt.axis([-10,3000,-0.0001,0.01])
plt.legend()

plt.subplot(222)
g2=sns.distplot(np.log(train.TransactionAmt),color='red',kde=True,bins=180,label='train')
g2=sns.distplot(np.log(test.TransactionAmt),color='blue',kde=True,bins=180,label='test')
g2.set_title("Transaction Amount Distribuition(Log)", fontsize=18)
plt.axis([-1,10,-0.01,1.5])
plt.legend()
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_9_0.png)

We can see that the data has obvious amesiality, so we use log function to transform the data, after the transformation we can see that the distribution of data becomes clearer.

### 3.2 Product/Card

**Product CD** is a categorical feature, and we calculate the total for each category, and we separate the two types of target values.

```python
# product CD

fig,ax=plt.subplots(1,1,figsize=(11,11))
sns.countplot(x="ProductCD", ax=ax, hue = "isFraud", data=train)
ax.set_title('ProductCD train', fontsize=14)
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_10_0.png)

**cards 4** and **cards 6** are also categorical features so we do the same process.

```python
# cards 4 & 6

fig,ax=plt.subplots(1,4,figsize=(26,8))
sns.countplot(x="card4",ax=ax[0],data=train.loc[train['isFraud']==0])
ax[0].set_title("card4 isFraud=0",fontsize=14)
sns.countplot(x="card4",ax=ax[1],data=train.loc[train['isFraud']==1])
ax[1].set_title("card4 isFraud=1",fontsize=14)
sns.countplot(x="card6",ax=ax[2],data=train.loc[train['isFraud']==0])
ax[2].set_title("card6 isFraud=0",fontsize=14)
sns.countplot(x="card6",ax=ax[3],data=train.loc[train['isFraud']==1])
ax[3].set_title("card6 isFraud=1",fontsize=14)
```




    Text(0.5, 1.0, 'card6 isFraud=1')




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_11_1.png)

**card 1,2,3,5** are numerical features, so we plot the distributions.

```python
# card 1,2,3,5

fig,ax=plt.subplots(1,4,figsize=(26,8))
sns.distplot(train.loc[train['isFraud']==0]['card1'],bins=50,ax=ax[0],label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['card1'],bins=50,ax=ax[0],label='isFraud==1')
ax[0].legend()
ax[0].set_title("card1")
sns.distplot(train.loc[train['isFraud']==0]['card2'],bins=50,ax=ax[1],label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['card2'],bins=50,ax=ax[1],label='isFraud==1')
ax[1].legend()
ax[1].set_title("card2")
sns.distplot(train.loc[train['isFraud']==0]['card3'],bins=50,ax=ax[2],label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['card3'],bins=50,ax=ax[2],label='isFraud==1')
ax[2].legend()
ax[2].set_title("card3")
sns.distplot(train.loc[train['isFraud']==0]['card5'],bins=50,ax=ax[3],label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['card5'],bins=50,ax=ax[3],label='isFraud==1')
ax[3].legend()
ax[3].set_title("card5")

```




    Text(0.5, 1.0, 'card5')




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_12_1.png)

We can see that the distributions may show the correlation with fraud or not.

### 3.3 Addr/Dist

Let see the order columns in the data set.

```python
# addr1/2

fig,ax=plt.subplots(2,1,figsize=(16,12))
sns.distplot(train.loc[train['isFraud']==0]['addr1'],ax=ax[0],bins=200,label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['addr1'],ax=ax[0],bins=200,label='isFraud==1')
ax[0].legend()
ax[0].set_title('addr1')
sns.distplot(train.loc[train['isFraud']==0]['addr2'],ax=ax[1],bins=200,label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['addr2'],ax=ax[1],bins=200,label='isFraud==1')
ax[1].legend()
ax[1].set_title('addr2')
```




    Text(0.5, 1.0, 'addr2')




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_13_1.png)

Addr1 is relatively decentralized, but addr2 basically gathering around a particular number, that's weird before we know the actual meaning of the data.

What about the **dist** features?

```python
# dist1/2

fig,ax=plt.subplots(2,1,figsize=(16,12))
sns.distplot(train.loc[train['isFraud']==0]['dist1'],ax=ax[0],bins=200,label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['dist1'],ax=ax[0],bins=200,label='isFraud==1')
ax[0].legend()
ax[0].set_title('dist1')
sns.distplot(train.loc[train['isFraud']==0]['dist2'],ax=ax[1],bins=200,label='isFraud==0')
sns.distplot(train.loc[train['isFraud']==1]['dist2'],ax=ax[1],bins=200,label='isFraud==1')
ax[1].legend()
ax[1].set_title('dist2')
```




    Text(0.5, 1.0, 'dist2')




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_14_1.png)

We can see that most of the values for these two characteristics are zero, and there are a few other values lie scattered.

### 3.4 Emails

```python
# email

fig,ax=plt.subplots(1,3,figsize=(26,10))
sns.countplot(y="P_emaildomain",ax=ax[0],data=train)
ax[0].set_title("P_emaildomain",fontsize=14)
sns.countplot(y="P_emaildomain",ax=ax[1],data=train.loc[train['isFraud']==1])
ax[1].set_title("P_emaildomain isFraud==1",fontsize=14)
sns.countplot(y="P_emaildomain",ax=ax[2],data=train.loc[train['isFraud']==0])
ax[2].set_title("P_emaildomain isFraud==0",fontsize=14)
fig,ax=plt.subplots(1,3,figsize=(26,10))
sns.countplot(y="R_emaildomain",ax=ax[0],data=train)
ax[0].set_title("R_emaildomain",fontsize=14)
sns.countplot(y="R_emaildomain",ax=ax[1],data=train.loc[train['isFraud']==1])
ax[1].set_title("R_emaildomain isFraud==1",fontsize=14)
sns.countplot(y="R_emaildomain",ax=ax[2],data=train.loc[train['isFraud']==0])
ax[2].set_title("R_emaildomain isFraud==0",fontsize=14)
```




    Text(0.5, 1.0, 'R_emaildomain isFraud==0')




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_15_1.png)



![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_15_2.png)


### 3.5 C/D/M/V columns

These features are numerous in the dataset, up to hundreds. However, we can not infer their actual meaning by the column names only. So we process a basic visual chart to see what's going on in these columns.

```python
# C columns

c_cols = [c for c in train if c[0] == 'C']
sampled_train = pd.concat([train.loc[train['isFraud'] == 0].sample(1000),
          train.loc[train['isFraud'] == 1].sample(1000)])
sns.pairplot(sampled_train, hue='isFraud',vars=c_cols)
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_16_0.png)



```python
# D columns

d_cols = [c for c in train if c[0] == 'D']
sampled_train = pd.concat([train.loc[train['isFraud'] == 0].sample(1000),
          train.loc[train['isFraud'] == 1].sample(1000)])
sns.pairplot(sampled_train, hue='isFraud',vars=c_cols)
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_17_0.png)



```python
fig,ax=plt.subplots(1,3,figsize=(18,8))
sns.countplot(x="M1", ax=ax[0], hue = "isFraud", data=train)
ax[0].set_title('M1', fontsize=14)
sns.countplot(x="M2", ax=ax[1], hue = "isFraud", data=train)
ax[1].set_title('M2', fontsize=14)
sns.countplot(x="M3", ax=ax[2], hue = "isFraud", data=train)
ax[2].set_title('M3', fontsize=14)
fig,ax=plt.subplots(1,3,figsize=(18,8))
sns.countplot(x="M4", ax=ax[0], hue = "isFraud", data=train)
ax[0].set_title('M4', fontsize=14)
sns.countplot(x="M5", ax=ax[1], hue = "isFraud", data=train)
ax[1].set_title('M5', fontsize=14)
sns.countplot(x="M6", ax=ax[2], hue = "isFraud", data=train)
ax[2].set_title('M6', fontsize=14)
fig,ax=plt.subplots(1,3,figsize=(18,8))
sns.countplot(x="M7", ax=ax[0], hue = "isFraud", data=train)
ax[0].set_title('M7', fontsize=14)
sns.countplot(x="M8", ax=ax[1], hue = "isFraud", data=train)
ax[1].set_title('M8', fontsize=14)
sns.countplot(x="M9", ax=ax[2], hue = "isFraud", data=train)
ax[2].set_title('M9', fontsize=14)
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_18_0.png)



![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_18_1.png)



![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_18_2.png)



```python
v_cols = [c for c in train if c[0] == 'V']
train['v_mean']=train[v_cols].mean(axis=1)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
train.loc[train['isFraud'] == 1]['v_mean'].sample(10000) \
    .apply(np.log) \
    .plot(kind='hist',bins=339,title='log transformed mean of V columns - Fraud',ax=ax1)
train.loc[train['isFraud'] == 0]['v_mean'].sample(10000) \
    .apply(np.log) \
    .plot(kind='hist',bins=339,title='log transformed mean of V columns - Not Fraud',ax=ax2)
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_19_0.png)


### 3.6 Device type

```python
# device type

train.groupby('DeviceType') \
    .mean()['isFraud'] \
    .sort_values() \
    .plot(kind='barh',figsize=(15, 5),title='Percentage of Fraud by Device Type')
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_20_0.png)



```python
# device info

train.groupby('DeviceInfo') \
    .mean()['isFraud']\
    .sort_values()\
    .plot(kind='barh',figsize=(10,60),title='Percentage of Fraud by Device Info',fontsize=10)
plt.show()
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_21_0.png)

We can see that the precentage of fraud with different devices may vary a lot. This may contain important information.

Finally, let's look at the correlation of these variables visually, some features may just contain same information, so it will be inefficient for us to build the models with all the features.

```python
col=['isFraud','TransactionAmt','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
     'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15',
    'V126','V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137']
data_corr=train[col]
corrmat=data_corr.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,vmax=0.9,square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2f8d84190b8>




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/004%20IEEE%20Fraud%20Detection/img/1/output_22_1.png)

We can see that the features in the same types (For example, with same initial) have strong correlation, the features in different types may be not so relevant.
