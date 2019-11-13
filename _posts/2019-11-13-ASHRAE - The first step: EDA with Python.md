---
layout:     post                    # 使用的布局（不需要改）
title:      ASHRAE-The first step: EDA with Python               # 标题 
subtitle:   Kaggle Competition #副标题
date:       2019-11-13             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Kaggle
--- 

# ASHRAE - The first step: EDA with Python

In this competition we will develop models to predict the energy usage in each building. The dataset contains 1450+ buildings information. Different buildings may have different types of meters.

Before we start to build a model, we should have some intuition about the data first. So using visualization tools such as Seaborn we can be more familiar with the data we can get.

If you find this notebook interesting or useful, please do not hesitate to vote for me!

## 1 Start from Here

### 1.1 Loading Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
import gc
sns.set()
```


```python
PATH='/kaggle/ashrae-energy-prediction/'
train=pd.read_csv(PATH+'train.csv')
test=pd.read_csv(PATH+'test.csv')
weather_train=pd.read_csv(PATH+'weather_train.csv')
weather_test=pd.read_csv(PATH+'weather_test.csv')
building=pd.read_csv(PATH+'building_metadata.csv')
```


```python
# merge the data
train = train.merge(building, on='building_id', how='left')
test = test.merge(building, on='building_id', how='left')
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')
del weather_train, weather_test,building
gc.collect()
```




    27

### 1.2 Check some Information about the Data

Before we start, let's check the missing ratio of each column:

```python
# check whether there are missing values
data_na=(train.isnull().sum()/len(train))*100
data_na=data_na.drop(data_na[data_na==0].index).sort_values(ascending=False)
missing_data=pd.DataFrame({'MissingRatio':data_na})
print(missing_data)
```

                        MissingRatio
    floor_count            82.652772
    year_built             59.990033
    cloud_coverage         43.655131
    precip_depth_1_hr      18.544739
    wind_direction          7.167792
    sea_level_pressure      6.092515
    wind_speed              0.710701
    dew_temperature         0.495348
    air_temperature         0.478124
    

We can see that after merging the dataset some of the columns contain a large percentage of missing values, such as 'floor_count' and 'year_bulit'. We can further check the information of the dataset.

```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 20216100 entries, 0 to 20216099
    Data columns (total 16 columns):
    building_id           int16
    meter                 int8
    timestamp             datetime64[ns]
    meter_reading         float64
    site_id               int8
    primary_use           category
    square_feet           int32
    year_built            float16
    floor_count           float16
    air_temperature       float32
    cloud_coverage        float16
    dew_temperature       float32
    precip_depth_1_hr     float16
    sea_level_pressure    float32
    wind_direction        float16
    wind_speed            float32
    dtypes: category(1), datetime64[ns](1), float16(5), float32(4), float64(1), int16(1), int32(1), int8(2)
    memory usage: 1.1 GB
    
We can save the store memory by transforming some of the data types.

```python
# Saving the memory space
data_types = {'building_id': np.int16,
          'meter': np.int8,
          'site_id': np.int8,
          'square_feet': np.int32,
          'year_built': np.float16,
          'floor_count': np.float16,    
          'cloud_coverage': np.float16,
          'precip_depth_1_hr': np.float16,
           'wind_direction': np.float16,     
          'dew_temperature': np.float32,
          'air_temperature': np.float32,
          'sea_level_pressure': np.float32,
          'wind_speed': np.float32,
          'primary_use': 'category',}

for feature in data_types:
    train[feature] = train[feature].astype(data_types[feature])
    test[feature] = test[feature].astype(data_types[feature])
    
train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"] = pd.to_datetime(test["timestamp"])
gc.collect();
```

Now let's see the data sample.

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
      <th>building_id</th>
      <th>meter</th>
      <th>timestamp</th>
      <th>meter_reading</th>
      <th>site_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-01-01</td>
      <td>0.0</td>
      <td>0</td>
      <td>Education</td>
      <td>7432</td>
      <td>2008.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.700012</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2016-01-01</td>
      <td>0.0</td>
      <td>0</td>
      <td>Education</td>
      <td>2720</td>
      <td>2004.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.700012</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2016-01-01</td>
      <td>0.0</td>
      <td>0</td>
      <td>Education</td>
      <td>5376</td>
      <td>1991.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.700012</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2016-01-01</td>
      <td>0.0</td>
      <td>0</td>
      <td>Education</td>
      <td>23685</td>
      <td>2002.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.700012</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>2016-01-01</td>
      <td>0.0</td>
      <td>0</td>
      <td>Education</td>
      <td>116607</td>
      <td>1975.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.700012</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


## 2 Data Visualization

### 2.1 Boxplot

First let's see the relationship between the usage of the building and the age of the building:

```python
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='primary_use', y='year_built', data=train)
plt.xticks(rotation=90)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
     <a list of 16 Text xticklabel objects>)




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_7_1.png)

It seems that some kinds of buildings were built eariler, such as those used as technology building. What about the relationship between the building area and building types?

```python
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='primary_use', y='square_feet', data=train)
plt.xticks(rotation=90)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
     <a list of 16 Text xticklabel objects>)




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_8_1.png)


The above is an intuitive connection, as different types of buildings may install different types of meters, we will the this factor into consideration.

```python
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='primary_use', y='square_feet', hue='meter',data=train)
plt.xticks(rotation=90)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
     <a list of 16 Text xticklabel objects>)




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_9_1.png)

The result are clear. Some of the buildings only install two or three types of meters, and the corresponding building area may also have significant differences.

```python
sample=pd.DataFrame(train,columns=['site_id','primary_use'])
sample.drop_duplicates(keep='first')
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
      <th>site_id</th>
      <th>primary_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>Education</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0</td>
      <td>Lodging/residential</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0</td>
      <td>Office</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0</td>
      <td>Entertainment/public assembly</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0</td>
      <td>Other</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2098</td>
      <td>15</td>
      <td>Manufacturing/industrial</td>
    </tr>
    <tr>
      <td>2129</td>
      <td>15</td>
      <td>Religious worship</td>
    </tr>
    <tr>
      <td>2139</td>
      <td>15</td>
      <td>Lodging/residential</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>15</td>
      <td>Utility</td>
    </tr>
    <tr>
      <td>2203</td>
      <td>15</td>
      <td>Healthcare</td>
    </tr>
  </tbody>
</table>
<p>110 rows × 2 columns</p>
</div>

We can see that the buildings with a same 'site_id' may have different primary usage.

### 2.2 View the trends of the time series data

In this competition the data is shown in time series form, each row has a corresponding Time Label. So it is important for us to visualize the trends of the data.

First of all, we will observe the change of data from a monthly perspective. Specifically, the data will be separated by 'site_id' and the meter types.

```python
# We can see it by month
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by month', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_11_0.png)

From the visualization result, we can observe some rules of the data changing. For example, in most buildings the values of meter type 2 and 3 have a significant decrease during the summer, because type 2 provides steam and type 3 provides hot water. But the condition of site_id 13 have a little different.

We can also observe the data from a weekly perspective:

```python
# We can also see it by week
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('W').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by week', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_12_0.png)

From the weekly perspective,  some data show a cyclical pattern such as the condition of site_id 12.

And what about from a daily perspective?

```python
# By day
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by day', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_13_0.png)

We can also further separate the data by hours:

```python
# By hour
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by hour', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_14_0.png)


### 2.3 The Weather Condition

The weather condition may also have connections with the energy consumption, we can visualize it, too.

```python
# the weather condition(by day)
fig, axes = plt.subplots(figsize=(20,8))
axes1=axes.twinx()
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')
train[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes,alpha=0.9, label='air_temperature', color='red')
train[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes,alpha=0.9, label='dew_temperature', color='blue')
plt.legend()
```




    <matplotlib.legend.Legend at 0x233fe00b518>


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_15_1.png)

The green line represents 'meter_reading', we may not find any correlation between the temperature and the meter_reading, and it is strange that the value come across a sudden drop in Jun.

We can see the cloud condition.

```python
# the cloud condition(by day)
fig, axes = plt.subplots(figsize=(20,8))
axes1=axes.twinx()
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')
train[['timestamp', 'cloud_coverage']].set_index('timestamp').resample('D').mean()['cloud_coverage'].plot(ax=axes,alpha=0.9, label='cloud_coverage', color='cyan')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2340872ae10>




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_16_1.png)

We can see the wind condition, too.

```python
# the wind condition(by day)
fig, axes = plt.subplots(figsize=(20,8))
axes1=axes.twinx()
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')
train[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes,alpha=0.9, label='wind_speed', color='purple')
plt.legend()
```




    <matplotlib.legend.Legend at 0x233fe380b00>




![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_17_1.png)

In the visualization report above, we can see that the weather conditions are almost random, and there seems no correlation between weather conditions and the meter_reading if we just take such a cursory look.

### 2.4 Check the correlation matrix

Using the heatmap in seaborn can benefit us a lot, we can observe the corrlation of various features in this map.

```python
for i in range(0,4):
    corr = train[train.meter == i][['timestamp','meter_reading','square_feet','year_built','floor_count',
             'air_temperature','cloud_coverage','dew_temperature','sea_level_pressure','wind_direction','wind_speed']].corr()
    f, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(corr,annot=True,cmap='RdGy')
```


![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_18_0.png)



![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_18_1.png)



![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_18_2.png)



![png](https://github.com/Tinky2013/Kernel-Collection/raw/master/008%20ASHRAE%20-%20Great%20Energy%20Predictor%20III/img/output_18_3.png)


Now we have a basic understanding towards the data and the next step will be feature engineering based on the characteristics of the data and we will try a few types of models.
