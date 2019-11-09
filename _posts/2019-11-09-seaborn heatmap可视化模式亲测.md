---
layout:     post                    # 使用的布局（不需要改）
title:      seaborn heatmap可视化模式亲测             # 标题 
subtitle:   Something just for fun! #副标题
date:       2019-11-09             # 时间
author:     WZY                      # 作者
header-img: img/vis2.jpg   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - visualization
--- 

# seaborn heatmap可视化模式亲测 

利用heatmap绘制协方差矩阵是数据可视化中常见的操作，而对颜色的选取则是一种艺术了。在不同的场景下有可能我们需要不同的色调或者颜色的搭配。而seaborn中的heatmap函数为我们提供了便捷。

Seaborn中有非常多的颜色选项可以选择，这里将效果一一亲测。数据如下：这里我们用最为简单的数据绘制协方差矩阵的图。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
data=pd.DataFrame({'A':[1,4,5,2,5,6,3,5,6,3,3],
                 'B':[4,3,7,3,5,2,4,3,5,5,2],
                 'C':[5,8,9,3,5,7,3,5,3,4,4],
                 'D':[5,4,3,6,7,3,5,2,6,6,4],
                 'E':[4,5,7,3,6,3,2,4,5,2,1],
                 'F':[7,6,4,7,4,7,9,3,2,2,3],
                 'G':[4,5,2,5,8,9,1,2,4,4,3],
                 'H':[6,4,2,6,2,6,1,3,8,9,6],
                 'I':[4,3,5,2,7,8,3,4,2,5,3],
                 'J':[4,2,5,7,8,4,5,2,5,1,2],
                 'K':[4,5,8,2,3,1,4,5,8,3,2]})
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax)
ax.set_title('default')
```




    Text(0.5, 1, 'default')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_0_1.png)

可以看到，这就是heatmap的默认颜色模式，除了这种模式，我们可以指定cmap的值，使得绘制出来的图像呈现不同的颜色（在字符串后面加'_r'则进行反向）

## 1 Accent

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Accent')
ax.set_title('Accent')
```




    Text(0.5, 1, 'Accent')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_1_1.png)


## 2 Blues

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Blues')
ax.set_title('Blues')
```




    Text(0.5, 1, 'Blues')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_2_1.png)

## 3 BrBG

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='BrBG')
ax.set_title('BrBG')
```




    Text(0.5, 1, 'BrBG')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_3_1.png)

## 4 BuGn

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='BuGn')
ax.set_title('BuGn')
```




    Text(0.5, 1, 'BuGn')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_4_1.png)

## 5 BuPu

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='BuPu')
ax.set_title('BuPu')
```




    Text(0.5, 1, 'BuPu')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_5_1.png)

## 6 CMRmap

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='CMRmap')
ax.set_title('CMRmap')
```




    Text(0.5, 1, 'CMRmap')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_6_1.png)

## 7 Dark2

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Dark2')
ax.set_title('Dark2')
```




    Text(0.5, 1, 'Dark2')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_7_1.png)

## 8 GnBu

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='GnBu')
ax.set_title('GnBu')
```




    Text(0.5, 1, 'GnBu')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_8_1.png)

## 9 Greens

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Greens')
ax.set_title('Greens')
```




    Text(0.5, 1, 'Greens')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_9_1.png)

## 10 Greys

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Greys')
ax.set_title('Greys')
```




    Text(0.5, 1, 'Greys')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_10_1.png)

## 11 OrRd

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='OrRd')
ax.set_title('OrRd')
```




    Text(0.5, 1, 'OrRd')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_11_1.png)

## 12 Oranges

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Oranges')
ax.set_title('Oranges')
```




    Text(0.5, 1, 'Oranges')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_12_1.png)

## 13 PRGn

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='PRGn')
ax.set_title('PRGn')
```




    Text(0.5, 1, 'PRGn')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_13_1.png)

## 14 Paired

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Paired')
ax.set_title('Paired')
```




    Text(0.5, 1, 'Paired')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_14_1.png)

## 15 Pastel1

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Pastel1')
ax.set_title('Pastel1')
```




    Text(0.5, 1, 'Pastel1')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_15_1.png)

## 16 Pastel2

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Pastel2')
ax.set_title('Pastel2')
```




    Text(0.5, 1, 'Pastel2')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_16_1.png)

## 17 PiYG

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='PiYG')
ax.set_title('PiYG')
```




    Text(0.5, 1, 'PiYG')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_17_1.png)

## 18 PuBu

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='PuBu')
ax.set_title('PuBu')
```




    Text(0.5, 1, 'PuBu')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_18_1.png)

## 19 PuBuGn

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='PuBuGn')
ax.set_title('PuBuGn')
```




    Text(0.5, 1, 'PuBuGn')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_19_1.png)

## 20 PuOr

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='PuOr')
ax.set_title('PuOr')
```




    Text(0.5, 1, 'PuOr')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_20_1.png)

## 21 PuRd

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='PuRd')
ax.set_title('PuRd')
```




    Text(0.5, 1, 'PuRd')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_21_1.png)

## 22 Purples

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Purples')
ax.set_title('Purples')
```




    Text(0.5, 1, 'Purples')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_22_1.png)

## 23 RdBu

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='RdBu')
ax.set_title('RdBu')
```




    Text(0.5, 1, 'RdBu')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_23_1.png)

## 24 RdGy

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='RdGy')
ax.set_title('RdGy')
```




    Text(0.5, 1, 'RdGy')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_24_1.png)

## 25 RdPu

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='RdPu')
ax.set_title('RdPu')
```




    Text(0.5, 1, 'RdPu')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_25_1.png)

## 26 RdYlBu

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='RdYlBu')
ax.set_title('RdYlBu')
```




    Text(0.5, 1, 'RdYlBu')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_26_1.png)

## 27 RdYlGn

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='RdYlGn')
ax.set_title('RdYlGn')
```




    Text(0.5, 1, 'RdYlGn')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_27_1.png)

## 28 Reds

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Reds')
ax.set_title('Reds')
```




    Text(0.5, 1, 'Reds')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_28_1.png)

## 29 Set1

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Set1')
ax.set_title('Set1')
```




    Text(0.5, 1, 'Set1')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_29_1.png)

## 30 Set2

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Set2')
ax.set_title('Set2')
```




    Text(0.5, 1, 'Set2')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_30_1.png)

## 31 Set3

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Set3')
ax.set_title('Set3')
```




    Text(0.5, 1, 'Set3')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_31_1.png)

## 32 Spectral

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Spectral')
ax.set_title('Spectral')
```




    Text(0.5, 1, 'Spectral')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_32_1.png)

## 33 Wistia

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='Wistia')
ax.set_title('Wistia')
```




    Text(0.5, 1, 'Wistia')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_33_1.png)

## 34 YlGn

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='YlGn')
ax.set_title('YlGn')
```




    Text(0.5, 1, 'YlGn')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_34_1.png)

## 35 YlGnBu

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='YlGnBu')
ax.set_title('YlGnBu')
```




    Text(0.5, 1, 'YlGnBu')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_35_1.png)

## 36 YlOrBr

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='YlOrBr')
ax.set_title('YlOrBr')
```




    Text(0.5, 1, 'YlOrBr')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_36_1.png)

## 37 YlOrRd

```python
f, ax = plt.subplots(figsize=(13,11))
sns.heatmap(data.corr(), annot=True, ax=ax,cmap='YlOrRd')
ax.set_title('YlOrRd')
```




    Text(0.5, 1, 'YlOrRd')




![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/seaborn/heatmap/output_37_1.png)



```python

```
