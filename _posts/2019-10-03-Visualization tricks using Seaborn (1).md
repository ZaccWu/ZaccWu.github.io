---
layout:     post                    # 使用的布局（不需要改）
title:      Visualization tricks using Seaborn (1)             # 标题 
subtitle:    #副标题
date:       2019-10-03             # 时间
author:     WZY                      # 作者
header-img: img/vis2.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - visualization
--- 

# Visualization tricks using Seaborn (1)

In the process of making visual charts, we often need to deal with the relationship between numeric variables(N) and category variables(C). Somethings we need to deal with time series, too. In this article, I will introduce these visualization methods.

## 01 Catplot for 2C1N
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")


data={'class':(['1']*10+['2']*10+['3']*10)*2,
     'sex':['male']*30+['female']*30,
     'pay_rate':[0,1,1,0,1,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,
                0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,1,1,1,0,
                1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,0]}
data=pd.DataFrame(data)

g = sns.catplot(x="class", y="pay_rate", hue="sex", data=data,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("pay_rate")
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/01.png)

## 02 barplot for 1C3N
```python
# prepare the data
import pandas as pd
data={'type':['A','B','C','D','E','F','G','H','I','J'],
     'fare':[34,24,55,35,34,55,12,34,44,56],
     'score':[2.5,2.5,4.5,3,3.5,4.5,4,4,2.5,2]}
data=pd.DataFrame(data)

# plot
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk")
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13,11), sharex=True)

x = data['type']
y1 = data['fare']
sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("fare")

y2 = 80-y1
sns.barplot(x=x, y=y2, palette="vlag", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("relief")

y3 = data['score']
sns.barplot(x=x, y=y3, palette="deep", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("Qualitative")

sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=1)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/02.png)

## 03 facetgrid for 2C2N
```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="darkgrid")

data={'score':[7,6,8,7,5,4,5,7,8,9,5,6,11,10,5,
               4,5,6,8,4,9,4,10,7,5,7,4,8,9,6,
               11,5,6,8,8,8,11,9,5,6,7,7,8,8,10,
               6,7,4,7,8,9,5,6,8,6,7,7,11,11,11],
     'gender':['male']*30+['female']*30,
     'domain':(['yes']*15+['no']*15)*2}
data=pd.DataFrame(data)


g=sns.FacetGrid(data, row="gender", col="domain", margin_titles=True)
bins = np.linspace(4, 11, 8)
g.map(plt.hist, "score", color="steelblue", bins=bins)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/03.png)

## 04 barplot for 1C2N
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

data={'cont':['LTE','HUP','BYW','CIR','TRY','CUM','KRE','RAU','YHU','GNE',
             'ARC','SUB','DTE','AGN','USW','MIE','NSW','VYI','PAB','FDE',
             'RML','SAE','JCD','CFN','CNU','LWB','VQA','NRT','WEG','GVE'],
     'register':[129,118,116,109,102,101,98,94,94,91,90,88,85,83,82,
                81,79,78,75,74,71,70,68,67,64,60,59,55,53,50],
     'active':[65,55,49,51,45,69,45,44,43,34,49,51,32,33,36,41,33,30,28,22,
              24,32,21,28,39,31,20,22,21,19]}
data=pd.DataFrame(datq)
f, ax = plt.subplots(figsize=(9, 15))
sns.set_color_codes("pastel")
sns.barplot(x="register", y="cont", data=data,
            label="Register", color="b")

sns.set_color_codes("muted")
sns.barplot(x="active", y="cont", data=data,
            label="Active", color="b")

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 140), ylabel="",
       xlabel="Total number")
sns.despine(left=True, bottom=True)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/04.png)

## 05 boxplot for 2C1N
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="ticks", palette="pastel")

data={'class':['A']*20+['B']*20+['C']*20+['D']*20,
      'gender':(['male']*10+['female']*10)*4,
     'bill':[17,15,17,19,33,12,14,10,23,21,12,15,22,19,18,20,13,15,16,21,
            11,14,17,22,21,27,13,20,15,15,16,19,33,31,21,24,12,11,10,19,
            14,16,17,16,15,17,29,11,10,15,16,26,25,15,19,20,21,22,20,15,
            14,16,14,13,11,10,22,25,23,21,28,19,18,16,17,16,15,11,19,19]}
data=pd.DataFrame(data)

sns.boxplot(x="class", y="bill",
            hue="gender", palette=["m", "g"],
            data=data)
sns.despine(offset=10, trim=True)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/05.png)

## 06 boxplot for 1C1N
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")

data={'type':['A']*10+['B']*10+['C']*10+['D']*10+['E']*10+['F']*10+['G']*10+['H']*10+['I']*10+['J']*10,
     'value':[45,44,42,49,41,53,55,53,51,42,42,39,38,31,30,45,45,49,34,40,
             34,43,31,32,20,39,28,29,28,21,45,49,45,43,30,49,54,53,52,49,
             44,45,44,42,44,41,40,45,45,43,56,53,45,55,39,41,50,54,59,45,
             45,65,40,66,62,44,49,60,49,49,56,50,52,53,51,44,46,49,49,42,
             34,35,39,29,29,32,31,30,40,44,32,22,30,22,39,32,34,31,29,32]}
data=pd.DataFrame(data)
f, ax = plt.subplots(figsize=(11, 11))
sns.boxplot(x="value", y="type", data=data,
            whis="range", palette="vlag")
sns.swarmplot(x="value", y="type", data=data,
              size=2, color=".3", linewidth=0)
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/06.png)

## 07 violinplot for 2C1N
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)

data={'class':['A']*20+['B']*20+['C']*20+['D']*20,
      'gender':(['male']*10+['female']*10)*4,
     'bill':[17,15,17,19,33,12,14,10,23,21,12,15,22,19,18,20,13,15,16,21,
            11,14,17,22,21,27,13,20,15,15,16,19,33,31,21,24,12,11,10,19,
            14,16,17,16,15,17,29,11,10,15,16,26,25,15,19,20,21,22,20,15,
            14,16,14,13,11,10,22,25,23,21,28,19,18,16,17,16,15,11,19,19]}
data=pd.DataFrame(data)

sns.violinplot(x="class", y="bill", hue="gender",
               split=True, inner="quart",
               palette={"male": "y", "female": "b"},
               data=data)
sns.despine(left=True)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/07.png)

## 08 violinplot for 1C1N
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
# Create a random dataset across several variables
rs = np.random.RandomState(0)
n, p = 40, 8
d = rs.normal(0, 2, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 10

plt.subplots(figsize=(11,11))
pal = sns.cubehelix_palette(p, rot=-.5, dark=.3)
sns.violinplot(data=d, palette=pal, inner="points")
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/08.png)

## 09 boxenplot for 1C1N
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

data={'type':['AD']*20+['BN']*20+['CG']*20+['EL']*20+['JR']*20+['LK']*20+['PU']*20+['SC']*20,
     'value':[455,314,356,334,461,420,441,406,405,394,377,355,365,389,390,355,401,411,432,473,
           378,388,395,412,406,365,330,423,421,388,354,313,456,340,355,400,396,320,356,366,
           349,388,394,383,406,442,432,356,398,391,422,354,322,368,390,385,422,409,355,394,
           445,405,455,354,387,357,390,355,367,322,318,405,406,444,402,324,477,351,388,398,
           399,384,377,393,332,441,422,398,355,358,473,492,333,394,384,355,357,344,398,394,
           356,399,385,357,347,390,405,406,411,345,353,359,385,399,390,400,422,434,392,444,
           441,361,398,389,388,370,405,443,349,399,404,412,425,356,367,399,309,445,409,422,
           389,399,358,399,387,366,404,412,435,465,354,399,387,369,405,441,439,438,452,323]}

data=pd.DataFrame(data)
plt.subplots(figsize=(11,11))
sns.boxenplot(x="type", y="value",
              color="b", scale="linear", data=data)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/09.png)

## 10 heatmap for 1C2N
```python
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/10.png)

## 11 heatmap for 2C1N
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
data={'year':['2008']*12+['2009']*12+['2010']*12+['2011']*12+['2012']*12+['2013']*12
            +['2014']*12+['2015']*12+['2016']*12+['2017']*12+['2018']*12+['2019']*12,
     'month':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']*12,
     'value':[457,556,312,122,129,341,591,507,401,220,105,142,
             455,509,334,118,127,356,600,524,388,209,144,106,
             468,559,306,134,150,388,626,587,412,233,145,146,
             460,550,344,143,147,368,630,569,440,241,145,127,
             478,526,345,168,177,400,696,624,445,256,137,160,
             510,621,318,201,189,401,722,686,495,225,213,190,
             501,598,444,260,173,399,692,665,455,289,201,198,
             525,625,413,275,200,442,737,701,459,306,244,203,
             560,677,435,298,264,440,781,746,494,368,289,221,
             562,647,424,305,300,460,777,745,498,369,290,221,
             592,658,405,309,298,457,813,820,477,391,269,218,
             591,679,407,334,294,460,828,845,501,355,298,246,]}
data=pd.DataFrame(data)
data = data.pivot("year", "month", "value")  # group the data

f, ax = plt.subplots(figsize=(11,11))
sns.heatmap(data, annot=True, fmt="d", linewidths=.5, ax=ax)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/11.png)

## 12 jointplot for distributions A
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

sns.jointplot(x, y, kind="hex", color="#4CB391")
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/12.png)

## 13 jointplot for distributions B
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

# Generate a random correlated bivariate dataset
rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(1, .5), (.5, 1)]
x1, x2 = rs.multivariate_normal(mean, cov, 500).T
x1 = pd.Series(x1, name="$X_1$")
x2 = pd.Series(x2, name="$X_2$")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(x1, x2, kind="kde", height=7, space=0)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/13.png)
