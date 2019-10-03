---
layout:     post                    # 使用的布局（不需要改）
title:      Visualization tricks using Seaborn (2)             # 标题 
subtitle:    #副标题
date:       2019-10-03             # 时间
author:     WZY                      # 作者
header-img: img/vis2.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - visualization
--- 

# Visualization tricks using Seaborn (2) 

In the process of making visual charts, we often need to deal with the relationship between numeric variables(N) and category variables(C). Somethings we need to deal with time series, too. In this article, I will introduce these visualization methods.

## 14 jointplot for 2N-regression
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

data={'income':[123,144,168,171,180,185,187,189,190,193,194,195,199,202,203,204,210,222,224,228,
               229,230,231,231,234,238,239,242,244,245,250,253,254,258,260,278,292,313,346,357],
     'spent':[88,110,102,119,144,92,138,131,130,166,156,159,185,113,144,159,159,166,178,179,
             181,168,177,190,188,185,177,171,189,202,198,192,201,194,204,219,244,241,323,286]}
data=pd.DataFrame(data)
g = sns.jointplot("income", "spent", data=data, kind="reg",
                  xlim=(100, 400), ylim=(80, 350), color="m", height=7)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/14.png)

## 15 kdeplot for 1C1N
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the data
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "x")
# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)
# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/15.png)

## 16 stripplot for 2C1N
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

data={'location':['north']*30+['south']*30+['east']*30+['west']*30,
     'type':(['A']*10+['B']*10+['C']*10)*4,
     'value':[44,5,3,5,4,6,7,3,5,4,6,8,5,5,5,3,4,7,4,5,4,7,5,13,5,4,6,8,8,5,5,4,3,5,2,7,6,5,4,5,
             4,6,7,15,3,6,3,6,6,4,5,7,5,6,8,8,6,4,5,7,3,6,5,5,5,3,6,3,5,5,35,4,2,2,6,5,6,7,8,6,
             5,6,3,6,3,6,8,9,7,7,6,4,15,43,6,4,6,3,5,7,3,7,6,6,5,7,5,5,7,7,65,7,4,5,4,6,7,8,7,5]}
data=pd.DataFrame(data)
# Initialize the figure
f, ax = plt.subplots(figsize=(11,11))
sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot
sns.stripplot(x="value", y="location", hue="type",
              data=data, dodge=True, jitter=True,
              alpha=.25, zorder=1)
# Show the conditional means
sns.pointplot(x="value", y="location", hue="type",
              data=data, dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)
# Improve the legend 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[3:], labels[3:], title="type",
          handletextpad=0, columnspacing=1,
          loc="lower right", ncol=3, frameon=True)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/16.png)

## 17 replot for 1C2N time series
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

# Load the example mpg dataset
mpg = sns.load_dataset("mpg")

data={'city':['Ubn']*10+['Rsk']*10+['Pcd']*10+['Grt']*10+['Rbn']*10+['Yml']*10+['Gcr']*10+['Ati']*10,
     'year':[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,
            1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,
            1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,
            1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10],
     'ave_eco':[23,24,26,33,36,45,58,70,79,90,22,26,36,39,47,58,68,79,90,101,
               18,20,22,27,31,44,51,56,63,74,11,12,14,16,25,34,37,42,47,51,
               14,18,22,23,29,35,45,51,56,63,32,37,46,57,70,79,86,100,112,123,
               10,15,19,22,24,28,34,40,46,58,12,15,19,24,34,44,53,59,60,66],
     'population':[36,37,38,39,42,44,45,46,47,49,51,52,52,54,55,57,58,59,61,62,
                  98,92,94,95,108,119,121,122,133,135,144,149,155,169,173,176,182,191,194,199,
                  23,25,27,29,31,33,35,37,44,48,51,52,54,56,57,57,58,58,59,61,
                  22,21,22,24,30,36,37,38,39,44,46,49,51,54,56,58,62,69,77,86]}
data=pd.DataFrame(data)
# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="year", y="ave_eco", hue="city", size="population",
            sizes=(10, 500), alpha=.5, palette="muted",
            height=6, data=data)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/17.png)

## 18 lineplot for 2C1N time series
```python
# prepare the data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

data={'time':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
             1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
              1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
              1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
     'type':['strong']*30+['weak']*30,
     'shape':(['dark']*15+['light']*15)*2,
     'value':[5,6,3,5,5,3,4,4,5,6,7,8,5,9,11,
             4,5,3,5,6,3,2,4,6,6,7,8,10,7,5,
             3,4,5,2,4,5,3,4,3,5,6,7,8,8,9,
             2,1,4,3,4,4,5,5,6,3,3,5,5,7,7]}
data=pd.DataFrame(data)

# Plot
plt.subplots(figsize=(11,11))
sns.lineplot(x="time", y="value",
             hue="type", style="shape",
             data=data)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/18.png)

## 19 relplot for 2C1N time series
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="ticks")

data={'type':['North']*100+['South']*100,
      'category':(['A']*20+['B']*20+['C']*20+['D']*20+['E']*20)*2,
      'time':['01','02','03','04','05','06','07','08','09','10',
              '11','12','13','14','15','16','17','18','19','20']*10,
      'rate':[40,41,46,42,47,46,57,63,55,64,67,62,57,45,37,22,29,19,27,28,
             40,41,44,47,49,55,52,56,57,56,67,66,53,41,22,35,36,30,31,33,
             40,41,42,49,43,45,55,58,67,66,69,56,55,51,35,39,24,21,28,37,
             40,41,42,46,44,52,56,67,68,69,65,63,66,40,20,25,28,44,34,35,
             40,41,41,46,44,42,51,58,67,58,60,64,62,45,34,33,37,23,26,29,
             40,42,41,45,47,47,55,57,57,65,64,67,64,55,24,29,24,39,33,34,
             40,42,47,52,45,56,60,64,69,65,68,64,69,55,41,35,37,33,32,35,
             40,41,43,44,47,57,66,61,64,67,69,61,54,51,30,33,22,29,28,25,
             40,41,44,48,49,51,52,59,53,68,62,64,56,44,29,33,32,30,38,31,
             40,41,49,48,50,41,52,54,68,62,66,64,60,44,34,39,32,30,33,38,]
}
data=pd.DataFrame(data)

# Define a palette to ensure that colors will be
palette = dict(zip(data.category.unique(),
                   sns.color_palette("rocket_r", 6)))

# Plot the lines on two facets
sns.relplot(x="time", y="rate",
            hue="category", col="type",
            palette=palette,
            height=8, aspect=.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=data)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/19.png)

## 20 scatterplot for 1C3N
```python
# prepare the data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(style="whitegrid")
data={'size':[1,1.2,1.3,1,1.1,1,1.5,1,1.7,1.7,1.9,1.9,2,2,2.2,2.3,2.5,2.5,2.6,2.7,
             2.9,2.9,3,3.1,3.3,3.4,3.5,3.7,3.7,4.1,4.2,4.4,4.4,4.7,5.7,5.9,5.9,6,6.6,6.9],
     'price':[13,15,16,14,16,17,18,18,17,19,18,29,28,27,34,35,36,37,36,33,
             23,24,45,46,41,42,46,43,47,50,51,53,56,57,67,67,63,76,87,89],
     'huge':[8,8,9,8,9,10,11,11,9,10,11,14,15,7,11,9,8,10,9,9,
            10,11,8,8,7,9,10,10,12,8,9,10,13,12,9,8,10,10,11,11],
     'quality':['D','B','B','C','C','B','A','A','E','C','D','A','B','D','D','C','E','A','F','G',
               'G','D','B','C','C','D','B','E','B','F','E','D','C','C','B','B','E','A','C','A']}
data=pd.DataFrame(data)

# plot
f, ax = plt.subplots(figsize=(11, 11))
sns.despine(f, left=True, bottom=True)
clarity_ranking = ['A','B','C','D','E','F','G']
sns.scatterplot(x="size", y="price",
                hue="quality", size="huge",
                palette="ch:r=-.2,d=.3_r",
                hue_order=clarity_ranking,
                sizes=(50, 400), linewidth=0,
                data=data, ax=ax)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/20.png)

## 21 distplot for distributions
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.set(style="white",palette="muted",color_codes=True)
rs=np.random.RandomState(10)
data=rs.normal(size=1000)

# plot
f,axes=plt.subplots(2,2,figsize=(11,11),sharex=True)
sns.despine(left=True)
sns.distplot(data,kde=False,color='purple',ax=axes[0,0])
sns.distplot(data,hist=False,color='red',ax=axes[0, 1])
sns.distplot(data,hist=False, color="blue", kde_kws={"shade": True}, ax=axes[1, 0])
sns.distplot(data,rug=True,color="gold", ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()

'''
rug: density below the graph (default: False)
kde: density line (default:True)
hist: histgram (default:True)
bins: the number of histgram bar
'''
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/21.png)

## 22 lmplot for 1C2N-regression
```python
# prepare the data
import pandas as pd
data={'dataset':['A']*10+['B']*10+['C']*10+['D']*10,
     'x':[1,2,3,4,6,7,5,7,4,7,
         1,3,5,6,2,4,5,3,4,5,
         3,4,5,1,3,6,7,9,7,9,
         8,5,3,6,5,4,3,2,1,1],
     'y':[1,2,3,4,5,6,7,8,9,10,
         1,2,3,4,5,6,7,8,9,10,
         1,2,3,4,5,6,7,8,9,10,
         1,2,3,4,5,6,7,8,9,10]}
data=pd.DataFrame(data)

# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=data,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})
'''
col_wrap=2 means two subplots a line
"s" means scatter size
'''
plt.show()
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/22.png)

## 23 implot for 1C2N-regression
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

# Load the iris dataset
iris = sns.load_dataset("iris")
data={'species':['A']*10+['B']*10+['C']*10+['D']*10,
     'x':[13,14,16,18,19,22,23,25,26,27,9,11,11,14,14,15,16,17,19,20,
         13,15,16,16,16,18,18,26,29,35,20,21,22,23,28,29,29,30,30,31],
     'y':[15,16,19,13,10,22,29,24,19,25,22,26,29,22,21,34,39,44,30,45,
         15,16,15,14,16,11,11,10,22,20,22,25,21,19,22,28,28,24,25,30]}
data=pd.DataFrame(data)

g = sns.lmplot(x="x", y="y", hue="species",
               truncate=True, height=5, data=data)
g.set_axis_labels("x", "y")
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/23.png)

## 24 facetgrid for 1C2N
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks")

# Create a dataset with many short random walks
data={'step':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,
             1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,
             1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,
             1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,],
     'position':[3,4,5,6,7,3,5,4,2,6,4,3,3,2,6,6,4,3,5,6,4,3,2,5,6,
                3,7,6,6,4,4,5,6,3,5,5,6,7,4,5,6,5,4,3,4,4,5,6,3,4,
                4,5,6,3,5,5,6,7,5,4,2,4,5,3,5,6,4,5,3,6,5,4,4,5,6,
                3,5,4,5,3,5,6,4,5,6,3,5,5,4,6,6,5,4,6,5,5,6,3,6,3],
     'category':['01']*5+['02']*5+['03']*5+['04']*5+['05']*5+
             ['06']*5+['07']*5+['08']*5+['09']*5+['10']*5+
             ['11']*5+['12']*5+['13']*5+['14']*5+['15']*5+
             ['16']*5+['17']*5+['18']*5+['19']*5+['20']*5}
data=pd.DataFrame(data)

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(data, col="category", hue="step", palette="tab20c",
                     col_wrap=4, height=1.5)
# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "step", "position", marker="o")
grid.set(xlim=(-.5, 5.5), ylim=(1, 7))
grid.fig.tight_layout(w_pad=1)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/24.png)

## 25 kdeplot for distributions
```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="dark")
rs = np.random.RandomState(50)
f, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)
# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 16)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=10, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))
f.tight_layout()
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/25.png)

## 26 combined plot A
```python
import seaborn as sns
sns.set(style="white")

df = sns.load_dataset("iris")

g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_diag(sns.kdeplot, lw=3)
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/26.png)

## 27 combined plot B
```python
import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")
```

![png](https://github.com/Tinky2013/My-Code-Hub/raw/master/visualization/img/27.png)
