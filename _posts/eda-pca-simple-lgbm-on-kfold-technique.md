
<h1><center><font size="6">Santander EDA, PCA and Light GBM Classification Model</font></center></h1>

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg/640px-Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg"></img>

<br>
<b>
In this challenge, Santander invites Kagglers to help them identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data they have available to solve this problem. 
The data is anonimyzed, each row containing 200 numerical values identified just with a number.</b>

<b>Inspired by Jiwei Liu's Kernel. I added Data Augmentation Segment to my kernel</b>

<pre>
<a id='0'><b>Content</b></a>
- <a href='#1'><b>Import the Data</b></a>
- <a href='#11'><b>Data Exploration</b></a>  
- <a href='#2'><b>Check for the missing values</b></a>  
- <a href='#3'><b>Visualizing the Satendar Customer Transactions Data</b></a>   
 - <a href='#31'><b>Check for Class Imbalance</b></a>   
 - <a href='#32'><b>Distribution of Mean and Standard Deviation</b></a>   
 - <a href='#33'><b>Distribution of Skewness</b></a>   
 - <a href='#34'><b>Distribution of Kurtosis</b></a>   
- <a href='#4'><b>Principal Component Analysis</b></a>
 - <a href='#41'><b>Kernel PCA</b></a>
- <a href = "#16"><b>Data Augmentation</b></a>
- <a href='#6'><b>Build the Light GBM Model</b></a></pre>


```python
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,KFold
import warnings
from six.moves import urllib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('seaborn')
from scipy.stats import norm, skew
```

<a id=1><pre><b>Import the Data</b></pre></a>


```python
#Load the Data
train=pd.read_csv("C:/Users/Tinky/Desktop/santander-customer-transaction-prediction/train.csv")
test=pd.read_csv("C:/Users/Tinky/Desktop/santander-customer-transaction-prediction/test.csv")
features = [c for c in train.columns if c not in ['ID_code', 'target']]
```

<a id=11><pre><b>Data Exploration</b></pre></a>


```python
train.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>var_8</th>
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
      <th>count</th>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>...</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.100490</td>
      <td>10.679914</td>
      <td>-1.627622</td>
      <td>10.715192</td>
      <td>6.796529</td>
      <td>11.078333</td>
      <td>-5.065317</td>
      <td>5.408949</td>
      <td>16.545850</td>
      <td>0.284162</td>
      <td>...</td>
      <td>3.234440</td>
      <td>7.438408</td>
      <td>1.927839</td>
      <td>3.331774</td>
      <td>17.993784</td>
      <td>-0.142088</td>
      <td>2.303335</td>
      <td>8.908158</td>
      <td>15.870720</td>
      <td>-3.326537</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.300653</td>
      <td>3.040051</td>
      <td>4.050044</td>
      <td>2.640894</td>
      <td>2.043319</td>
      <td>1.623150</td>
      <td>7.863267</td>
      <td>0.866607</td>
      <td>3.418076</td>
      <td>3.332634</td>
      <td>...</td>
      <td>4.559922</td>
      <td>3.023272</td>
      <td>1.478423</td>
      <td>3.992030</td>
      <td>3.135162</td>
      <td>1.429372</td>
      <td>5.454369</td>
      <td>0.921625</td>
      <td>3.010945</td>
      <td>10.438015</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.408400</td>
      <td>-15.043400</td>
      <td>2.117100</td>
      <td>-0.040200</td>
      <td>5.074800</td>
      <td>-32.562600</td>
      <td>2.347300</td>
      <td>5.349700</td>
      <td>-10.505500</td>
      <td>...</td>
      <td>-14.093300</td>
      <td>-2.691700</td>
      <td>-3.814500</td>
      <td>-11.783400</td>
      <td>8.694400</td>
      <td>-5.261000</td>
      <td>-14.209600</td>
      <td>5.960600</td>
      <td>6.299300</td>
      <td>-38.852800</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>8.453850</td>
      <td>-4.740025</td>
      <td>8.722475</td>
      <td>5.254075</td>
      <td>9.883175</td>
      <td>-11.200350</td>
      <td>4.767700</td>
      <td>13.943800</td>
      <td>-2.317800</td>
      <td>...</td>
      <td>-0.058825</td>
      <td>5.157400</td>
      <td>0.889775</td>
      <td>0.584600</td>
      <td>15.629800</td>
      <td>-1.170700</td>
      <td>-1.946925</td>
      <td>8.252800</td>
      <td>13.829700</td>
      <td>-11.208475</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>10.524750</td>
      <td>-1.608050</td>
      <td>10.580000</td>
      <td>6.825000</td>
      <td>11.108250</td>
      <td>-4.833150</td>
      <td>5.385100</td>
      <td>16.456800</td>
      <td>0.393700</td>
      <td>...</td>
      <td>3.203600</td>
      <td>7.347750</td>
      <td>1.901300</td>
      <td>3.396350</td>
      <td>17.957950</td>
      <td>-0.172700</td>
      <td>2.408900</td>
      <td>8.888200</td>
      <td>15.934050</td>
      <td>-2.819550</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>12.758200</td>
      <td>1.358625</td>
      <td>12.516700</td>
      <td>8.324100</td>
      <td>12.261125</td>
      <td>0.924800</td>
      <td>6.003000</td>
      <td>19.102900</td>
      <td>2.937900</td>
      <td>...</td>
      <td>6.406200</td>
      <td>9.512525</td>
      <td>2.949500</td>
      <td>6.205800</td>
      <td>20.396525</td>
      <td>0.829600</td>
      <td>6.556725</td>
      <td>9.593300</td>
      <td>18.064725</td>
      <td>4.836800</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>20.315000</td>
      <td>10.376800</td>
      <td>19.353000</td>
      <td>13.188300</td>
      <td>16.671400</td>
      <td>17.251600</td>
      <td>8.447700</td>
      <td>27.691800</td>
      <td>10.151300</td>
      <td>...</td>
      <td>18.440900</td>
      <td>16.716500</td>
      <td>8.402400</td>
      <td>18.281800</td>
      <td>27.928800</td>
      <td>4.272900</td>
      <td>18.321500</td>
      <td>12.000400</td>
      <td>26.079100</td>
      <td>28.500700</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 201 columns</p>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200000 entries, 0 to 199999
    Columns: 202 entries, ID_code to var_199
    dtypes: float64(200), int64(1), object(1)
    memory usage: 308.2+ MB
    


```python
train.shape
```




    (200000, 202)




```python
train.head(5)
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
      <th>var_8</th>
      <th>var_9</th>
      <th>var_10</th>
      <th>var_11</th>
      <th>var_12</th>
      <th>var_13</th>
      <th>var_14</th>
      <th>var_15</th>
      <th>var_16</th>
      <th>var_17</th>
      <th>var_18</th>
      <th>var_19</th>
      <th>var_20</th>
      <th>var_21</th>
      <th>var_22</th>
      <th>var_23</th>
      <th>var_24</th>
      <th>var_25</th>
      <th>var_26</th>
      <th>var_27</th>
      <th>var_28</th>
      <th>var_29</th>
      <th>var_30</th>
      <th>var_31</th>
      <th>var_32</th>
      <th>var_33</th>
      <th>var_34</th>
      <th>var_35</th>
      <th>var_36</th>
      <th>var_37</th>
      <th>...</th>
      <th>var_160</th>
      <th>var_161</th>
      <th>var_162</th>
      <th>var_163</th>
      <th>var_164</th>
      <th>var_165</th>
      <th>var_166</th>
      <th>var_167</th>
      <th>var_168</th>
      <th>var_169</th>
      <th>var_170</th>
      <th>var_171</th>
      <th>var_172</th>
      <th>var_173</th>
      <th>var_174</th>
      <th>var_175</th>
      <th>var_176</th>
      <th>var_177</th>
      <th>var_178</th>
      <th>var_179</th>
      <th>var_180</th>
      <th>var_181</th>
      <th>var_182</th>
      <th>var_183</th>
      <th>var_184</th>
      <th>var_185</th>
      <th>var_186</th>
      <th>var_187</th>
      <th>var_188</th>
      <th>var_189</th>
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
      <th>0</th>
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
      <td>-4.9200</td>
      <td>5.7470</td>
      <td>2.9252</td>
      <td>3.1821</td>
      <td>14.0137</td>
      <td>0.5745</td>
      <td>8.7989</td>
      <td>14.5691</td>
      <td>5.7487</td>
      <td>-7.2393</td>
      <td>4.2840</td>
      <td>30.7133</td>
      <td>10.5350</td>
      <td>16.2191</td>
      <td>2.5791</td>
      <td>2.4716</td>
      <td>14.3831</td>
      <td>13.4325</td>
      <td>-5.1488</td>
      <td>-0.4073</td>
      <td>4.9306</td>
      <td>5.9965</td>
      <td>-0.3085</td>
      <td>12.9041</td>
      <td>-3.8766</td>
      <td>16.8911</td>
      <td>11.1920</td>
      <td>10.5785</td>
      <td>0.6764</td>
      <td>7.8871</td>
      <td>...</td>
      <td>15.4576</td>
      <td>5.3133</td>
      <td>3.6159</td>
      <td>5.0384</td>
      <td>6.6760</td>
      <td>12.6644</td>
      <td>2.7004</td>
      <td>-0.6975</td>
      <td>9.5981</td>
      <td>5.4879</td>
      <td>-4.7645</td>
      <td>-8.4254</td>
      <td>20.8773</td>
      <td>3.1531</td>
      <td>18.5618</td>
      <td>7.7423</td>
      <td>-10.1245</td>
      <td>13.7241</td>
      <td>-3.5189</td>
      <td>1.7202</td>
      <td>-8.4051</td>
      <td>9.0164</td>
      <td>3.0657</td>
      <td>14.3691</td>
      <td>25.8398</td>
      <td>5.8764</td>
      <td>11.8411</td>
      <td>-19.7159</td>
      <td>17.5743</td>
      <td>0.5857</td>
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
      <th>1</th>
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
      <td>3.1468</td>
      <td>8.0851</td>
      <td>-0.4032</td>
      <td>8.0585</td>
      <td>14.0239</td>
      <td>8.4135</td>
      <td>5.4345</td>
      <td>13.7003</td>
      <td>13.8275</td>
      <td>-15.5849</td>
      <td>7.8000</td>
      <td>28.5708</td>
      <td>3.4287</td>
      <td>2.7407</td>
      <td>8.5524</td>
      <td>3.3716</td>
      <td>6.9779</td>
      <td>13.8910</td>
      <td>-11.7684</td>
      <td>-2.5586</td>
      <td>5.0464</td>
      <td>0.5481</td>
      <td>-9.2987</td>
      <td>7.8755</td>
      <td>1.2859</td>
      <td>19.3710</td>
      <td>11.3702</td>
      <td>0.7399</td>
      <td>2.7995</td>
      <td>5.8434</td>
      <td>...</td>
      <td>29.4846</td>
      <td>5.8683</td>
      <td>3.8208</td>
      <td>15.8348</td>
      <td>-5.0121</td>
      <td>15.1345</td>
      <td>3.2003</td>
      <td>9.3192</td>
      <td>3.8821</td>
      <td>5.7999</td>
      <td>5.5378</td>
      <td>5.0988</td>
      <td>22.0330</td>
      <td>5.5134</td>
      <td>30.2645</td>
      <td>10.4968</td>
      <td>-7.2352</td>
      <td>16.5721</td>
      <td>-7.3477</td>
      <td>11.0752</td>
      <td>-5.5937</td>
      <td>9.4878</td>
      <td>-14.9100</td>
      <td>9.4245</td>
      <td>22.5441</td>
      <td>-4.8622</td>
      <td>7.6543</td>
      <td>-15.9319</td>
      <td>13.3175</td>
      <td>-0.3566</td>
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
      <th>2</th>
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
      <td>-4.9193</td>
      <td>5.9525</td>
      <td>-0.3249</td>
      <td>-11.2648</td>
      <td>14.1929</td>
      <td>7.3124</td>
      <td>7.5244</td>
      <td>14.6472</td>
      <td>7.6782</td>
      <td>-1.7395</td>
      <td>4.7011</td>
      <td>20.4775</td>
      <td>17.7559</td>
      <td>18.1377</td>
      <td>1.2145</td>
      <td>3.5137</td>
      <td>5.6777</td>
      <td>13.2177</td>
      <td>-7.9940</td>
      <td>-2.9029</td>
      <td>5.8463</td>
      <td>6.1439</td>
      <td>-11.1025</td>
      <td>12.4858</td>
      <td>-2.2871</td>
      <td>19.0422</td>
      <td>11.0449</td>
      <td>4.1087</td>
      <td>4.6974</td>
      <td>6.9346</td>
      <td>...</td>
      <td>13.2070</td>
      <td>5.8442</td>
      <td>4.7086</td>
      <td>5.7141</td>
      <td>-1.0410</td>
      <td>20.5092</td>
      <td>3.2790</td>
      <td>-5.5952</td>
      <td>7.3176</td>
      <td>5.7690</td>
      <td>-7.0927</td>
      <td>-3.9116</td>
      <td>7.2569</td>
      <td>-5.8234</td>
      <td>25.6820</td>
      <td>10.9202</td>
      <td>-0.3104</td>
      <td>8.8438</td>
      <td>-9.7009</td>
      <td>2.4013</td>
      <td>-4.2935</td>
      <td>9.3908</td>
      <td>-13.2648</td>
      <td>3.1545</td>
      <td>23.0866</td>
      <td>-5.3000</td>
      <td>5.3745</td>
      <td>-6.2660</td>
      <td>10.1934</td>
      <td>-0.8417</td>
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
      <th>3</th>
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
      <td>-5.8609</td>
      <td>8.2450</td>
      <td>2.3061</td>
      <td>2.8102</td>
      <td>13.8463</td>
      <td>11.9704</td>
      <td>6.4569</td>
      <td>14.8372</td>
      <td>10.7430</td>
      <td>-0.4299</td>
      <td>15.9426</td>
      <td>13.7257</td>
      <td>20.3010</td>
      <td>12.5579</td>
      <td>6.8202</td>
      <td>2.7229</td>
      <td>12.1354</td>
      <td>13.7367</td>
      <td>0.8135</td>
      <td>-0.9059</td>
      <td>5.9070</td>
      <td>2.8407</td>
      <td>-15.2398</td>
      <td>10.4407</td>
      <td>-2.5731</td>
      <td>6.1796</td>
      <td>10.6093</td>
      <td>-5.9158</td>
      <td>8.1723</td>
      <td>2.8521</td>
      <td>...</td>
      <td>31.8833</td>
      <td>5.9684</td>
      <td>7.2084</td>
      <td>3.8899</td>
      <td>-11.0882</td>
      <td>17.2502</td>
      <td>2.5881</td>
      <td>-2.7018</td>
      <td>0.5641</td>
      <td>5.3430</td>
      <td>-7.1541</td>
      <td>-6.1920</td>
      <td>18.2366</td>
      <td>11.7134</td>
      <td>14.7483</td>
      <td>8.1013</td>
      <td>11.8771</td>
      <td>13.9552</td>
      <td>-10.4701</td>
      <td>5.6961</td>
      <td>-3.7546</td>
      <td>8.4117</td>
      <td>1.8986</td>
      <td>7.2601</td>
      <td>-0.4639</td>
      <td>-0.0498</td>
      <td>7.9336</td>
      <td>-12.8279</td>
      <td>12.4124</td>
      <td>1.8489</td>
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
      <th>4</th>
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
      <td>6.2654</td>
      <td>7.6784</td>
      <td>-9.4458</td>
      <td>-12.1419</td>
      <td>13.8481</td>
      <td>7.8895</td>
      <td>7.7894</td>
      <td>15.0553</td>
      <td>8.4871</td>
      <td>-3.0680</td>
      <td>6.5263</td>
      <td>11.3152</td>
      <td>21.4246</td>
      <td>18.9608</td>
      <td>10.1102</td>
      <td>2.7142</td>
      <td>14.2080</td>
      <td>13.5433</td>
      <td>3.1736</td>
      <td>-3.3423</td>
      <td>5.9015</td>
      <td>7.9352</td>
      <td>-3.1582</td>
      <td>9.4668</td>
      <td>-0.0083</td>
      <td>19.3239</td>
      <td>12.4057</td>
      <td>0.6329</td>
      <td>2.7922</td>
      <td>5.8184</td>
      <td>...</td>
      <td>33.5107</td>
      <td>5.6953</td>
      <td>5.4663</td>
      <td>18.2201</td>
      <td>6.5769</td>
      <td>21.2607</td>
      <td>3.2304</td>
      <td>-1.7759</td>
      <td>3.1283</td>
      <td>5.5518</td>
      <td>1.4493</td>
      <td>-2.6627</td>
      <td>19.8056</td>
      <td>2.3705</td>
      <td>18.4685</td>
      <td>16.3309</td>
      <td>-3.3456</td>
      <td>13.5261</td>
      <td>1.7189</td>
      <td>5.1743</td>
      <td>-7.6938</td>
      <td>9.7685</td>
      <td>4.8910</td>
      <td>12.2198</td>
      <td>11.8503</td>
      <td>-7.8931</td>
      <td>6.4209</td>
      <td>5.9270</td>
      <td>16.0201</td>
      <td>-0.2829</td>
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
</div>



<a id=2><b><pre>Check for the Missing Values.</pre></b></a> 


```python
#Check for Missing Values after Concatination

obs = train.isnull().sum().sort_values(ascending = False)
percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)
pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number of Observations</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>var_199</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_61</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_71</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_70</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_69</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_68</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_67</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_66</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_65</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_64</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_63</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_62</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_60</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_48</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_59</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_58</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_57</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_56</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_55</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_54</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_53</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_52</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_51</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_50</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_72</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_73</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_74</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_75</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_96</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_95</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>var_104</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_103</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_102</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_101</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_122</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_124</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_147</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_125</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_146</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_145</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_144</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_143</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_142</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_141</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_140</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_139</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_138</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_137</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_136</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_135</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_134</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_133</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_132</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_131</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_130</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_129</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_128</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_127</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>var_126</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ID_code</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>202 rows × 2 columns</p>
</div>



<pre>There are no missing values in the dataset</pre>

<pre><a id = 3><b>Visualizing the Satendar Customer Transactions Data</b></a></pre>

<pre><a id = 31 ><b>Check for Class Imbalance</b></a></pre>


```python
target = train['target']
train = train.drop(["ID_code", "target"], axis=1)
sns.set_style('whitegrid')
sns.countplot(target)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a2b392be48>




![png](output_14_1.png)


<pre><a id = 32 ><b>Distribution of Mean and Standard Deviation</b></a></pre>

<pre>EDA Reference : https://www.kaggle.com/gpreda/santander-eda-and-prediction</pre>


```python
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[features].mean(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()
```


![png](output_16_0.png)


<pre>Let's check the distribution of the mean of values per columns in the train and test datasets.</pre>


```python
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train[features].mean(axis=0),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="red", kde=True,bins=120, label='test')
plt.legend();
plt.show()
```


![png](output_18_0.png)


<pre>Distribution for Standard Deviation</pre>


```python
plt.figure(figsize=(16,6))
plt.title("Distribution of std values per rows in the train and test set")
sns.distplot(train[features].std(axis=1),color="blue",kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=1),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()
```


![png](output_20_0.png)


<pre>Let's check the distribution of the standard deviation of values per columns in the train and test datasets.</pre>


```python
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train[features].mean(axis=0),color="blue", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend();plt.show()
```


![png](output_22_0.png)


<pre>Let's check now the distribution of the mean value per row in the train dataset, grouped by value of target</pre>


```python
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_24_0.png)


<pre>Let's check now the distribution of the mean values per columns in the train and test datasets.</pre>


```python
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_26_0.png)


<pre>Let's check now the distribution of the standard deviation  per row in the train dataset, grouped by value of target</pre>


```python
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of standard deviation values per row in the train set")
sns.distplot(t0[features].std(axis=1),color="blue", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].std(axis=1),color="red", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_28_0.png)


<pre>Let's check now the distribution of standard deviation per columns in the train and test datasets.</pre>


```python
t0 = train.loc[target  == 0]
t1 = train.loc[target  == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of standard deviation values per column in the train set")
sns.distplot(t0[features].std(axis=0),color="blue", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].std(axis=0),color="red", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_30_0.png)


<pre><a id = 33 ><b>Distribution of Skewness</b></a></pre>

<pre>Let's see now the distribution of skewness on rows in train separated for values of target 0 and 1. We found the distribution is left skewed</pre>


```python
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per row in the train set")
sns.distplot(t0[features].skew(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_32_0.png)


<pre>Let's see now the distribution of skewness on columns in train separated for values of target 0 and 1.</pre>


```python
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per column in the train set")
sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_34_0.png)


<pre><a id = 34 ><b>Distribution of Kurtosis</b></a></pre>

<pre>Let's see now the distribution of kurtosis on rows in train separated for values of target 0 and 1. We found the distribution to be Leptokurtic</pre>


```python
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per row in the train set")
sns.distplot(t0[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=1),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_37_0.png)


<pre>Let's see now the distribution of kurtosis on columns in train separated for values of target 0 and 1.</pre>


```python
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per column in the train set")
sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=0),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```


![png](output_39_0.png)


<a id=4><pre><b>Principal Component Analysis to check Dimentionality Reduction<b></pre></a>


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)         
PCA_train_x = PCA(2).fit_transform(train_scaled)
plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=target, cmap="copper_r")
plt.axis('off')
plt.colorbar()
plt.show()
```


![png](output_41_0.png)


<pre><a id = 41><b>Kernel PCA (Since the Graph above doesn't represent meaningful analysis)</b></a></pre>


```python
from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)


plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), 
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
       
    PCA_train_x = PCA(2).fit_transform(train_scaled)
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=target, cmap="nipy_spectral_r")
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()
```


![png](output_43_0.png)


<pre>Since PCA hasn't been useful, I decided to proceed with the existing dataset</pre>

<pre><a id = 16><b>Data Augmentation</b></a></pre>


```python
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y
```

<pre><a id = 6><b>Build the Light GBM Model</b></a></pre>


```python
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}
```


```python
train.shape
```




    (200000, 200)




```python
num_folds = 11
features = [c for c in train.columns if c not in ['ID_code', 'target']]

folds = KFold(n_splits=num_folds, random_state=2319)
oof = np.zeros(len(train))
getVal = np.zeros(len(train))
predictions = np.zeros(len(target))
feature_importance_df = pd.DataFrame()

print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    
    X_train, y_train = train.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], target.iloc[val_idx]
    
    X_tr, y_tr = augment(X_train.values, y_train.values)
    X_tr = pd.DataFrame(X_tr)
    
    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    getVal[val_idx]+= clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
```

    Light GBM Model
    Fold idx:1
    Training until validation scores don't improve for 4000 rounds.
    [5000]	training's auc: 0.911892	valid_1's auc: 0.896408
    [10000]	training's auc: 0.921761	valid_1's auc: 0.900838
    [15000]	training's auc: 0.929305	valid_1's auc: 0.901561
    [20000]	training's auc: 0.936156	valid_1's auc: 0.901417
    Early stopping, best iteration is:
    [17288]	training's auc: 0.932505	valid_1's auc: 0.901652
    Fold idx:2
    Training until validation scores don't improve for 4000 rounds.
    [5000]	training's auc: 0.912225	valid_1's auc: 0.895235
    [10000]	training's auc: 0.9221	valid_1's auc: 0.898094
    [15000]	training's auc: 0.929682	valid_1's auc: 0.898226
    Early stopping, best iteration is:
    [12491]	training's auc: 0.92601	valid_1's auc: 0.898381
    Fold idx:3
    Training until validation scores don't improve for 4000 rounds.
    [5000]	training's auc: 0.913119	valid_1's auc: 0.889068
    [10000]	training's auc: 0.92287	valid_1's auc: 0.892629
    [15000]	training's auc: 0.930405	valid_1's auc: 0.89296
    Early stopping, best iteration is:
    [13226]	training's auc: 0.927857	valid_1's auc: 0.89307
    Fold idx:4
    Training until validation scores don't improve for 4000 rounds.
    [5000]	training's auc: 0.911674	valid_1's auc: 0.899838
    [10000]	training's auc: 0.921606	valid_1's auc: 0.902381
    [15000]	training's auc: 0.929224	valid_1's auc: 0.902582
    Early stopping, best iteration is:
    [14237]	training's auc: 0.928125	valid_1's auc: 0.902699
    Fold idx:5
    Training until validation scores don't improve for 4000 rounds.
    [5000]	training's auc: 0.911893	valid_1's auc: 0.895829
    [10000]	training's auc: 0.9218	valid_1's auc: 0.899003
    [15000]	training's auc: 0.929374	valid_1's auc: 0.899348
    [20000]	training's auc: 0.936239	valid_1's auc: 0.899492
    Early stopping, best iteration is:
    [20272]	training's auc: 0.936595	valid_1's auc: 0.899521
    Fold idx:6
    Training until validation scores don't improve for 4000 rounds.
    [5000]	training's auc: 0.912214	valid_1's auc: 0.899031
    [10000]	training's auc: 0.922061	valid_1's auc: 0.90203
    [15000]	training's auc: 0.929643	valid_1's auc: 0.90215
    Early stopping, best iteration is:
    [11899]	training's auc: 0.925069	valid_1's auc: 0.9023
    Fold idx:7
    Training until validation scores don't improve for 4000 rounds.
    


```python
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
```


![png](output_51_0.png)



```python
num_sub = 26
print('Saving the Submission File')
sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = predictions
sub.to_csv('submission{}.csv'.format(num_sub), index=False)
getValue = pd.DataFrame(getVal)
getValue.to_csv("Validation_kfold.csv")
```

    Saving the Submission File
    
