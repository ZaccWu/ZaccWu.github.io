# 数据操作笔记

## 1 数据预处理

### 1.1 基本认识

#### 1.1.1 直观认识

```python
'''
以下数据都基于pd.dataFrame
'''
# 1.1.1.1 查看基本信息
data.info       

# 1.1.1.2 简单描述
data.describe   

# 1.1.1.3 显示前几行/后几行
data.head()     
data.tail()
```

#### 1.1.2 查看数据维度/大小的两种方法

```python
# 1.1.1.4
data.shape  # 其中，查看行数可用data.shape[1]，查看列数可用data.shape[2]
data.size   # 输出行列数乘积
```
### 1.2 数据预处理

#### 1.2.1 缺失值处理

## 2 数据可视化
