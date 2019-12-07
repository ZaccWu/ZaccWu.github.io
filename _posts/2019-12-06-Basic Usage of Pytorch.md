---
layout:     post                    # 使用的布局（不需要改）
title:      Basic Usage of Pytorch               # 标题 
subtitle:    #副标题
date:       2019-12-06             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Deeplearning
--- 

# Basic Usage of Pytorch

In this passage we summarize the basic usage of Pytorch. Pytorch is really convenient when we need to build a neural network and do all kinds of work in Deeplearning.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

## 1 Compare with Numpy

### 1.1 Import

```python
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          
    '\ntorch tensor:', torch_data,      
    '\ntensor to array:', tensor2array, 
)
```

    
    numpy array: [[0 1 2]
     [3 4 5]] 
    torch tensor: tensor([[0, 1, 2],
            [3, 4, 5]], dtype=torch.int32) 
    tensor to array: [[0 1 2]
     [3 4 5]]
    
By this mean we can implement interconversion between ndarray and torch tensor.

### 1.2 Operations

Basic operations may include the followings:

* Abs

```python
# Operations
# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),
    '\ntorch: ', torch.abs(tensor)
)
```

    
    abs 
    numpy:  [1 2 1 2] 
    torch:  tensor([1., 2., 1., 2.])
    
* Sin

```python
# sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),
    '\ntorch: ', torch.sin(tensor)
)
```

    
    sin 
    numpy:  [-0.84147098 -0.90929743  0.84147098  0.90929743] 
    torch:  tensor([-0.8415, -0.9093,  0.8415,  0.9093])
    
* Mean

```python
# mean
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),
    '\ntorch: ', torch.mean(tensor)
)
```

    
    mean 
    numpy:  0.0 
    torch:  tensor(0.)
    
* Matrix multiplication

```python
# matrix multiplication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)
print(

    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),
    '\ntorch: ', torch.mm(tensor, tensor)
)
```

    
    matrix multiplication (matmul) 
    numpy:  [[ 7 10]
     [15 22]] 
    torch:  tensor([[ 7., 10.],
            [15., 22.]])
    

## 2 Variable in pytorch

```python
'''
002
Variable
'''
from torch.autograd import Variable
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)
print(tensor)
print(variable)

t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)
```

    tensor([[1., 2.],
            [3., 4.]])
    tensor([[1., 2.],
            [3., 4.]], requires_grad=True)
    tensor(7.5000)
    tensor(7.5000, grad_fn=<MeanBackward1>)
    

## 3 Activation function

Pytorch includes frequently-used activation functions and we can use them directly.

For example, let's visualize the activation functions: relu, sigmoid and tanh.

```python
'''
003
Activation function
'''
x=torch.linspace(-5,5,200)
x=Variable(x)

x_np = x.data.numpy()
y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()

import matplotlib.pyplot as plt
plt.figure(1, figsize=(18, 6))
plt.subplot(131)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(132)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(133)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
plt.show()
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_7_0.png)


## 4 Using Pytorch for Regression

### 4.1 Create the dataset and construct the network

To test this module, we create a representative dataset.

```python
'''
004
Regression
'''
# Create dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_8_0.png)

Here's what the datasets look like. Now we create a basic Nerual Network.

```python
# Create NN
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # inherit  __init__ function
        # define the form of each level
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # Forward propagation of the input value, neural network analysis of the output value
        x = torch.relu(self.hidden(x))      # activation function
        x = self.predict(x)             # output value
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)
```

    Net(
      (hidden): Linear(in_features=1, out_features=10, bias=True)
      (predict): Linear(in_features=10, out_features=1, bias=True)
    )
    
We can see the structure of our neural network. The training process includes: Calling, Loss definition, Derivation, Error back propagation and Parameter renew.

### 4.2 Train the net

```python
# train the net
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x) # Calling the network
    loss = loss_func(prediction, y) # loss function

    optimizer.zero_grad()   # Clear the remaining update parameter values from the previous step
    loss.backward(retain_graph=True)         # Error back propagation and calculation of parameter update value
    optimizer.step()        # Renew the parameter
```

### 4.3 Visualization

```python
# Visualize the training process
import matplotlib.pyplot as plt
plt.ion()
plt.show()
for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad() 
    loss.backward(retain_graph=True)
    optimizer.step()

    if t % 40 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
        plt.pause(0.1)
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_11_0.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_11_1.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_11_2.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_11_3.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_11_4.png)


## 5 Using Pytorch for Classification

### 5.1 Create the dataset and construct the network


```python
'''
005
Classification
'''
import torch.nn.functional as F 
n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)

# Combine the data
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.LongTensor)

x,y=Variable(x),Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_12_0.png)



```python
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        return x
net = Net(2,10,2)
print(net)

```

    Net(
      (hidden): Linear(in_features=2, out_features=10, bias=True)
      (out): Linear(in_features=10, out_features=2, bias=True)
    )
    

### 5.2 Train the net

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()


```

### 5.3 Visualization

```python
plt.ion()
plt.show()
for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    if t % 100 == 0:
        plt.cla()
        prediction = torch.max(torch.sigmoid(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 10, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_15_0.png)


## 6 Build a Neural Network Swiftly

The modules are integrated well in pytorch so we can build the neural network swiftly. Here's an example, we can combine the exhaustive one and the concise one.

```python
'''
006 
Build a NN swiftly
'''
# The older one
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_feature,n_output)

    def forward(self,x):
        x=torch.relu(self.hidden(x))
        x=self.predict(x)
        return x
net1=Net(1,10,1)

# The concise one
net2=torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)
print(net1)
print(net2)
```

    Net(
      (hidden): Linear(in_features=1, out_features=10, bias=True)
      (predict): Linear(in_features=1, out_features=1, bias=True)
    )
    Sequential(
      (0): Linear(in_features=1, out_features=10, bias=True)
      (1): ReLU()
      (2): Linear(in_features=10, out_features=1, bias=True)
    )
    

## 7 Data Loader

Data loader can help us iterate the data efficiently.

```python
'''
007
Data Loader
'''
import torch.utils.data as Data
torch.manual_seed(1)
BATCH_SIZE=5
x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)

# put the dataset into DataLoader
loader=Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,           # whether disorganize the data
    num_workers=2,          # multithreading
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        '''
        Here's the place you train your data.
        Every step the loader release a small batch of data for learning
        '''
        print('Epoch: ',epoch,'| Step: ',step,' | batch x: ',batch_x.numpy(),' | batch y: ',batch_y.numpy())
        '''
        We can see that 5 data are exported in each step.
        The data was disordered than exported in each epoch.
        '''
```

    Epoch:  0 | Step:  0  | batch x:  [ 5.  7. 10.  3.  4.]  | batch y:  [6. 4. 1. 8. 7.]
    Epoch:  0 | Step:  1  | batch x:  [2. 1. 8. 9. 6.]  | batch y:  [ 9. 10.  3.  2.  5.]
    Epoch:  1 | Step:  0  | batch x:  [ 4.  6.  7. 10.  8.]  | batch y:  [7. 5. 4. 1. 3.]
    Epoch:  1 | Step:  1  | batch x:  [5. 3. 2. 1. 9.]  | batch y:  [ 6.  8.  9. 10.  2.]
    Epoch:  2 | Step:  0  | batch x:  [ 4.  2.  5.  6. 10.]  | batch y:  [7. 9. 6. 5. 1.]
    Epoch:  2 | Step:  1  | batch x:  [3. 9. 1. 8. 7.]  | batch y:  [ 8.  2. 10.  3.  4.]
    

We can see that in every step, the loader export 5 data for learning process. The data are disorganized first before exporting.

## 8 Optimizer

In this section we create different types of optimizers, including SGD, momentum, RMSprop, Adam. We can plot the training process of different optimizer. These optimizers are evaluated by the same loss function.

```python
'''
008
Optimizer
'''
# Prepare the data
torch.manual_seed(1)    # reproducible
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12
# fake dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()
# Using data loader
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)
    def forward(self,x):
        x=torch.relu(self.hidden(x))
        x=self.predict(x)
        return x

net_SGD=Net()
net_Momentum=Net()
net_RMSprop=Net()
net_Adam=Net()
nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]

# different optimizers
opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

loss_func=torch.nn.MSELoss()
losses_his=[[],[],[],[]]

# training and plot
for epoch in range(EPOCH):
    print('Epoch: ',epoch)
    for step, (b_x, b_y) in enumerate(loader):
        for net, opt, l_his in zip(nets,optimizers,losses_his):
            output=net(b_x)
            loss=loss_func(output,b_y)
            opt.zero_grad()     # clear gradients for next train
            loss.backward()     # backpropagation, compute gradients
            opt.step()          # apply gradients
            l_his.append(loss.data.numpy())

labels=['SGD','Momentum','RMSprop','Adam']
for i,l_his in enumerate(losses_his):
    plt.plot(l_his,label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_18_0.png)


    Epoch:  0
    Epoch:  1
    Epoch:  2
    Epoch:  3
    Epoch:  4
    Epoch:  5
    Epoch:  6
    Epoch:  7
    Epoch:  8
    Epoch:  9
    Epoch:  10
    Epoch:  11
    


![png](output_18_2.png)

