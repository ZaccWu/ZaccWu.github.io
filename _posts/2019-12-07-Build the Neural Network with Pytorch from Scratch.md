---
layout:     post                    # 使用的布局（不需要改）
title:      Build the Neural Network with Pytorch from Scratch             # 标题 
subtitle:    #副标题
date:       2019-12-07             # 时间
author:     WZY                      # 作者
header-img: img/post-bg-universe.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Deeplearning
--- 

# Build the Neural Network with Pytorch from Scratch

In this article I summarize the constructing processes of different kinds of neural networks. By using the tools in Pytorch, you can build these neural networks and train your data swiftly!

```python
import torch
import torch.nn as nn
import torch.utils.data as Data # Abstract class for data set in pytorch

import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
```

## 1 CNN

First we need to import the data. Here I use the popular dataset in deeplearning: MNIST.


```python
'''
001 CNN
'''
torch.manual_seed(1)

# Hyper parameters
EPOCH=1     # how many times we train the whole data

BATCH_SIZE=50
LR=0.001    # learning rate

DOWNLOAD_MNIST=True

# mnist
train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),    # transform PIL.Image or numpy.ndarray to torch.FloatTensor (C x H x W)
    
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# batch training 50samples, 1 channel, 28x28 (50, 1, 28, 28)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255.
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# torch.Size([2000, 1, 28, 28])

test_y = test_data.targets[:2000]   # Test the first 2000

# torch.Size([2000])
```

The structure of CNN is relative simple, here's the exhaustive structure of the network.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height

                out_channels=16,            # n_filters

                kernel_size=5,              # filter size

                stride=1,                   # filter movement/step

                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1

            ),                              # output shape (16, 28, 28)

            nn.ReLU(),                      # activation

            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)

            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)

            nn.ReLU(),                      # activation

            nn.MaxPool2d(2),                # output shape (32, 7, 7)

        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)

        output = self.out(x)
        return output, x    # return x for visualization

cnn=CNN()
print(cnn)
```

    CNN(
      (conv1): Sequential(
        (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv2): Sequential(
        (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (out): Linear(in_features=1568, out_features=10, bias=True)
    )
    
This network has two convolution layers. We use ReLU activation function and MaxPooling. Also the network has a fully connected layer(Linear).

In the training process we pack x and y using 'Variable', and put them into the CNN model to calculate the output and the error.

```python
# training
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()      # the target label is not one-hottedfor epoch in range(EPOCH):
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        # b_x, b_y: torch.tensor
        b_x = Variable(x)   # batch x; torch.Size([50, 1, 28, 28])
        b_y = Variable(y)   # batch y; torch.Size([50])

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # print the training process: Epoch, loss and accuracy
        if step % 100 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

# We can check the performance with 10 new data
# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
```

    Epoch:  0 | train loss: 2.3105 | test accuracy: 0.06
    Epoch:  0 | train loss: 0.1270 | test accuracy: 0.87
    Epoch:  0 | train loss: 0.4052 | test accuracy: 0.93
    Epoch:  0 | train loss: 0.1942 | test accuracy: 0.95
    Epoch:  0 | train loss: 0.1270 | test accuracy: 0.96
    Epoch:  0 | train loss: 0.2244 | test accuracy: 0.96
    Epoch:  0 | train loss: 0.0242 | test accuracy: 0.97
    Epoch:  0 | train loss: 0.2164 | test accuracy: 0.97
    Epoch:  0 | train loss: 0.0433 | test accuracy: 0.97
    Epoch:  0 | train loss: 0.0551 | test accuracy: 0.98
    Epoch:  0 | train loss: 0.0315 | test accuracy: 0.98
    Epoch:  0 | train loss: 0.0264 | test accuracy: 0.98
    [7 2 1 0 4 1 4 9 5 9] prediction number
    [7 2 1 0 4 1 4 9 5 9] real number
    

## 2 RNN LSTM

Same as the issue discussed above, we use MNIST dataset to test the model.

```python
'''
002
RNN LSTM
'''
torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28      # picture hight
INPUT_SIZE = 28     # picture length
LR = 0.01           # learning rate
DOWNLOAD_MNIST = True

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255. # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.targets.numpy().squeeze()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            input_size=28,      # picture length
            hidden_size=64,     # rnn hidden unit
            num_layers=1,       # rnn layers
            batch_first=True,   # (batch,time_step,input_size)
        )
        self.out=nn.Linear(64,10)

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n\h_c shape (n_layers, batch, hidden_size)
        r_out,(h_n,h_c)=self.rnn(x,None)

        out=self.out(r_out[:,-1,:])
        return out

rnn=RNN()
print(rnn)
```

    RNN(
      (rnn): LSTM(28, 64, batch_first=True)
      (out): Linear(in_features=64, out_features=10, bias=True)
    )
    
Now we train the model, we can view the image as continuous data. Every pixel can be the input at the certain moment.

```python
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=Variable(x.view(-1,28,28))    # reshape x to (batch,time_step,input_size)
        b_y=Variable(y)

        output=rnn(b_x)   # rnn output
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y==test_y)/float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

# test with ten example
test_output=rnn(test_x[:10].view(-1,28,28))
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10],'real number')
```

    Epoch:  0 | train loss: 2.2883 | test accuracy: 0.10
    Epoch:  0 | train loss: 0.8128 | test accuracy: 0.60
    Epoch:  0 | train loss: 0.9328 | test accuracy: 0.74
    Epoch:  0 | train loss: 0.8865 | test accuracy: 0.78
    Epoch:  0 | train loss: 0.2789 | test accuracy: 0.87
    Epoch:  0 | train loss: 0.2639 | test accuracy: 0.87
    Epoch:  0 | train loss: 0.3271 | test accuracy: 0.91
    Epoch:  0 | train loss: 0.4145 | test accuracy: 0.93
    Epoch:  0 | train loss: 0.1252 | test accuracy: 0.93
    Epoch:  0 | train loss: 0.3034 | test accuracy: 0.91
    Epoch:  0 | train loss: 0.0502 | test accuracy: 0.91
    Epoch:  0 | train loss: 0.2650 | test accuracy: 0.94
    Epoch:  0 | train loss: 0.0322 | test accuracy: 0.95
    Epoch:  0 | train loss: 0.1711 | test accuracy: 0.95
    Epoch:  0 | train loss: 0.2042 | test accuracy: 0.95
    Epoch:  0 | train loss: 0.1331 | test accuracy: 0.94
    Epoch:  0 | train loss: 0.0755 | test accuracy: 0.95
    Epoch:  0 | train loss: 0.1991 | test accuracy: 0.95
    Epoch:  0 | train loss: 0.3084 | test accuracy: 0.95
    [7 2 1 0 4 1 4 9 6 9] prediction number
    [7 2 1 0 4 1 4 9 5 9] real number
    
## 3 RNN regression

```python
'''
003
RNN regression
'''
# sin to predict cos
import numpy as np

# Hyper parameters
TIME_STEP=10    # rnn time step / image height
INPUT_SIZE=1    # rnn input size / image width
LR=0.02         # learning rate

# show data
steps=np.linspace(0,np.pi*2,100,dtype=np.float32)
x_np=np.sin(steps)
y_np=np.cos(steps)
plt.plot(steps,y_np,'r-',label='target(cos)')
plt.plot(steps,x_np,'b-',label='input(sin)')
plt.legend(loc='best')
plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32, # rnn hidden unit
            num_layers=1,   # how many layer in RNN
            batch_first=True,   # (batch, time_step, input_size)
        )
        self.out=nn.Linear(32,1)

    def forward(self,x,h_state):    # the hidden state is continuous, so we need to continuously converting this state
        # x (batch,time_step,input_size)
        # h_state (n_layers,batch,hidden_state)
        # r_out (batch,time_step,output_size)
        r_out,h_state=self.rnn(x,h_state)   # h_state need to be a input of RNN

        outs=[] # save all the predict values at all moments
        for time_step in range(r_out.size(1)):  # calculate the output at each data point
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state

rnn=RNN()
print(rnn)
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_25_0.png)


    RNN(
      (rnn): RNN(1, 32, batch_first=True)
      (out): Linear(in_features=32, out_features=1, bias=True)
    )
    
Now we define the optimizer.

```python
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.MSELoss()

plt.figure(1,figsize=(12,5))
plt.ion()   # continue plot
plt.show()

h_state=None
for step in range(60):
    start,end=step*np.pi, (step+1)*np.pi    # time steps
    steps=np.linspace(start,end,TIME_STEP,dtype=np.float32)
    x_np=np.sin(steps)
    y_np=np.cos(steps)

    x=Variable(torch.from_numpy(x_np[np.newaxis,:,np.newaxis])) # shape (batch,time_step,input_size)
    y=Variable(torch.from_numpy(y_np[np.newaxis,:,np.newaxis]))

    prediction,h_state=rnn(x,h_state)
    h_state=h_state.data

    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plot
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw();
    plt.pause(0.05)
```

Now we can fit COS function with input SIN function.

## 4 AutoEncoder

The neural network can do the unsupervised learning, we only need the training data without labels. The tool is AutoEncoder.

```python
'''
004
AutoEncoder
'''
#Hyper parameters
EPOCH=10
BATCH_SIZE=64
LR=0.005
N_TEST_IMG=5

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        # Compress the data
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3),    # Compress to 3 features and we can visualize them
        )

        # Unpack the data
        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            nn.Sigmoid(),   # output (0,1)
        )

    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded

autoencoder=AutoEncoder()
print(autoencoder)
```

    AutoEncoder(
      (encoder): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): Tanh()
        (2): Linear(in_features=128, out_features=64, bias=True)
        (3): Tanh()
        (4): Linear(in_features=64, out_features=12, bias=True)
        (5): Tanh()
        (6): Linear(in_features=12, out_features=3, bias=True)
      )
      (decoder): Sequential(
        (0): Linear(in_features=3, out_features=12, bias=True)
        (1): Tanh()
        (2): Linear(in_features=12, out_features=64, bias=True)
        (3): Tanh()
        (4): Linear(in_features=64, out_features=128, bias=True)
        (5): Tanh()
        (6): Linear(in_features=128, out_features=784, bias=True)
        (7): Sigmoid()
      )
    )
    
Now we can train the data, we can compare the original data with the information output of 'decoder'.

```python
optimizer=torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func=nn.MSELoss()

# for visualization
view_data = Variable(train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.)

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=Variable(x.view(-1,28*28))  # batch x, shape(batch,28*28)
        b_y=Variable(x.view(-1,28*28))  # batch y, shape(batch,28*28)
        b_label=Variable(y)             # batch label

        encoded,decoded=autoencoder(b_x)

        loss=loss_func(decoded,b_y) # mean square error
        optimizer.zero_grad()   # clear gradients for this training step
        loss.backward()     # backpropagation, compute gradients
        optimizer.step()    # apply gradients

        # plot
        if step % 500 == 0 and epoch in [0, 5, EPOCH - 1]:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)

            # initialize figure
            f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))

            for i in range(N_TEST_IMG):
                a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
                a[0][i].set_xticks(());
                a[0][i].set_yticks(())

            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())
            plt.show();
            plt.pause(0.05)

# visualize in 3D plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
view_data = Variable(train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.)
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()
```

    Epoch:  0 | train loss: 0.2324
    


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_28_1.png)


    Epoch:  0 | train loss: 0.0542
    


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_28_3.png)


    Epoch:  5 | train loss: 0.0360
    


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_28_5.png)


    Epoch:  5 | train loss: 0.0362
    


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_28_7.png)


    Epoch:  9 | train loss: 0.0356
    


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_28_9.png)


    Epoch:  9 | train loss: 0.0378
    


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_28_11.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_28_12.png)


## 5 DQN

In this part, we can use Pytorch to build network for reinforcement learning. Here we need **gym** module.

```python
'''
005
DQN
'''
import gym
# Hyper Parameters
BATCH_SIZE=32
LR=0.01
EPSILON=0.9             # greedy policy
GAMMA=0.9               # reward discount
TARGET_REPLACE_ITER=100 # target update frequency
MEMORY_CAPACITY=2000    # The size of Memory Capacity
# example
env=gym.make('CartPole-v0')
env=env.unwrapped
N_ACTIONS=env.action_space.n    # The action of the rod
N_STATES=env.observation_space.shape[0] # The information in the environment that the rod can get

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(N_STATES,10)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.out=nn.Linear(10,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)
        actions_value=self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        # build the target net and eval net as well as memory
        self.eval_net,self.target_net=Net(),Net()
        self.learn_step_counter=0   # for target updating
        self.memory_counter=0       # for storing memory
        self.memory=np.zeros((MEMORY_CAPACITY,N_STATES*2+2))
                                    # initialize memory
        self.optimizer=torch.optim.Ada(self.eval_net.parameters(),lr=LR)    # the torch optimizer
        self.loss_func=nn.MSELoss() # error function

    def choose_action(self,x):
        # choose the action based on the observation of environment
        x=Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        # input one sample
        if np.random.uniform()<EPSILON:     # select the optimal action (greedy)
            actions_value=self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # return the argmax
        else:   # randomly choose the action
            action=np.random.randint(0,N_ACTIONS)
        return action

    def store_transition(self,s,a,r,s_):
        # Store the memory
        transition=np.hstack((s,[a,r],s_))
        # replace the old memory with new memory
        index=self.memory_counter % MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.memory_counter+=1

    def learn(self):
        # target net update
        # the memory in the learning capacity
        if self.learn_step_counter % TARGET_REPLACE_ITER==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index=np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory=self.memory[sample_index,:]
        b_s=Variable(torch.FloatTensor(b_memory[:,:N_STATES]))
        b_a=Variable(torch.LongTensor(b_memory[:,N_STATES:N_STATES+1].astype(int)))
        b_r=Variable(torch.FloatTensor(b_memory[:,N_STATES+1:N_STATES+2]))
        b_s_=Variable(torch.FloatTensor(b_memory[:,-N_STATES:]))

        q_eval=self.eval_net(b_s).gather(1,b_a)
        q_next=self.target_net(b_s_).detach()   # shape (batch,1)
        q_target=b_r+GAMMA*q_next.max(1)[0]
        loss=self.loss_func(q_eval,q_target)    # shape (batch,1)

        # calculate, renew eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn=DQN()
```

Now we train the model:

```python
# the process of reinforce learning
print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()    # visualization
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward, make it learn faster
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # store the memory
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
```

    
    Collecting experience...
    Ep:  198 | Ep_r:  2.37
    Ep:  199 | Ep_r:  3.79
    Ep:  200 | Ep_r:  1.41
    Ep:  201 | Ep_r:  2.47
    Ep:  202 | Ep_r:  2.72
    ...
    Ep:  395 | Ep_r:  2.57
    Ep:  396 | Ep_r:  1.45
    Ep:  397 | Ep_r:  2.93
    Ep:  398 | Ep_r:  1.0
    Ep:  399 | Ep_r:  3.19
    

## 6 GAN

```python
'''
006
GAN
'''
torch.manual_seed(1)
np.random.seed(1)

# Hyper parameters
BATCH_SIZE=64
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# plot
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()

def artist_works():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)

G = nn.Sequential(                      # Generator
    nn.Linear(N_IDEAS, 128),            # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),     # making a painting from these random ideas
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(ART_COMPONENTS, 128),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)


opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_31_0.png)



```python
for step in range(10000):
    artist_paintings = artist_works()           # real painting from artist
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))    # random ideas
    G_paintings = G(G_ideas)                    # fake painting from G (random ideas)

    prob_artist0 = D(artist_paintings)          # D try to increase this prob
    prob_artist1 = D(G_paintings)               # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # retain_variables for reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    # plot
    if step % 1000 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=12);plt.draw();plt.pause(0.01)
        plt.show()
```


![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_0.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_1.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_2.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_3.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_4.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_5.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_6.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_7.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_8.png)



![png](https://github.com/Tinky2013/Machine-Learning-Lab/raw/master/img/Pytorch/output_32_9.png)

