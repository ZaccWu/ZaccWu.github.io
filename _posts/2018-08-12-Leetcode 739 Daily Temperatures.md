---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 739               # 标题 
subtitle:   Daily Temperatures #副标题
date:       2018-08-12             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/739.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
--- 
# 739 Daily Temperatures
**[Medium]**
Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].

***

The main issue that we need to consider is how to 'remember' the current temperature and compare it with those in the future.

Here's my idea, the numbers in the stack represent the position(index) need to be renewed in the return array. If the today's temperature is higher than yesterday,
than the element will be push into the stack.


C Code
```c

// Return an array of size *returnSize.
// Note: The returned array must be malloced, assume caller calls free().

int* dailyTemperatures(int* T, int TSize, int* returnSize) {
    *returnSize=TSize;
    int temp=0;
    int count=TSize-2;
    int *ret=(int*)malloc(sizeof(int)*TSize);
    ret[TSize-1]=0;
    for(int i=TSize-2;i>=0;i--){
        for(int j=1;j+i<TSize;j=j+temp){
            if(T[i]<T[i+j]){
                ret[count--]=j;
                break;
            }
            else if(ret[i+j]==0){
                ret[count--]=0;
                break;
            }           
            temp=ret[i+j];          
        }   
    }
    return ret;
}
```
The idea may be more clearly presented in the following image:

![](https://github.com/Tinky2013/Leetcode-solving/raw/master/img/739.gif)

In the Python solution, I used 'enumerate' to achieve the same goal, ans is the returning array.

Python Code
```python
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        ans = [0]*len(T)
        stack = []
        for i, t in enumerate(T):
            while(stack and T[stack[-1]] < t):
                cur = stack.pop()
                ans[cur] = i - cur
            stack.append(i)
        return ans
```

