---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 682               # 标题 
subtitle:   Baseball Game #副标题
date:       2018-07-25             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/682.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
--- 
# 682 Baseball Game
**[Easy]**
You're now a baseball game point recorder.

Given a list of strings, each string can be one of the 4 following types:

Integer (one round's score): Directly represents the number of points you get in this round.

"+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.

"D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.

"C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.

Each round's operation is permanent and could have an impact on the round before and the round after.

You need to return the sum of the points you could get in all the rounds.

**Example 1:**

Input: ["5","2","C","D","+"]
Output: 30

Explanation: 

Round 1: You could get 5 points. The sum is: 5.
Round 2: You could get 2 points. The sum is: 7.
Operation 1: The round 2's data was invalid. The sum is: 5.  
Round 3: You could get 10 points (the round 2's data has been removed). The sum is: 15.
Round 4: You could get 5 + 10 = 15 points. The sum is: 30.

**Example 2:**

Input: ["5","-2","4","C","D","9","+","+"]
Output: 27

Explanation: 

Round 1: You could get 5 points. The sum is: 5.
Round 2: You could get -2 points. The sum is: 3.
Round 3: You could get 4 points. The sum is: 7.
Operation 1: The round 3's data is invalid. The sum is: 3.  
Round 4: You could get -4 points (the round 3's data has been removed). The sum is: -1.
Round 5: You could get 9 points. The sum is: 8.
Round 6: You could get -4 + 9 = 5 points. The sum is 13.
Round 7: You could get 9 + 5 = 14 points. The sum is 27.

Note:
The size of the input list will be between 1 and 1000.
Every integer represented in the list will be between -30000 and 30000.

***

My first solution is a regular way to solve this problem with C. 

C Code
```c
int calPoints(char** ops, int opsSize) {
    int* pint = malloc(sizeof(int)*opsSize), idx = -1, mark = 0;
    char c;
    for(int i = 0; i < opsSize; i++){
        c = *(ops[i]);
        if(c == 'C'){
            mark -= pint[idx--]; 
        }else{
            if(isdigit(c) || c == '-' ) 
                pint[++idx] = atoi(ops[i]);     /* Converts a string to an integer */
            else if(c == 'D') 
                pint[++idx] = pint[idx]*2; 
            else if(c == '+')
                pint[++idx] = pint[idx] + pint[idx - 1];
            else
                return 0;
            mark += pint[idx];
        }
    }
    free(pint);
    return mark;
}
```

The C code may be a little complicated. I used the idea of stack in my Python code and the gif below will show the priciple of the operation.

Python Code
```python
class Solution(object):
    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        mark=[]
        for i in range(len(ops)):
            if ops[i]=='D':
                mark.append(2*mark[-1])
            elif ops[i]=='C':
                mark.pop()
            elif ops[i]=='+':
                mark.append(mark[-1]+mark[-2])
            else:
                mark.append(int(ops[i]))
        return sum(mark)
```

![](https://github.com/Tinky2013/Leetcode-solving/raw/master/img/682.gif)
