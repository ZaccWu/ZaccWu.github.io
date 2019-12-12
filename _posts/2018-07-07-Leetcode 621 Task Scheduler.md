---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 621               # 标题 
subtitle:   Task Scheduler #副标题
date:       2018-07-07             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/621.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 621 Task Scheduler
**[Medium]**
Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between two same tasks, there must be at **least** n intervals that CPU are doing different tasks or just be idle.

You need to return the least number of intervals the CPU will take to finish all the given tasks.


**Example:**

Input: tasks = ["A","A","A","B","B","B"], n = 2

Output: 8

Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
 

Note:
The number of tasks is in the range [1, 10000].
The integer n is in the range [0, 100].

***

This is an really interesting problem and their may be some technique to solve the problem.
For implementation with C, I create an array named 'map' to store the frequency of all the letters.
It will be really convenient to solve the problem by this mean.

C Code
```c
int max(int x, int y){
    return x > y ? x : y; 
}

int leastInterval(char *tasks, int tasksSize, int n)
{
    int map[26] = {0};
    int ans, fre = 0;

    for (int i = 0; i < tasksSize; ++i) {
        int idx = tasks[i] - 'A';
        map[idx]++;
        fre = max(fre, map[idx]);   
    }/* After the circulation, fre means the most frequent letter's appearence time.*/

    ans = (n + 1)*(fre - 1);
    for (int i = 0; i < 26; ++i) {
        if (map[i] == fre)
            ans++;
    }/* For the case that more than one letter appears most frequently */

    return max(tasksSize, ans);
}
```

My original Python solution has the same idea with C, and I used 'sort' function to achieve the goal rather than another circulation.

Python Code
```python
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        Map=[0]*26
        for i in tasks:
            Map[ord(i)-ord('A')]+=1
    
        Map.sort()
        i=25
        while i>=0 and Map[i] == Map[25]:
            i-=1
        return max(len(tasks),((Map[25]-1)*(n+1)+25-i))   
```

Here's another way to solve the problem, this time, we can use Python standard library.

```python
class Solution(object):
  def leastInterval(self, tasks, n):
      d = collections.Counter(tasks) # Counter will count the frequency of all of the letters
      
      counts = d.values()            # Extract the value of the dictionary
      
      longest = max(counts)
      ans = (longest - 1) * (n + 1)
      for count in counts:
          ans += count == longest and 1 or 0
      return max(len(tasks), ans)
```
