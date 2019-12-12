---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 66               # 标题 
subtitle:   Plus One #副标题
date:       2018-07-20             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/66.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 66 Plus One
Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.
You may assume the integer does not contain any leading zero, except the number 0 itself.

**Example 1:**
Input: [1,2,3]
Output: [1,2,4]

Explanation: The array represents the integer 123.

**Example 2:**
Input: [4,3,2,1]
Output: [4,3,2,2]

Explanation: The array represents the integer 4321.

***

The main issue is carry problem and we need to consider that. So I malloc 1 more space unit than 'digitsSize'.

C Code
```c

// Return an array of size *returnSize.
// Note: The returned array must be malloced, assume caller calls free().

int* plusOne(int* digits, int digitsSize, int* returnSize) {
    int n = digitsSize + 1;
    int *ret = (int*)malloc(sizeof(int) * n);
   
    int carry = 1;
    for(int i = digitsSize - 1; i >= 0; i--){
        int sum = digits[i] + carry;
        carry = sum / 10;       /*If exceed 9, than 'carry' will become 1*/
        sum = sum % 10;
        ret[i + 1] = sum;
    }
    if(carry){
        ret[0] = carry;
        *returnSize = digitsSize + 1;
        return ret;
    }
    *returnSize = digitsSize;
    return ret + 1;
}
```

Python will be more flexible to solve this problem, for its great operation on list.

Python Code
```python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        for i in range(len(digits)-1, -1, -1):
            carry, digits[i] = divmod(digits[i]+carry, 10) #divide and remainer
            if carry == 0:
                return digits
        return [1]+digits
```
