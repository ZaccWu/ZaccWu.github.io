---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 496               # 标题 
subtitle:   Next Great Element I #副标题
date:       2018-05-30             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/496.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 496 Next Great Element I
**[Easy]**
You are given two arrays (without duplicates) nums1 and nums2 where nums1’s elements are subset of nums2. 
Find all the next greater numbers for nums1's elements in the corresponding places of nums2.

The Next Greater Number of a number x in nums1 is the first greater number to its right in nums2. If it does not exist, output -1 for this number.

**Example 1**:
Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
Output: [-1,3,-1]

Explanation:
For number 4 in the first array, you cannot find the next greater number for it in the second array, so output -1.
For number 1 in the first array, the next greater number for it in the second array is 3.
For number 2 in the first array, there is no next greater number for it in the second array, so output -1.
    
**Example 2**:
Input: nums1 = [2,4], nums2 = [1,2,3,4].
Output: [3,-1]

Explanation:
For number 2 in the first array, the next greater number for it in the second array is 3.
For number 4 in the first array, there is no next greater number for it in the second array, so output -1.
    
Note:
All elements in nums1 and nums2 are unique.
The length of both nums1 and nums2 would not exceed 1000.

***

This question is labeled with 'stack', but the first idea that came across me is not using stack.
Here's a solution. First I used two loops to find out the same elements in the two arrays. The innermost loop is used for finding the next greater elements in array2.

C Code
```c
/**
 * Return an array of size *returnSize.
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* nextGreaterElement(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize) {
    int* result =(int*)malloc(nums1Size*sizeof(int));
    *returnSize = nums1Size;
    for (int i = 0; i < nums1Size ; i ++)
        for (int j = 0; j < nums2Size; j ++) {
            if (*(nums2 + j) == *(nums1 + i)) {
                int k = j + 1;
                for (; k < nums2Size; k++) {
                    if (*(nums2 + k) > *(nums2 + j)) {
                        *(result + i) = *(nums2 + k);
                        break;
                    }
                }
                if (k == nums2Size) {
                    *(result + i) = -1;
                }
                break;
            }
        }
    return result;
}
```

Here I found a short Python solution. 'map' function put every elements in nums1 into function 'helper'.
A technique here is to use index(), which can detect whether a string has a certain substring. It will return the location where the substring located.

Python Code
```python
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        def helper(num):
            for tmp in nums2[nums2.index(num):]:
                if tmp > num:
                    return tmp
            return -1
        return map(helper, nums1)
        
```

### Other solutions
Here another faster Python solution that use stack:
```python
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if not nums1 or not nums2:
            return []
        
        set_1 = set(nums1)
        memo = {nums2[-1]: -1}
        
        N = len(nums2)
        stack = [nums2[-1]]
        
        for i in range(N-2, -1, -1):
            n = nums2[i]
            while stack and stack[-1] < n:
                stack.pop()
            
            if n in set_1 and stack:
                memo[n] = stack[-1]
            
            stack.append(n)
            
        res = []
        
        for n in nums1:
            if n in memo:
                res.append(memo[n])
            else:
                res.append(-1)
        return res
```
