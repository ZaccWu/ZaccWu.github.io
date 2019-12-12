---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 104               # 标题 
subtitle:   Maximum Depth of Binary Tree #副标题
date:       2018-07-01             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/104.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 104 Maximum Depth of Binary Tree
**[Easy]**
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

***

My idea is that we can operate on the left-hand-side and right-hand-side at the same time.
Recursion may be suitable for this question.
The easy implementation with C code is as follow:

C Code
```c 

// Definition for a binary tree node.
// struct TreeNode {
//     int val;
//     struct TreeNode *left;
//     struct TreeNode *right;
// };

int maxDepth(struct TreeNode *root) {
	if(NULL==root) return 0;
    
	int LeftDepth=maxDepth(root->left);
	int RightDepth=maxDepth(root->right);

	return LeftDepth>RightDepth? LeftDepth+1 : RightDepth+1;
}
```

Here's the solution with Python.

Python Code
```python
# Definition for a binary tree node.

# class TreeNode(object):

#     def __init__(self, x):

#         self.val = x

#         self.left = None

#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return 1+max(self.maxDepth(root.left),self.maxDepth(root.right))
```

When I was refering to order's code, I found some interesting way to solve this problem.

The first one uses stack and the second one uses queue.


```python
# Other solutions
# Using Stack

class Solution(object):
    def maxDepth(self, root):     
        if not root:
            return 0
        tstack,h = [root],0

        #count number of levels
        while tstack:
            nextlevel = []
            while tstack:
                top = tstack.pop()
                if top.left:
                     nextlevel.append(top.left)
                if top.right:
                     nextlevel.append(top.right)
            tstack = nextlevel
            h+=1
        return h
        
# Using Queue

class Solution(object):
    def maxDepth(self, root):     
        if not root:
             return 0

        tqueue, h = collections.deque(),0
        tqueue.append(root)
        while tqueue:
            nextlevel = collections.deque()
            while tqueue:
                front = tqueue.popleft()
                if front.left:
                    nextlevel.append(front.left)
                if front.right:
                    nextlevel.append(front.right)
            tqueue = nextlevel
            h += 1
        return h
```
