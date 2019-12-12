---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 100               # 标题 
subtitle:   Same Tree #副标题
date:       2018-07-31             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/100.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 100 Same Tree
**[Easy]**
Given two binary trees, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

***

The best way to solve this kind of problem is using recursion or iteration. Here's a simple way implemented by C:

C Code
```c

// Definition for a binary tree node.
// struct TreeNode {
//     int val;
//     struct TreeNode *left;
//     struct TreeNode *right;
// };

bool isSameTree(struct TreeNode* p, struct TreeNode* q) {
    if(p==NULL && q==NULL) return true;
    if((p==NULL && q!=NULL) || (p!=NULL && q==NULL)) return false;
    if(p->val!=q->val) return false;
    return isSameTree(p->left,q->left) && isSameTree(p->right,q->right);
}
```

We need to judge the special condition first. When the two trees are all empty, they are also the same. When one of them is empty but the other is not, the function will return false.
After the judgment, the code will process the same operation with the left child and the right child.

Here's the Python solutions, as you can see, there are more interesting implementation using Python.
The code has the same idea with C above:

Python Code
```python
# Definition for a binary tree node.

# class TreeNode(object):

#     def __init__(self, x):

#         self.val = x

#         self.left = None

#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return p is q
```

Here's the code I found in the discussion table, triple is also suitable for solving this kind of problem.

```python
class Solution(object):
    def isSameTree(self, p, q):
        def t(n):
            return n and (n.val, t(n.left), t(n.right))
        return t(p) == t(q)
```

And the following code can achieve the goal with ONE LINE:

```python
class Solution(object):
    def isSameTree(self, p, q):
        return p and q and p.val == q.val and all(map(self.isSameTree, (p.left, p.right), (q.left, q.right))) or p is q
```
