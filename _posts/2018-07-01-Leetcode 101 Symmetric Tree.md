---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 101               # 标题 
subtitle:   Symmetric Tree #副标题
date:       2018-07-01             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/101.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 101 Symmetric Tree
**[Easy]**
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

***

When I first met with the problem I thought that it can be solved by similar idea with the last question(100.Same Tree), however, this question may be a little more complex. 
Instead of comparing in one of the subtrees at the end, the separation operation should take place at the nearest layer to the root.

To solve the problem, I create another function that can operate two nodes at the same time, and again, I used recursion.

C Code
```c
// Definition for a binary tree node.
// struct TreeNode {
//     int val;
//     struct TreeNode *left;
//     struct TreeNode *right;
// };

bool checknode(struct TreeNode* left,struct TreeNode* right){
    if(left==NULL && right==NULL) return true;
    if((left!=NULL && right==NULL)||(left==NULL && right!=NULL)) return false;
    if(left->val!=right->val) return false;
    return(checknode(left->left,right->right) && checknode(left->right,right->left));
}

bool isSymmetric(struct TreeNode* root) {
    if(root==NULL) return true;
    return(checknode(root->left,root->right));
}
```

The Python solution uses the same idea but it is somehow more concise.

Python Code
```python
# Definition for a binary tree node.

# class TreeNode:

#     def __init__(self, x):

#         self.val = x

#         self.left = None

#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        return not root or self.is_same(root.left, root.right)
    
    def is_same(self, left, right):
        return left and right and left.val == right.val and self.is_same(left.left, right.right) and self.is_same(left.right, right.left) or left is right
        
```

And here's Python solutions created by other people:

```python
    def isSymmetric(self, root):
        def isSym(L,R):
            if not L and not R: return True
            if L and R and L.val == R.val: 
                return isSym(L.left, R.right) and isSym(L.right, R.left)
            return False
        return isSym(root, root)
 ```
