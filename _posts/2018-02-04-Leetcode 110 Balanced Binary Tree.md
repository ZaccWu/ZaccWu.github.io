---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 110               # 标题 
subtitle:   Balanced Binary Tree #副标题
date:       2018-02-04             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/110.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
--- 
# 110 Balanced Binary Tree
**[Easy]**
Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:

> a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

***

This question needs a little technique, my first idea is declaring a variable called 'diff', which will record the gap of the left subtree and the right subtree.
If the gap is larger than 1, than the function will return -1.

C Code
```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */

#define MAX(a,b) ((a>b)?a:b)
int Balanced(struct TreeNode* root) {
    int left, right, diff;
    if (root==NULL) return 0;
    
    right = Balanced(root->right);
    left = Balanced(root->left);
    if (right == -1 || left == -1) return -1; /* Not balanced! */
        
    diff = (right > left) ? right - left : left -right;
    if (diff > 1) return -1;
    else return 1 + MAX(left, right); /* In every recursion the level will plus 1 */
        
}

bool isBalanced(struct TreeNode* root) {
    return (Balanced(root) != -1);
}
```

Here's the Python implementation, we use self.height() to complete the recursion('self' means Instances of the class in Python)

Python Code
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        height = self.height(root)
        return height != -1
    
    def height(self, root):
        if not root:
            return 0
        left_depth = self.height(root.left)
        if left_depth == -1:
            return -1
        right_depth = self.height(root.right)
        if right_depth == -1:
            return -1
        if abs(left_depth - right_depth) > 1:
            return -1
        else:
            return max(left_depth, right_depth) + 1
```

