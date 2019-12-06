---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 110               # 标题 
subtitle:   Balanced Binary Tree #副标题
date:       2018-07-04             # 时间
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
        
