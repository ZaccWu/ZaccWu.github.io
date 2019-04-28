---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 94               # 标题 
subtitle:   Binary Tree Inorder Traversal #副标题
date:       2018-01-29             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/94.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
--- 
# 94 Binary Tree Inorder Traversal
**[Medium]**
Given a binary tree, return the inorder traversal of its nodes' values.

***

This C solution is really fundamental and classical. Using the idea of stack can somehow be more comprehensible.

It's not intuitively obvious what the code does, so I will try to add more detailed explation after every line of the code.

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


/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
void recur(struct TreeNode* root, int* count){
    if (root == NULL) return;
    recur(root->left, count);   # For every node, count the number of nodes in its left subtree.
    *count += 1;                # Count itself.
    recur(root->right, count);  # Count the number of nodes in its right subtree.
}

int* inorderTraversal(struct TreeNode* root, int* returnSize){    
    int count = 0;
    int top = -1;
    int i = 0;
    
    recur(root, &count);    # We need this recur function to count how many nodes we have in the tree.
    int* out = (int *)malloc(sizeof(int)*(count));
    int** stack = (int *)malloc(sizeof(int*)*(count));
    
    while(root != NULL || top >= 0){
        while(root != NULL){
            top++;
            stack[top] = root;
            root = root->left;  # First we need to add the left part.
        }
        root = stack[top];
        out[i] = root->val; i++;
        top--;
        root = root->right;    # If the node don't have rightchild, return to the PARENT NODE of that node.
    }
    
    *returnSize = count;
    return out;
}
```

It is really convenient if we use Python, the 'append' function will undoubtedly be our first choice to solve the problem.
We have two ways to achieve it: recursively and iteratively.

Python Code
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    # recursively
    def inorderTraversal_1(self, root):
        res = []
        self.helper(root, res)
        return res

    def helper(self, root, res):
        if root:      # The next three lines are the core of Inorder Traversal
            self.helper(root.left, res)
            res.append(root.val)
            self.helper(root.right, res)

    # iteratively 
    # the idea of this way is similar with the implementation using C above.
    def inorderTraversal_2(self, root):
        res, stack = [], []
        while True:
            while root:
                stack.append(root)
                root = root.left
            if not stack:
                return res
            node = stack.pop()
            res.append(node.val)
            root = node.right
```

