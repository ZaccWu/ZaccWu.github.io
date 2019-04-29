---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 144               # 标题 
subtitle:   Binary Tree Preorder Traversal #副标题
date:       2018-02-14             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/144.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
--- 
# 144 Binary Tree Preorder Traversal
**[Medium]**
Given a binary tree, return the preorder traversal of its nodes' values.

***

The C implementation of the problem is as follow, here I use stack to store the nodes temperarily.

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
 * Return an array of size *returnSize.
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* preorderTraversal(struct TreeNode* root, int* returnSize) {
   
    int *result;
    int top, count;
    struct TreeNode **stack;    /* stack points to the node */
    struct TreeNode  *current;  /* current points to the node's value */
    
    result = malloc(100 * sizeof(int));
    stack  = malloc(100 * sizeof(struct TreeNode *));
    top    = 0;
    count  = 0;
    current = root;
    
    while (top > 0 || current != NULL) {  
        while (current != NULL) {
            result[count++] = current->val;
            stack[top++]    = current;
            current = current->left;
        }
        
        current = stack[--top]; 
        /* Here we have arrived the leaf node, so the next step is to go back and search the right one */
        /* The feature of stack will help us to reach this goal easily */
        if (current->right == NULL) {
           current = NULL;
        } else {
           current = current->right;
        }
    }
    
    *returnSize = count;
    return (result);
}
```

The Python code may be easiler to understand, the recursive solution shows the operation sequence when we use preorder traversal.
The iterative solution also uses stack's idea. In every subtree, we first pop the root node, than we add the rightchild and leftchild to the stack.

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
    def recur(self, root, res):
        if root:
            res.append(root.val)
            self.recur(root.left,res)
            self.recur(root.right,res)
        
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res=[]
        self.recur(root,res)
        return res
        
    # iteratively
    def preorderTraversal(self, root):
        stack, res = [root], []
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.append(node.right) # first in last out, so we first append 'right'
                stack.append(node.left)
        return res
```

