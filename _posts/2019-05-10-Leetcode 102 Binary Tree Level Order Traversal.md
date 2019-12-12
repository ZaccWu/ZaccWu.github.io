---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 102               # 标题 
subtitle:   Binary Tree Level Order Traversal #副标题
date:       2019-05-10             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/102.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
--- 

# 102 Binary Tree Level Order Traversal
**[Medium]**
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],return its level order traversal as:[[3],[9,20],[15,7]]

***

My C code may be a little complex, the main part of the code is 'getLevelValue' function, we visit the node level by level recursively in the 'getLevelValue' function.

To make the process more clear, I used 'TreeDepth' function to calculate the depth of the tree. 'ReturnSize' is equal to depth of the tree.

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

 * Return an array of arrays of size *returnSize.
 
 * The sizes of the arrays are returned as *returnColumnSizes array.
 
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 
 */

/* First we need to figure out the depth of the Binary Tree */

int TreeDepth(struct TreeNode* root){
    if(!root) return 0;
    int leftDepth=TreeDepth(root->left)+1;
    int rightDepth=TreeDepth(root->right)+1;
    return leftDepth>rightDepth?leftDepth:rightDepth;
}

void getLevelValue(int ** levelArray, struct TreeNode* root, int* columnSizes, int level){
    if(root==NULL) return ;
    
    columnSizes[level]++;
    levelArray[level] = (int *)realloc(levelArray[level], columnSizes[level]*sizeof(int));
    levelArray[level][columnSizes[level] - 1] = root->val;
        
    getLevelValue(levelArray, root->left, columnSizes, level+1);
    getLevelValue(levelArray, root->right, columnSizes, level+1);
}

int** levelOrder(struct TreeNode* root, int* returnSize, int** ColumnSizes){
    if(returnSize == NULL)
        return NULL;
    if(root == NULL){
        *returnSize = 0;
        return NULL;
    }
    *returnSize = TreeDepth(root);
    int ** levelArray = (int **)calloc(*returnSize, sizeof(int*));
    *ColumnSizes = (int *)calloc(*returnSize, sizeof(int));
    
    getLevelValue(levelArray, root, *ColumnSizes, 0);
    
    return levelArray;
}
```

The Python code may be more concise, it also uses the recursive idea. We can also use Queue to solve this problem.

Python Code
```python
# Definition for a binary tree node.

# class TreeNode(object):

#     def __init__(self, x):

#         self.val = x

#         self.left = None

#         self.right = None

#recursive solution

class Solution:
    def _levelOrder(self, level, result, node):
        if node:
            if level == len(result):
                result.append([])
                
            result[level].append(node.val)
            self._levelOrder(level+1, result, node.left)
            self._levelOrder(level+1, result, node.right)

    def levelOrder(self, root):
        """
        
        :type root: TreeNode
        
        :rtype: List[List[int]]
        
        """
        level, result = 0, list()
        self._levelOrder(level, result, root)
        return result 
```
