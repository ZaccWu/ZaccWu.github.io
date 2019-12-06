---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 111               # 标题 
subtitle:   Minimum Depth of Binary Tree #副标题
date:       2018-11-04             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/111.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 111 Minimum Depth of Binary Tree
**[Easy]**
Given a binary tree, find its minimum depth.
The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

***

The question is really similar with the one that find a binary tree's maximum depth in Leetcode 104.
The first idea is to follow the thought of that question, which is reflected in the returning value of the function, in every recursion level we will return the smaller one.

There's also one important thing that we need to consider, if the tree's leftchild node is NULL, we need to return the minimum depth of its right part and plus 1(the root). The rightchild be the same.

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


int minDepth(struct TreeNode* root){
	if(NULL==root) return 0;
    
	int LeftDepth=minDepth(root->left);
	int RightDepth=minDepth(root->right);
    if(root->left!=0 && root->right!=0){
        return LeftDepth<RightDepth? LeftDepth+1 : RightDepth+1;
    }
	return LeftDepth>RightDepth? LeftDepth+1 : RightDepth+1;/* One branch is NULL */
}
```

Here's solution using Python, DFS and BFS can both solve the question.

In BFS solution, 'deque' module will be really power. 'deque' means double-ended-queue, and it can insert or delete elements on both sides.

Python Code
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
#DFS
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if None in [root.left, root.right]:
            return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
#BFS      
    def minDepth1(self, root):
        if not root:
            return 0
        queue = collections.deque([(root, 1)])
        while queue:
            node, level = queue.popleft() #pop the left element in the list
            if node:
                if not node.left and not node.right:
                    return level
                else:
                    queue.append((node.left, level+1))
                    queue.append((node.right, level+1))
```
