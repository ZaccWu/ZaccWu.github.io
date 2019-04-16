---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 19               # 标题 
subtitle:   Partition List #副标题
date:       2019-04-16             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/86.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 86 Partition List
**[Medium]**
Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
You should preserve the original relative order of the nodes in each of the two partitions.

**Example**:
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5

It will be hard to operate on the original linked list, so we may create new linked lists to help us accomplished the task. Here's an interesting idea implemented by C.

C Code
```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */
struct ListNode* partition(struct ListNode* head, int x) {
    struct ListNode left,right;
    struct ListNode *left_cursor,*right_cursor;

    left_cursor = &left;
    right_cursor = &right;

    while(head){
        if(head->val < x){
            left_cursor->next = head;
            left_cursor = left_cursor->next;
        }else{
            right_cursor->next = head;
            right_cursor = right_cursor->next;
        }
        head = head->next;
    }

    right_cursor->next = NULL;
    left_cursor->next = right.next;

    return left.next;

}
```

![](https://github.com/Tinky2013/Tinky2013.github.io/raw/master/img/Leetcode/86.gif)
Python is similar to C in implementing linked list operations. Here's the reference.

Python Code
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        def partition(self, head, x):
            h1 = l1 = ListNode(0)
            h2 = l2 = ListNode(0)
            while head:
                if head.val < x:
                    l1.next = head
                    l1 = l1.next
                else:
                    l2.next = head
                    l2 = l2.next
                head = head.next
            l2.next = None
            l1.next = h2.next
            return h1.next
```
