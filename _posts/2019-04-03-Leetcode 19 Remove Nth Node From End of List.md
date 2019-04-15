---
layout:     post                    # 使用的布局（不需要改）
title:      Leetcode 19               # 标题 
subtitle:   Remove Nth Node From End of List #副标题
date:       2019-04-03             # 时间
author:     WZY                      # 作者
header-img: img/Leetcode/19.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Leetcode
---
# 19 Remove Nth Node From End of List
**[Medium]**
Given a linked list, remove the n-th node from the end of list and return its head.

**Example**:
Given linked list: 1->2->3->4->5, and n = 2.
After removing the second node from the end, the linked list becomes 1->2->3->5.

Note:
Given n will always be valid.
Follow up:
Could you do this in one pass?

C Code
```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */
struct ListNode* removeNthFromEnd(struct ListNode* head, int n) {
    struct ListNode* first=head;
    struct ListNode* second=head;
    while(first!=NULL){
        first=first->next;
        if(n--<0){
            second=second->next;
        }
    }
    if(n==0) head=head->next;
    else second->next=second->next->next;
    return head;
}
```
![](E:\Tinky\大学课件及作业\6 自学课\6-8.Leetcode\19\19.gif)

Python Code
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head
```
