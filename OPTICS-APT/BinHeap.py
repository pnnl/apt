"""
Implementation of a binary heap, with the minimum value at root (a min heap).
Created during Coursera Algorithm course.

Author: Jing Wang
"""

class BinHeap:
    def __init__(self):
        self.heap = []
        self.heap_size = 0
        self.pos = dict()

    @staticmethod
    def parent(idx):
        return (idx-1)//2

    @staticmethod
    def children(idx):
        return 2*idx+1, 2*idx+2     


    def build_heap(self, items):
        """
        Build a binary min heap based on a list alist.

        item is a list of format [(key, val), ...]
        """
        self.heap = items
        self.heap_size = len(items)            
        last_parent = self.parent(self.heap_size-1)
        for idx in range(last_parent, -1, -1):
            self.bubbling_down(idx)
        
        for idx in range(self.heap_size):
            self.pos[self.heap[idx][1]] = idx

        return

    def swap_up(self, i):
        """
        Swap ith element up one position.
        """
        # It only checks the immediate parent and if immediate parent is 
        # correct, function returns. If not, it will continue move element up.
        p = self.parent(i)
        while p >= 0:
            if self.heap[p][0] > self.heap[i][0]:

                # switch index in pos first
                self.pos[self.heap[p][1]] = i
                self.pos[self.heap[i][1]] = p

                # then switch thing stored in heap
                tmp = self.heap[p]
                self.heap[p] = self.heap[i]
                self.heap[i] = tmp

                i = p
                p = self.parent(i)
            else:
                return
            
        return

    def min_child(self, i):
        """
        Find and return the min child of ith element.
        """
        c1, c2 = self.children(i)
        min_child = None
        if c1 < self.heap_size:
            if c2 < self.heap_size:
                if self.heap[c1][0] > self.heap[c2][0]:
                    min_child = c2
                else:
                    min_child = c1
            else:
                min_child = c1

        return min_child

    def bubbling_down(self, i):
        """
        Bubbling down ith element down one position.
        """
        min_c = self.min_child(i)
        while min_c != None:
            if self.heap[min_c][0] < self.heap[i][0]:

                # switch index in pos first
                self.pos[self.heap[min_c][1]] = i
                self.pos[self.heap[i][1]] = min_c

                # switch values in heap
                tmp = self.heap[min_c]
                self.heap[min_c] = self.heap[i]
                self.heap[i] = tmp

                i = min_c
                min_c = self.min_child(i)
            else:
                return
        return


    def insert(self, k):
        """
        Insert k to the heap. k is a turple (key, val).
        """
        self.heap.append(k)
        self.heap_size += 1
        self.pos[k[1]] = self.heap_size-1
        self.swap_up(self.heap_size-1)
        return


    def extract_element(self, i):
        """
        Delete the ith element (could be from the middle) from the heap.
        """
        key, val = self.heap[i]
        self.pos.pop(val, None)

        if i < self.heap_size - 1 and i >= 0:

            self.heap[i] = self.heap[-1]
            self.heap.pop()
            self.heap_size -= 1    
            self.pos[self.heap[i][1]] = i
            self.bubbling_down(i)

        elif i == self.heap_size - 1 or i == -1:
            self.heap.pop()
            self.heap_size -= 1    
        
        return key, val

    def update_key(self, k):
        """
        update the key of a value. To achieve this, the old (key, val) in heap is first deleted \
        and then a (new_key, val) is inserted.
        """
        idx = self.pos[k[1]]
        self.extract_element(idx)
        self.insert(k)
        
        return

if __name__ == '__main__':
    test_heap = BinHeap()
    items = [(9, 'a'), (8, 'b'), (7, 'c'), (6, 'd'), (5, 'e'), (4, 'f'), (3, 'g'), (2, 'h')]

    test_heap.build_heap(items)
    print('test heap is\n', test_heap.heap, '\n')
    print('test heap pos array is\n', test_heap.pos, '\n')

    test_heap.extract_element(-1)
    print('test extract element\n', )
    print('test heap is\n', test_heap.heap, '\n')
    print('test heap pos array is\n', test_heap.pos, '\n')

    test_heap.insert((0, 'b'))
    print('test insert\n', )
    print('test heap is\n', test_heap.heap, '\n')
    print('test heap pos array is\n', test_heap.pos, '\n')