""" 
A TreeNode class that supports hierachical clustering.
Each node represents a cluster at current hierachical level.
"""

import numpy as np
import heapdict

__version__ = '0.2'

class TreeNode:
    """
    A data class that represents hierarchy in hierachical clustering using a tree representation.

    A node can have many children but only one parent.
    """
    def __init__(self, start, end, local_maxima):
        """
        start: 
            start index of the current node, note that start is inclusive and end is exclusive, [start, end)
        end:
            end index of the current node, note that start is inclusive and end is exclusive, [start, end)
        LM:
            local maximum list, a priority queue (using heapdict)
        """
        self.index_range = (start, end)
        self.size = end - start
        self.local_maxima = local_maxima
        self.children = []
        self.parent = None
        self.split_point = None

    def average_RD(self, ordered_RD):
        return np.mean(ordered_RD[self.index_range[0]:self.index_range[1]])

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def remove_child(self, child_node):
        """
        Remove child from current node. All the entire subtree started at child node
        will be lost after deletion.
        It the child node is not a child of current node, do nothing.
        """
        try:
            child_node.parent = None
            self.children.remove(child_node)
        except ValueError:
            pass

    def is_root(self):
        return self.parent == None

    def is_leaf(self):
        return len(self.children) == 0

    def next_lm(self):
        try:
            return self.local_maxima.popitem()
        except IndexError: # index error for popitem in heapdict. Not key error.
            return None

    def is_lm_empty(self):
        return len(self.local_maxima) == 0    

def merge(node_1, node_2):
    """
    Merge two nodes. The merged node will be a child of the common parent, and will \
    inherit children from both nodes.
    
    Only children from the same parent can be mergered. And these two nodes has \
    to be neighboer in terms of index.

    That is e.g. if node_1 index range is (0, 10), and node_2 (20, 30), merge will not proceed.
    Note will be proceed if node_2 is (10, 20).

    node_1, node_2:
        the node to be merged with.

    return:
        the new node.
    """
    #check if has the same parent
    if node_1.parent is node_2.parent:
        #check neighbor condition
        start_1, end_1 = node_1.index_range
        start_2, end_2 = node_2.index_range

        if end_1 == start_2:
            new_start = start_1
            new_end = end_2
        elif end_2 == start_1:
            new_start = start_2
            new_end = end_1
        else:
            print('merge failed, two node are not neighbors')
            return

        new_lm = heapdict.heapdict()
        new_lm.update(node_1.local_maxima)
        new_lm.update(node_2.local_maxima)

        new_children = node_1.children + node_2.children

        new_node = TreeNode(new_start, new_end, new_lm)
        node_1.parent.add_child(new_node)
        for child in new_children:
            new_node.add_child(child)
        node_1.parent.remove_child(node_2)
        node_1.parent.remove_child(node_1)
        
        return new_node
    
    else:
        print('merge failed, two nodes are not from same parent')
        return

def divide(node, split_point, is_similar=False):
    """
    Divide the current node into two nodes at the split point. \
    There are two ways to added these two nodes:
        1. If the RD of split point for current node is similar \
        to the RD of split point of its parent, the two new nodes \
        will be the children of the parent node, current node will be \
        deleted.
        2. if not similar, then two new nodes will be added as children \
        to current node.

    The node been divided must not have children.

    split_point:
        an index at which the current node will be splited.
    is_similar:
        flag to regulate how to add the two new nodes.

    Return:
        new_left_node, new_right node,\
        or \
        None, None
    """
    if node.is_leaf():        #check if has children
        start, end = node.index_range
        local_maxima_left = heapdict.heapdict() # LM for left node
        local_maxima_right = heapdict.heapdict() # LM for right node
        while not node.is_lm_empty():
            key, val = node.next_lm()
            if key < split_point:
                local_maxima_left[key] = val
            elif key > split_point:
                local_maxima_right[key] = val
        new_node_left = TreeNode(start, split_point, local_maxima_left)
        new_node_right = TreeNode(split_point, end, local_maxima_right)

        if (not is_similar) or (node.is_root()): # is not similar or is root node.
            node.add_child(new_node_left)
            node.add_child(new_node_right)
        else: # is similar and node is not root.
            node.parent.add_child(new_node_left)
            node.parent.add_child(new_node_right)
            node.parent.remove_child(node)
        return new_node_left, new_node_right
    else:
        print('node is not leaf, can not divide')
        return None, None

def retrieve_leaf_nodes(node, leaf_list):
    """
    A recursive algorithm to extract leaf nodes.
    """
    if node is not None:
        if node.is_leaf():
            leaf_list.append(node)
        for item in node.children:
            retrieve_leaf_nodes(item, leaf_list)
    return leaf_list

def extract_tree(root, ordered_RD):
    """
    Extract all nodes from the tree below root node using deepth-first search.

    reture:
        a list has the structure [[node_id, start_idx, end_idx, average_RD, level_id, parent_id],\
                                  [...]...]
    """
    level = 0
    node_id = 0
    parent_id = [-1]
    to_be_processed = [root]
    next_level = []
    next_level_parent_id = []
    tree = []
    while len(to_be_processed) > 0:
        current_node = to_be_processed.pop()
        next_level.extend(current_node.children)

        # keep track of parent for updating parent id
        current_parent_id = parent_id.pop()
        next_level_parent_id.extend([node_id]*len(current_node.children))

        start, end = current_node.index_range
        ave_RD = current_node.average_RD(ordered_RD)
        tree.append([node_id, start, end, ave_RD, level, current_parent_id])
        node_id += 1
        if len(to_be_processed) == 0: # unless current_children_list is also empty, current notde is refilled and loop continue
            to_be_processed = next_level
            parent_id = next_level_parent_id
            next_level_parent_id = []
            next_level = []
            level += 1
    return tree

def tree_to_file(tree, fname):
    """
    Write the extracted tree to txt file.
    """
    header = 'node_id start end ave_RD level parent_id\n'
    with open(fname, 'w') as out_f:
        out_f.write(header)
        for lst in tree:
            line = ' '.join(str(el) for el in lst) + '\n'
            out_f.write(line)

####################################
# Test
####################################
if __name__ == '__main__':
    # Test TreeNode
    RD = np.random.random(10)

    start_1 = 0
    end_1 = 6
    LM_1 = heapdict.heapdict(zip(range(start_1, end_1), RD[start_1:end_1]))    

    start_1_1 = 0
    end_1_1 = 3
    LM_1_1 = heapdict.heapdict(zip(range(start_1_1, end_1_1), RD[start_1_1:end_1_1]))    

    start_1_2 = 3
    end_1_2 = 6
    LM_1_2 = heapdict.heapdict(zip(range(start_1_2, end_1_2), RD[start_1_2:end_1_2])) 

    hr_node_1 = TreeNode(start_1, end_1, LM_1)
    hr_node_1_1 = TreeNode(start_1_1, end_1_1, LM_1_1)
    hr_node_1_2 = TreeNode(start_1_2, end_1_2, LM_1_2)
    
    start_2 = 6
    end_2 = 10
    LM_2 = heapdict.heapdict(zip(range(start_2, end_2), RD[start_2:end_2]))

    start_2_1 = 6
    end_2_1 = 8
    LM_2_1 = heapdict.heapdict(zip(range(start_2_1, end_2_1), RD[start_2_1:end_2_1]))
    
    hr_node_2 = TreeNode(start_2, end_2, LM_2)
    hr_node_2_1 = TreeNode(start_2_1, end_2_1, LM_2_1)

    hr_node_root = TreeNode(start_1, end_2, heapdict.heapdict(zip(range(start_1, end_2), RD)))
    hr_node_root.add_child(hr_node_1)
    hr_node_root.add_child(hr_node_2)
    hr_node_1.add_child(hr_node_1_1)
    hr_node_1.add_child(hr_node_1_2)
    hr_node_2.add_child(hr_node_2_1)
    # test add child
    print('Test add child, children of roots should be (2), actual are ', len(hr_node_root.children))
    # test remove child
    hr_node_2_1.parent.remove_child(hr_node_2_1)
    print('Test remove child, child of hr node 2 should be (0), actual are ', len(hr_node_2.children))

    # test merge
    new_child = merge(hr_node_1, hr_node_2)
    print('test merge, hr node 1 and 2 should be merged (True), is new node a child of root:', new_child == hr_node_root.children[0])
    print('test merge, (Two) grand children will be inhereted by new child, number of new children are: ', len(new_child.children))

    # test divide
    new_left, new_right = divide(hr_node_1_2, 4, False)
    print('test divide. assume new nodes are not similar, should be (2) child of new child:', len(hr_node_1_2.children))
    print('test divide. assume new nodes are not similar, new child parent is hr_node_1_2 (True):', new_left.parent is hr_node_1_2)
    
    new_left, new_right = divide(hr_node_1_1, 1, True)
    print('test divide. assume new nodes are similar, root should has no child hr_node_1_1 (True)', hr_node_1_1 not in new_child.children)
    print('test divide. assume new nodes are similar, new left parent is new_child (True):', new_left.parent is new_child)

    # test extract_tree
    tree = extract_tree(hr_node_root, RD)
    print(tree)

    # test tree_to_file
    fname = 'test_tree_file.txt'
    tree_to_file(tree, fname)