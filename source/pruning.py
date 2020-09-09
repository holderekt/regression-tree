import numpy as np

# Update tree nodes with information on ssr and leaves for fast pruning
def update_tree_pruning(tree, node):
    if(node.is_leaf()):
        node.ssr = tree.node_ssr(node)
        node.leaves = 1
    else:
        l_ssr, l_leaves = update_tree_pruning(tree, node.left)
        r_ssr, r_leaves = update_tree_pruning(tree, node.right)
        node.ssr = l_ssr + r_ssr
        node.leaves = l_leaves + r_leaves
    
    node.ssr_as_root = tree.node_ssr(node)
    return node.ssr, node.leaves

# Generate the alpha value for weakest link search 
def calculate_alpha(tree, node):
    return (node.ssr_as_root - node.ssr) / (node.leaves - 1)

# Return weakest node id and alpha value of a tree
def weakest_link(tree, pruned_nodes):
    if(tree.root.leaves == 1):
        return tree.root, tree.root.ssr_as_root
    return _weakest_link_search(tree, tree.root, np.inf, None, pruned_nodes)

# Search the weakest link starting from a node
def _weakest_link_search(tree, node, min_alpha, weakest_node, pruned_nodes):
    if(not node.is_leaf() and not (node in pruned_nodes)):
        alpha = calculate_alpha(tree, node)
        if (alpha < min_alpha):
            min_alpha = alpha
            weakest_node = node
        weakest_node, min_alpha = _weakest_link_search(tree, node.left, min_alpha, weakest_node, pruned_nodes)
        weakest_node, min_alpha = _weakest_link_search(tree, node.right, min_alpha, weakest_node, pruned_nodes)
    return weakest_node, min_alpha

# Update nodes ssr after a successful prune
def prune_ssr_update(tree, node, pruned_nodes):
    data = _get_leaf_ssr(tree, node, pruned_nodes)
    leaves_ssr = sum(data)
    node.ssr = tree.node_ssr(node)
    node.leaves = 1
    current_node = node.parent
    current_node_ssr = 0
    while(current_node != None):
        current_node.ssr = (current_node.ssr - leaves_ssr) + node.ssr
        current_node_ssr = current_node.ssr
        current_node.leaves = current_node.leaves - len(data) + 1
        current_node = current_node.parent
    return current_node_ssr
    
# Get ssr of leafs from a given node
def _get_leaf_ssr(tree, node, pruned_nodes):
    if(node in pruned_nodes):
        return [node.ssr_as_root]

    if(node.is_leaf()):
        return [node.ssr]
    
    return _get_leaf_ssr(tree, node.left, pruned_nodes) + _get_leaf_ssr(tree, node.right, pruned_nodes)

# Weakest link pruning
def weakest_link_pruning(tree):
    update_tree_pruning(tree, tree.root)
    pruned_nodes = [[]]
    cost_parameters = [0]
    subtrees_ssr = []
    index = 0
    print(tree.root.leaves)
    while (tree.root.leaves > 1):
        node, cp = weakest_link(tree, pruned_nodes[index])
        pruned_nodes.append(pruned_nodes[index] + [node])
        ssr = prune_ssr_update(tree, node, pruned_nodes[index])
        subtrees_ssr.append(ssr)
        cost_parameters.append(cp)
        index = index + 1

    pruned_nodes = [set(el) for el in pruned_nodes]
    return pruned_nodes, cost_parameters, subtrees_ssr