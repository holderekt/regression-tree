import utils as utl
import error_measures as err

# Regression Tree Node
class Node:
    def __init__(self, parent, node_id, index=None, value=None, examples=None, prediction=0):
        self.index = index
        self.id = node_id
        self.prediction = prediction
        self.value = value
        self.parent = parent
        self.examples = examples
        self.right = None
        self.left = None 
        self.ssr = 0
        self.leaves = 0
        self.ssr_as_root = 0
    
    def is_leaf(self):
        if(self.right == None and self.left == None):
            return True
        return False
   
    def leafs_id(self):
        if(not self.is_leaf()):
            return self._leafs_search(self.left) + self._leafs_search(self.right)
        return [1]
    
    def n_leafs(self):
        return len(self.leafs_id())

    def _leafs_search(self, node):
        if node.is_leaf():
            return [node.id]
        return self._leafs_search(node.left) + self._leafs_search(node.right)

    def __str__(self):
        return str(self.id)


# Regression Tree
class Regression_Tree:
    def __init__(self, y_train, root):
        self.y = y_train
        self.root = root

    # Generate Prediction given a test example
    def predict(self, example, deleted=[]):
        current_node = self.root
        while(not current_node.is_leaf() and ((current_node in deleted) == False)):
            if(example[current_node.index] <= current_node.value):
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.prediction

    # Generate Sum Square Residuals of a given node on training data
    def node_ssr(self, node):
        ssr = 0
        for example in node.examples:
            ssr = ssr + pow((self.y[example] - node.prediction) , 2)
        return ssr

    def leafs_id(self):
        return self.root.leafs_id()
    
    def n_leafs(self):
        return len(self.leafs_id())

    def __str__(self):
        return self._print(self.root)
    
    def print_leaf(self, node):
        if(node.is_leaf()):
            print(len(node.examples))
        else:
            self.print_leaf(node.left)
            self.print_leaf(node.right)

    def _print(self, node):
        node_id = str(node.id)
        r_string = node_id + " " + str(node.ssr)
        if(not node.is_leaf()):
            r_string = r_string + "\nLeft : " + node_id + "\n" + self._print(node.left)
            r_string = r_string + "\nRight: " + node_id + "\n" + self._print(node.right)
        return r_string
