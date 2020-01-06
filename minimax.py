class Node:
    def __init__(self, is_leaf=False, val=None):
        self.is_leaf = is_leaf
        self._value = None
        self.children = None
        self.value = val

    def set_children(self, *children):
        self.children = []
        for child in children:
            self.children.append(child)

    @property.setter
    def value(self, val):
        self._value = val

    @value.getter
    def value(self):
        return self._value


def create_minimax_game():
    root = Node()
    left = Node()
    right = Node()

    root.set_children(left, right)
    left.set_children(Node(val=3), Node(val=5))
    right.set_children(Node(val=3), Node(val=5))

    return root


def minimax(node, is_player1):
    if node.children[0].value is not None:
        if is_player1:
            return max(map(lambda som_node: som_node.value, node.children))
        else:
            return min(map(lambda som_node: som_node.value, node.children))
    else:
        if is_player1:
            return max(map(lambda som_node: minimax(som_node, not is_player1)))
        else:
            return min(map(lambda som_node: minimax(som_node, not is_player1)))


