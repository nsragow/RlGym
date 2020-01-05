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

    @property
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
