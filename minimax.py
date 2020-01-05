class Node:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self._value = None
        self.children = None

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

