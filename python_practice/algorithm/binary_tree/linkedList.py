from node import Node


class LinkedList:
    def __init__(self):
        self.root = None

    def isEmpty(self):
        return self.root is None

    def getRoot(self):
        return self.root.getData()

    def insert(self, item):
        temp = None
        x = self.root

        while x is not None:
            temp = x
            if item.getData() < x.getData():
                x = x.getLeft()
            else:
                x = x.getRight()

        item_parent = temp
        if temp is None:
            self.root = item
        elif item.getData() < x.getData():
            temp.left = item
        else:
            temp.right = item
