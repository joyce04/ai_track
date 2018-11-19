class Node:
    def __init__(self, initData):
        self.data = initData
        self.left = None
        self.right = None
    
    def getData(self):
        return self.data
    
    def getLeft(self):
        return self.left
    
    def getRight(self):
        return self.right
    
    def setData(self, newData):
        self.data = newData
    
    def setLeft(self, newLeft):
        self.left = newLeft
    
    def setRight(self, newRight):
        self.right = newRight