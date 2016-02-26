from numpy import *

def fromSerializableDict(d):
    return MazePolicy.fromSerializableDict(d)

class MazePolicy:
    POINTS = array([
        [1.25, 0.75],
        [1.75, 0.75],
        [1.75, 0.25],
        [0.75, 0.25],
        [0.75, 0.75],
        [0.25, 0.75],
        [0.25, 0.25]])
    IDX = 0
    DIRECTION = 1

    def __init__(self):
        pass

    def getTargetPosition(self, pos):
        if self.IDX == len(self.POINTS):
            self.IDX -= 1
            self.DIRECTION = -1

        if self.IDX == -1:
            self.IDX = 1
            self.DIRECTION = 1

        if linalg.norm(self.POINTS[self.IDX] - pos) < 0.1:
            self.IDX += self.DIRECTION

        return matrix(self.POINTS[min(max(self.IDX, 0), len(self.POINTS) - 1)])

    def getSerializableDict(self):
        return {}

    @staticmethod
    def fromSerializableDict(d):
        return MazePolicy()
