from numpy import *

def fromSerializableDict(d):
    return DribblingPolicy.fromSerializableDict(d)

class DribblingPolicy:
    POINTS = array([
         [0.25, 0.5, 0],
         [0.75, 0.5, 0],
         [0.75, 0.5, 3.14/2],
         [1.75, 0.5, 3.14/2]])
    #POINTS = array([
    #[0.25, 0.5],
    #[0.75, 0.5],
    #[0.75, 0.5],
    #[1.75, 0.5]])

    IDX = 0
    DIRECTION = 1

    def __init__(self):
        pass



    def getTargetPosition(self, pos, orientation):
        if self.IDX == len(self.POINTS):
            self.IDX -= 1
            self.DIRECTION = -1

        if self.IDX == -1:
            self.IDX = 1
            self.DIRECTION = 1

        rot_err = (orientation-self.POINTS[self.IDX][2] + pi) % (2 * pi) - pi
        if (linalg.norm(self.POINTS[self.IDX][0:2] - pos) < 0.1) & (abs(rot_err) < 0.1):
            self.IDX += self.DIRECTION

        return matrix(self.POINTS[min(max(self.IDX, 0), len(self.POINTS) - 1)])

    def getSerializableDict(self):
        return {}

    @staticmethod
    def fromSerializableDict(d):
        return DribblingPolicy()
