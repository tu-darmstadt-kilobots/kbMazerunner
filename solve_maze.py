#!/usr/bin/python3

"""
    Pushes an object through a maze using an object movement policy
    and a maze policy.
"""

from numpy import *

from zmq import Context, PAIR
import pickle
import sys

from SparseGPPolicy import SparseGPPolicy
from MazePolicy import MazePolicy


class MazeTester:
    def __init__(self, port):
        self.context = Context()
        self.socket = self.context.socket(PAIR)
        self.port = port

    def _sendPolicyModules(self):
        msg = {'message': 'sentPolicyModules',
               'modules': [
                   ('Helper.py', open('Helper.py').read()),
                   ('Kernel.py', open('Kernel.py').read()),
                   ('SparseGPPolicy.py', open('SparseGPPolicy.py').read()),
                   ('MazePolicy.py', open('MazePolicy.py').read())
                   ],
               'objPolicyModule': 'SparseGPPolicy',
               'mazePolicyModule': 'MazePolicy'}
        self.socket.send(pickle.dumps(msg, protocol=2))

    def _reshapeKbPositions(self, X):
        return c_[X.flat[0::2].T, X.flat[1::2].T]

    def connect(self):
        self.socket.bind('tcp://*:{}s'.format(self.port))

        self._sendPolicyModules()

    def loadPolicy(self, fileName):
        with open(fileName, 'rb') as f:
            s = f.read()
        d = pickle.loads(s)

        self.objPolicy = SparseGPPolicy.fromSerializableDict(d)

    def solve(self, movementPolicyFileName, numKilobots = 4,
            objectShape = 'circle', stepsPerSec = 8):
        self.loadPolicy(movementPolicyFileName)

        mazePolicy = MazePolicy()

        msg = {'message': 'testMaze',
               'objPolicyDict': self.objPolicy.getSerializableDict(),
               'mazePolicyDict': mazePolicy.getSerializableDict(),
               'numKilobots': numKilobots,
               'objectShape': objectShape,
               'stepsPerSec': stepsPerSec}
        self.socket.send(pickle.dumps(msg, protocol=2))

mazeTester = MazeTester(2358)
mazeTester.connect()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(('usage: {} <policy file> ' +
               '<# kilobots> <object shape (quad|circle)>').format(sys.argv[0]))
    else:
        while True:
            mazeTester.solve(sys.argv[1], int(sys.argv[2]), sys.argv[3])
