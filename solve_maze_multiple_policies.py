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

    def loadPolicies(self, filename_straight, filename_turn_left, filename_turn_right):
        self.objPolicyStraight = self.loadPolicy(filename_straight)
        self.objPolicyTurnLeft= self.loadPolicy(filename_turn_left)
        self.objPolicyTurnRight = self.loadPolicy(filename_turn_right)

    def loadPolicy(self, filename):
        with open(filename, 'rb') as f:
            s = f.read()
        d = pickle.loads(s)
        objPolicy = SparseGPPolicy.fromSerializableDict(d)
        return objPolicy

    def solve(self, straight_movement_policy_filename, turn_left_movement_policy_filename, turn_right_movement_policy_filename, numKilobots = 4,
              objectShape = 'circle', stepsPerSec = 8):
        self.loadPolicies(straight_movement_policy_filename, turn_left_movement_policy_filename, turn_right_movement_policy_filename)

        mazePolicy = MazePolicy()

        msg = {'message': 'testMaze',
               'objPolicyStraightDict': self.objPolicyStraight.getSerializableDict(),
               'objPolicyTurnLeftDict': self.objPolicyTurnLeft.getSerializableDict(),
               'objPolicyTurnRightDict': self.objPolicyTurnRight.getSerializableDict(),
               'mazePolicyDict': mazePolicy.getSerializableDict(),
               'numKilobots': numKilobots,
               'objectShape': objectShape,
               'stepsPerSec': stepsPerSec}
        self.socket.send(pickle.dumps(msg, protocol=2))

mazeTester = MazeTester(2358)
mazeTester.connect()

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print(('usage: {} <policy file> ' +
               '<# kilobots> <object shape (quad|circle)>').format(sys.argv[0]))
    else:
        mazeTester.solve(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
