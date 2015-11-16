#!/usr/bin/env python3

from numpy import array, asarray, zeros, c_, r_, repeat, sqrt, median, \
        newaxis, square, transpose, random, linspace, meshgrid

import matplotlib.pyplot as plt

from zmq import Context, PAIR
import pickle

import sys
sys.path.append('..')

from Helper import Helper
from LeastSquaresTD import LeastSquaresTD
from FeatureFunction import RBFFeatureFunction
from AC_REPS import AC_REPS
from Kernel import ExponentialQuadraticKernel
from SparseGPPolicy import SparseGPPolicy


class LightMovementLearner:
    def __init__(self, port):
        self.context = Context()
        self.socket = self.context.socket(PAIR)
        self.port = port

        kernel = ExponentialQuadraticKernel(2)
        # initial random range for actions
        aRange = array([[-1e-1, 1e-1], [-1e-1, 1e-1]])
        self.policy = SparseGPPolicy(kernel, aRange)

        self.lstd = LeastSquaresTD()
        self.reps = AC_REPS()

    def _sendPolicyModules(self):
        msg = {'message': 'sentPolicyModules',
               'modules': [
                   ('Helper.py', open('../Helper.py').read()),
                   ('Kernel.py', open('../Kernel.py').read()),
                   ('SparseGPPolicy.py', open('../SparseGPPolicy.py').read())
                   ],
               'policyModule': 'SparseGPPolicy'}
        self.socket.send(pickle.dumps(msg, protocol=2))

    def _getSamples(self):
        msg = {'message': 'getSamples',
               'policyDict': self.policy.getSerializableDict()}
        self.socket.send(pickle.dumps(msg, protocol=2))

        msg = pickle.loads(self.socket.recv(), encoding='latin1')
        if not msg['message'] == 'sentSamples':
            print('received unexpected message')
        else:
            return msg['samples']

    def _updateParameters(self, S, A):
        MuSA = Helper.getRepresentativeRows(c_[S, A], 500)

        MuS = Helper.getRepresentativeRows(S, 500)

        bwSA = Helper.getBandwidth(MuSA, 500, 2.5)
        bwS = Helper.getBandwidth(MuS, 500, 1.5)

        self.rbf = RBFFeatureFunction(MuSA, bwSA, MuS, bwS)

    def connect(self):
        self.socket.bind('tcp://*:{}s'.format(self.port))

        self._sendPolicyModules()

    def learn(self):
        S, A, R, S_ = self._getSamples()
        print('sum reward: {}'.format(R.sum()))

        self._updateParameters(S, A)

        for i in range(20):
            # DEBUG
            self.plotPolicyState2D(50, 25)

            PHI_SA = self.rbf.getStateActionFeatureMatrix(S, A)
            PHI_S = self.rbf.getStateFeatureMatrix(S)

            # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
            PHI_SA_ = Helper.getFeatureExpectation(S_, 20, self.policy, self.rbf)
            theta = self.lstd.learnLSTD(PHI_SA, PHI_SA_, R)

            # AC-REPS
            Q = PHI_SA * theta
            w = self.reps.computeWeighting(Q, PHI_S)

            # GP
            self.policy.train(S, A, w)

            St, At, Rt, S_t = self._getSamples()
            print('sum reward: {}'.format(Rt.sum()))

            S = r_[S, St]
            A = r_[A, At]
            R = r_[R, Rt]
            S_ = r_[S_, S_t]

            self._updateParameters(S, A)

    def plotPolicyState2D(self, stepsX, stepsY):
        [X, Y] = meshgrid(linspace(0.0, 2.0, stepsX), linspace(0.0, 1.0, stepsY))
        X = X.flatten()
        Y = Y.flatten()
        A = asarray(self.policy.evaluate(c_[X, Y]))
        U = A[:, 0]
        V = A[:, 1]

        plt.quiver(X, Y, U, V)
        plt.show()

if __name__ == '__main__':
    learner = LightMovementLearner(2357)
    learner.connect()
    learner.learn()
