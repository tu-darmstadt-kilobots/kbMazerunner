#!/usr/bin/env python3

import signal
import sys

from numpy import array, asarray, zeros, c_, r_, repeat, sqrt, median, \
        newaxis, square, transpose, random, linspace, meshgrid, empty, asmatrix
from numpy.linalg import norm

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

    def _updateRBFParameters(self):
        MuSA = Helper.getRepresentativeRows(c_[self.S, self.A], 500)

        MuS = Helper.getRepresentativeRows(self.S, 500)

        bwSA = Helper.getBandwidth(MuSA, 500, 2.5)
        bwS = Helper.getBandwidth(MuS, 500, 1.5)

        self.rbf = RBFFeatureFunction(MuSA, bwSA, MuS, bwS)

    def connect(self):
        self.socket.bind('tcp://*:{}s'.format(self.port))

        self._sendPolicyModules()

    def learn(self):
        # LSTD
        self.lstd.discountFactor = 0.9

        # GP
        self.policy.GPRegularizer = 1e-8

        # REPS
        self.reps.epsilonAction = 1.0

        self.S, self.A, self.R, self.S_ = empty((0, 2)), empty((0, 2)),\
                empty((0, 1)), empty((0, 2))

        it = 0
        while True:
            # get new samples
            St, At, Rt, S_t = self._getSamples()
            print('sum reward for last samples: {}'.format(Rt.sum()))

            # add samples
            self.S = r_[self.S, St]
            self.A = r_[self.A, At]
            self.R = r_[self.R, Rt]
            self.S_ = r_[self.S_, S_t]

            self._updateRBFParameters()

            self.PHI_SA = self.rbf.getStateActionFeatureMatrix(self.S, self.A)
            self.PHI_S = self.rbf.getStateFeatureMatrix(self.S)

            # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
            self.PHI_SA_ = Helper.getFeatureExpectation(self.S_, 5,
                    self.policy, self.rbf)
            self.theta = self.lstd.learnLSTD(self.PHI_SA, self.PHI_SA_, self.R)

            # AC-REPS
            self.Q = self.PHI_SA * self.theta
            self.w = self.reps.computeWeighting(self.Q, self.PHI_S)

            # GP
            self.policy.train(self.S, self.A, self.w)
            it += 1

            # plot current policy based on best (mean) action
            self.plotPolicyState2D(50, 25, it)

    def plotPolicyState2D(self, stepsX, stepsY, it):
        [X, Y] = meshgrid(linspace(0.0, 2.0, stepsX), linspace(0.0, 1.0, stepsY))
        X = X.flatten()
        Y = Y.flatten()
        A = asarray(self.policy.getMeanAction(c_[X, Y]))
        U = A[:, 0]
        V = A[:, 1]

        plt.quiver(X, Y, U, V)
        plt.title('iteration: {}'.format(it))
        plt.show()

learner = LightMovementLearner(2357)
learner.connect()
