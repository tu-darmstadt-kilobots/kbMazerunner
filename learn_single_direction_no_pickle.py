"""
    Multiple Kilobots move directly to the light.
    They learn to push an object in a single direction.
    The light is moved based on a policy provided by this learner.
"""

from numpy import *


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Polygon

import os
import gc
import pprint
import time
import sys

from zmq import Context, PAIR
import pickle

from Helper import Helper
from LeastSquaresTD import LeastSquaresTD
from AC_REPS import AC_REPS
from Kernel import ExponentialQuadraticKernel, KilobotKernel
from SparseGPPolicy import SparseGPPolicy
from kbSimulator.simulate_single_direction_no_pickle import KilobotsObjectMazeSimulator


class MazeLearner:
    def __init__(self, use_gui):
        # s: light.x light.y kb.x1 kb.y1 ... kb.xn kb.yn
        # a: light movement (dx, dy)
        self.NUM_NON_KB_DIM = 2

        # initial random range for actions
        self.aRange = array([[-0.015, 0.015], [-0.015, 0.015]])
        self.policy = SparseGPPolicy(KilobotKernel(self.NUM_NON_KB_DIM),
                self.aRange)

        # kernels used for LSTD
        self.kernelS = KilobotKernel(self.NUM_NON_KB_DIM)
        self.kernelSA = KilobotKernel(self.NUM_NON_KB_DIM + 2)

        self.lstd = LeastSquaresTD()
        self.reps = AC_REPS()

        self.it = 0
        self.S = None

        self.simulator = KilobotsObjectMazeSimulator(use_gui)
        self.use_gui = use_gui

    def _getStateActionMatrix(self, S, A):
        # states without kilobot positions + actions + kilobot positions
        return c_[S[:, 0:self.NUM_NON_KB_DIM], A, S[:, self.NUM_NON_KB_DIM:]]

    def _unpackSARS(self, SARS):
        return SARS[:, 0:self.sDim],\
               SARS[:, self.sDim:(self.sDim + 2)],\
               SARS[:, (self.sDim + 2):(self.sDim + 3)],\
               SARS[:, (self.sDim + 3):(2 * self.sDim + 3)]

    def _reshapeKbPositions(self, X):
        return c_[X.flat[0::2].T, X.flat[1::2].T]

    def _updateKernelParameters(self, S, A, random=True, normalize=True):
        SA = self._getStateActionMatrix(S, A)

        if random:
            self.MuS = Helper.getRandomSubset(S, self.numFeatures)
            self.MuSA = Helper.getRandomSubset(SA, self.numFeatures)
        else:
            self.MuS = Helper.getRepresentativeRows(S, self.numFeatures, normalize)
            self.MuSA = Helper.getRepresentativeRows(SA, self.numFeatures, normalize)

        NUM_SAMPLES_FOR_BW_ESTIMATE = 500

        # bandwidth for PHI_S
        bwNonKbS = Helper.getBandwidth(self.MuS[:, 0:self.NUM_NON_KB_DIM],
                NUM_SAMPLES_FOR_BW_ESTIMATE, self.bwFactorNonKbS)

        kbPosS = self._reshapeKbPositions(self.MuS[:, self.NUM_NON_KB_DIM:])
        bwKbS = Helper.getBandwidth(kbPosS, NUM_SAMPLES_FOR_BW_ESTIMATE,
                self.bwFactorKbS)

        self.kernelS.setBandwidth(bwNonKbS, bwKbS)
        self.kernelS.setWeighting(self.weightNonKbS)

        # bandwidth for PHI_SA
        bwNonKbSA = Helper.getBandwidth(self.MuSA[:, 0:(self.NUM_NON_KB_DIM + 2)],
                NUM_SAMPLES_FOR_BW_ESTIMATE, self.bwFactorNonKbSA)

        kbPosSA = self._reshapeKbPositions(self.MuSA[:, (self.NUM_NON_KB_DIM + 2):])
        bwKbSA = Helper.getBandwidth(kbPosSA, NUM_SAMPLES_FOR_BW_ESTIMATE,
                self.bwFactorKbSA)

        self.kernelSA.setBandwidth(bwNonKbSA, bwKbSA)
        self.kernelSA.setWeighting(self.weightNonKbSA)

    def _getSubsetForGP(self, S, random=True, normalize=True):
        Nsubset = min(self.numSamplesSubsetGP, S.shape[0])

        if random:
            return Helper.getRandomSubset(S, Nsubset)
        else:
            return Helper.getRepresentativeRows(S, Nsubset, normalize)

    def _updateBandwidthsGP(self, Ssub):
        bwNonKb = Helper.getBandwidth(Ssub[:, 0:self.NUM_NON_KB_DIM],
                Ssub.shape[0], self.bwFactorNonKbGP)

        kbPos = Ssub[:, self.NUM_NON_KB_DIM:]
        bwKb = Helper.getBandwidth(self._reshapeKbPositions(kbPos),
                Ssub.shape[0], self.bwFactorKbGP)

        self.policy.kernel.setBandwidth(bwNonKb, bwKb)
        self.policy.kernel.setWeighting(self.weightNonKbGP)

    def _getFeatureExpectation(self, S, N, MuSA):
        SA = self._getStateActionMatrix(S, self.policy.sampleActions(S))

        PHI = self.kernelSA.getGramMatrix(SA, MuSA)

        for i in range(N - 1):
            SA = self._getStateActionMatrix(S, self.policy.sampleActions(S))
            PHI += self.kernelSA.getGramMatrix(SA, MuSA)

        return PHI / N

    def savePolicyAndSamples(self, fileName):
        d = self.policy.getSerializableDict()
        d['SARS'] = c_[self.S, self.A, self.R, self.S_]
        d['numIt'] = self.it
        d['sDim'] = self.sDim

        s = pickle.dumps(d)
        with open(fileName, 'wb') as f:
            f.write(s)

    def loadPolicyAndSamples(self, fileName):
        with open(fileName, 'rb') as f:
            s = f.read()
        d = pickle.loads(s)

        self.policy = SparseGPPolicy.fromSerializableDict(d)

        self.it = d['numIt']
        self.sDim = d['sDim']

        SARS = d['SARS']
        self.S, self.A, self.R, self.S_ = self._unpackSARS(SARS)

    def saveParams(self, fileName):
        params = {
            'general': {
                'numLearnIt': self.numLearnIt,
                'startEpsilon': self.startEpsilon,
                'epsilonFactor': self.epsilonFactor},
            'sampling': {
                'objectShape': self.objectShape,
                'numKilobots': self.numKilobots,
                'numEpisodes': self.numEpisodes,
                'numStepsPerEpisode': self.numStepsPerEpisode},
            'LSTD': {
                'discountFactor': self.lstd.discountFactor,
                'numFeatures': self.numFeatures,
                'S': {
                    'bwNonKb': self.bwFactorNonKbS,
                    'bwKb': self.bwFactorKbS,
                    'weightNonKb': self.weightNonKbS},
                'SA': {
                    'bwNonKb': self.bwFactorNonKbSA,
                    'bwKb': self.bwFactorKbSA,
                    'weightNonKb': self.weightNonKbSA}},
            'REPS': {
                'epsilonAction': self.reps.epsilonAction},
            'GP': {
                'minVariance': self.policy.GPMinVariance,
                'regularizer': self.policy.GPRegularizer,
                'numSamplesSubset': self.numSamplesSubsetGP,
                'bwNonKb': self.bwFactorNonKbGP,
                'bwKb': self.bwFactorKbGP,
                'weightNonKb': self.weightNonKbGP}}

        with open(fileName, 'w') as f:
            f.write(pprint.pformat(params, width=1))

    def learn(self, savePrefix, numSampleIt, continueLearning = True,
            numLearnIt = 1, startEpsilon = 0.0, epsilonFactor = 1.0):

        """ sampling """
        self.objectShape = 'quad' #t-form
        self.numKilobots = 15
        self.numEpisodes = 5
        self.numStepsPerEpisode = 250

        self.stepsPerSec = 16384


        self.sDim = self.NUM_NON_KB_DIM + 2 * self.numKilobots

        if continueLearning:
            # dont reset sample or policy
            if self.S is None:
                self.S, self.A, self.R, self.S_ = empty((0, self.sDim)),\
                    empty((0, 2)), empty((0, 1)), empty((0, self.sDim))
        else:
            # reset samples, policy and number of iterations
            self.S, self.A, self.R, self.S_ = empty((0, self.sDim)),\
                    empty((0, 2)), empty((0, 1)), empty((0, self.sDim))
            self.policy = SparseGPPolicy(KilobotKernel(self.NUM_NON_KB_DIM),
                    self.aRange)
            self.it = 0


        """ LSTD """
        self.lstd.discountFactor = 0.99

        factor = 1.0
        factorKb = 1.0
        weightNonKb = 0.5

        self.bwFactorNonKbSA = factor
        self.bwFactorKbSA = factorKb
        self.weightNonKbSA = weightNonKb

        self.bwFactorNonKbS = factor
        self.bwFactorKbS = factorKb
        self.weightNonKbS = weightNonKb

        self.numFeatures = 200

        """ REPS """
        self.reps.epsilonAction = 0.5

        """ GP """
        self.policy.GPMinVariance = 0.0
        self.policy.GPRegularizer = 0.05

        self.numSamplesSubsetGP = 200

        self.bwFactorNonKbGP = factor
        self.bwFactorKbGP = factorKb
        self.weightNonKbGP = weightNonKb

        self.numLearnIt = numLearnIt

        self.startEpsilon = startEpsilon
        self.epsilon = startEpsilon
        self.epsilonFactor = epsilonFactor

        # make data dir and save params
        if savePrefix == '':
            savePath = ''
        else:
            savePath = os.path.join(savePrefix, Helper.getSaveName())
            os.makedirs(savePath)

            self.saveParams(os.path.join(savePath, 'params'))

        rewards = []

        for i in range(numSampleIt):
            t = time.time()

            # get new samples
            St, At, Rt, S_t = self.simulator.getSamples(self.policy, self.objectShape, self.numKilobots, self.numEpisodes, self.numStepsPerEpisode, self.stepsPerSec, self.epsilon, False)

            print('sum reward for last samples: {}'.format(Rt.sum()))
            rewards += [Rt.sum()]

            # add samples
            self.S = r_[self.S, St]
            self.A = r_[self.A, At]
            self.R = r_[self.R, Rt]
            self.S_ = r_[self.S_, S_t]

            # only keep 10000 samples
            SARS = c_[self.S, self.A, self.R, self.S_]
            SARS = Helper.getRandomSubset(SARS, 10000)

            self.S, self.A, self.R, self.S_ = self._unpackSARS(SARS)

            self._updateKernelParameters(self.S, self.A, random=True,
                    normalize=True)

            self.PHI_S = self.kernelS.getGramMatrix(self.S, self.MuS)

            SA = self._getStateActionMatrix(self.S, self.A)
            self.PHI_SA = self.kernelSA.getGramMatrix(SA, self.MuSA)

            for j in range(numLearnIt):
                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                self.PHI_SA_ = self._getFeatureExpectation(self.S_, 5, self.MuSA)
                self.theta = self.lstd.learnLSTD(self.PHI_SA, self.PHI_SA_, self.R)

                # AC-REPS
                self.Q = self.PHI_SA * self.theta
                self.w = self.reps.computeWeighting(self.Q, self.PHI_S)

                # GP
                Ssub = self._getSubsetForGP(self.S, random=True, normalize=True)
                self._updateBandwidthsGP(Ssub)
                self.policy.train(self.S, self.A, self.w, Ssub)

                self.it += 1
                print('finished learning iteration {}'.format(self.it))

                # save results
                if savePath != '':
                    figV = self.getValueFunctionFigure(50, 25, 4)
                    figP = self.getPolicyFigure(50, 25, self.objectShape)

                    figV.savefig(os.path.join(savePath, 'V_{}.svg'.format(self.it)))
                    figP.savefig(os.path.join(savePath, 'P_{}.svg'.format(self.it)))

                    self.savePolicyAndSamples(os.path.join(savePath,
                        'policy_samples_{}'.format(self.it)))

                    plt.close(figV)
                    plt.close(figP)

            self.epsilon *= self.epsilonFactor

            print('sampling iteration took: {}s'.format(time.time() - t))

            gc.collect()

        with open(os.path.join(savePath, 'rewards'), 'w') as f:
            for r in rewards:
                f.write('{}\n'.format(r))

    def getValueFunctionFigure(self, stepsX = 50, stepsY = 25, N = 4):
        [X, Y] = meshgrid(linspace(-0.25, 0.25, stepsX), linspace(-0.25, 0.25, stepsY))
        X = X.flatten()
        Y = -Y.flatten()

        lightX = X
        lightY = Y
        alpha = zeros(size(lightX))

        # kilobots at light position
        KB = tile(c_[X, Y], [1, self.numKilobots])

        Srep = repeat(c_[lightX, lightY, KB], N, axis=0)
        Arep = 0.05 * random.random((X.size * N, 2))

        SA = self._getStateActionMatrix(Srep, Arep)
        PHI_SA_rep = self.kernelSA.getGramMatrix(SA, self.MuSA)
        Qrep = PHI_SA_rep * self.theta

        # max over each N rows
        V = asarray(Qrep).reshape(-1, N, Qrep.shape[1]).max(1).reshape(stepsY, stepsX)

        fig = plt.figure()
        plt.imshow(V)

        plt.colorbar()
        plt.title('value function, iteration {}'.format(self.it))

        return fig

    def getPolicyFigure(self, stepsX, stepsY, figureShape):
        [X, Y] = meshgrid(linspace(-0.25, 0.25, stepsX), linspace(-0.25, 0.25, stepsY))
        X = X.flatten()
        Y = Y.flatten()

        print(figureShape)
        lightX = X
        lightY = Y
        alpha = zeros(size(lightX))

        # kilobots at light position
        KB = tile(c_[X, Y], [1, self.numKilobots])

        A = asarray(self.policy.getMeanAction(c_[lightX, lightY, KB]))
        A /= linalg.norm(A, axis=1).reshape((A.shape[0], 1))

        U = A[:, 0]
        V = A[:, 1]

        fig = plt.figure()

        # draw the object

        if figureShape == "t-form":
            ax = plt.gca()
            ax.add_patch(Polygon([[0, 0], [0.075, 0], [0.075, 0.05], [-0.075, 0.05], [-0.075, 0]], facecolor='grey'))
            ax.add_patch(Polygon([[0.025, 0], [0.025, -0.1], [-0.025, -0.1], [-0.025, 0]], facecolor='grey'))


        elif figureShape == "l-form":
            ax = plt.gca()
            ax.add_patch(Polygon([[0, -0.1], [0.05, -0.1],  [0.05, 0.05], [0, 0.05]], facecolor='black'))
            ax.add_patch(Polygon([[0.05, -0.1], [0.1, -0.1], [0.1, -0.05], [0.05, -0.05]], facecolor='black'))

        else:
            ax = plt.gca()
            ax.add_patch(Rectangle((-0.075, -0.075), 0.15, 0.15, facecolor='grey'))

        plt.quiver(X, Y, U, V)
        plt.title('policy, iteration {}'.format(self.it))

        return fig

if __name__ == '__main__':
    learner = MazeLearner(False)
    learner.learn('quad', 100, False)
