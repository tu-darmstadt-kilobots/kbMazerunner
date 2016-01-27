from numpy import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

import os
import gc
import pprint
import time

from zmq import Context, PAIR
import pickle

from Helper import Helper
from LeastSquaresTD import LeastSquaresTD
from AC_REPS import AC_REPS
from Kernel import ExponentialQuadraticKernel, KilobotKernel
from SparseGPPolicy import SparseGPPolicy


class MazeLearner:
    def __init__(self, port):
        self.context = Context()
        self.socket = self.context.socket(PAIR)
        self.port = port

        # s: light.x light.y kb.x1 kb.y1 ... kb.xn kb.yn
        # a: light movement (dx, dy)
        self.NUM_KILOBOTS = 4
        self.NUM_NON_KB_DIM = 2

        # initial random range for actions
        aRange = array([[-0.015, 0.015], [-0.015, 0.015]])
        self.policy = SparseGPPolicy(KilobotKernel(self.NUM_NON_KB_DIM), aRange)

        # kernels used for LSTD
        self.kernelS = KilobotKernel(self.NUM_NON_KB_DIM)
        self.kernelSA = KilobotKernel(self.NUM_NON_KB_DIM + 2)

        self.lstd = LeastSquaresTD()
        self.reps = AC_REPS()

        self.sDim = self.NUM_NON_KB_DIM + 2 * self.NUM_KILOBOTS
        self.S, self.A, self.R, self.S_ = empty((0, self.sDim)), empty((0, 2)),\
                empty((0, 1)), empty((0, self.sDim))

        self.it = 0

        # default parameters
        self.numEpisodes = 60
        self.numStepsPerEpisode = 40

        self.stepsPerSec = 4096

        self.goalReward = 1.0
        self.wallPunishment = 1.0

        self.numFeatures = 100

        self.bwFactorNonKbSA = 1.0
        self.bwFactorKbSA = 1.0

        self.bwFactorNonKbS = 1.0
        self.bwFactorKbS = 1.0

        self.numSamplesSubsetGP = 100

        self.bwFactorNonKbGP = 1.0
        self.bwFactorKbGP = 1.0

    def _sendPolicyModules(self):
        msg = {'message': 'sentPolicyModules',
               'modules': [
                   ('Helper.py', open('Helper.py').read()),
                   ('Kernel.py', open('Kernel.py').read()),
                   ('SparseGPPolicy.py', open('SparseGPPolicy.py').read())
                   ],
               'policyModule': 'SparseGPPolicy'}
        self.socket.send(pickle.dumps(msg, protocol=2))

    def _getSamples(self):
        msg = {'message': 'getSamples',
               'policyDict': self.policy.getSerializableDict(),
               'numEpisodes': self.numEpisodes,
               'numStepsPerEpisode': self.numStepsPerEpisode,
               'stepsPerSec': self.stepsPerSec,
               'goalReward': self.goalReward,
               'wallPunishment': self.wallPunishment,
               'epsilon': self.epsilon,
               'useMean': False}
        self.socket.send(pickle.dumps(msg, protocol=2))

        msg = pickle.loads(self.socket.recv(), encoding='latin1')
        if not msg['message'] == 'sentSamples':
            print('received unexpected message')
        else:
            return msg['samples']

    """
        for now just execute the mean policy to see how good it looks
        can later be done automatically
    """
    def testPolicy(self, numEpisodes = 100, numStepsPerEpisode = 50,
            stepsPerSec = 8):
        msg = {'message': 'getSamples',
               'policyDict': self.policy.getSerializableDict(),
               'numEpisodes': numEpisodes,
               'numStepsPerEpisode': numStepsPerEpisode,
               'stepsPerSec': stepsPerSec,
               'epsilon': 0.0,
               'useMean': True}
        self.socket.send(pickle.dumps(msg, protocol=2))

        # discard result
        self.socket.recv()

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

        # bandwidth for PHI_S
        bwNonKbS = Helper.getBandwidth(self.MuS[:, 0:self.NUM_NON_KB_DIM],
                500, self.bwFactorNonKbS)

        kbPosS = self._reshapeKbPositions(self.MuS[:, self.NUM_NON_KB_DIM:])
        bwKbS = Helper.getBandwidth(kbPosS, 500, self.bwFactorKbS)

        self.kernelS.setBandwidth(bwNonKbS, bwKbS)
        self.kernelS.setWeighting(self.weightNonKbS)

        # bandwidth for PHI_SA
        bwNonKbSA = Helper.getBandwidth(self.MuSA[:, 0:(self.NUM_NON_KB_DIM + 2)],
                500, self.bwFactorNonKbSA)

        kbPosSA = self._reshapeKbPositions(self.MuSA[:, (self.NUM_NON_KB_DIM + 2):])
        bwKbSA = Helper.getBandwidth(kbPosSA, 500, self.bwFactorKbSA)

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

    def connect(self):
        self.socket.bind('tcp://*:{}s'.format(self.port))

        self._sendPolicyModules()

    def savePolicyAndSamples(self, fileName):
        d = self.policy.getSerializableDict()
        d['SARS'] = c_[self.S, self.A, self.R, self.S_]

        s = pickle.dumps(d)
        with open(fileName, 'wb') as f:
            f.write(s)

    def loadPolicyAndSamples(self, fileName):
        with open(fileName, 'rb') as f:
            s = f.read()
        d = pickle.loads(s)

        self.policy = SparseGPPolicy.fromSerializableDict(d)

        SARS = d['SARS']
        self.S, self.A, self.R, self.S_ = self._unpackSARS(SARS)

    def saveParams(self, fileName):
        params = {'numEpisodes': self.numEpisodes,
                  'numStepsPerEpisode': self.numStepsPerEpisode,
                  'normalizeRepresentativeRows': self.normalizeRepRows,
                  'startEpsilon': self.startEpsilon,
                  'epsilonFactor': self.epsilonFactor,
                  'numLearnIterations': self.numLearnIt,
                  'lstd.bwFactorSAOuter': self.bwFactorSAOuter,
                  'lstd.bwFactorSAInner': self.bwFactorSAInner,
                  'lstd.bwFactorSOuter': self.bwFactorSOuter,
                  'lstd.bwFactorSInner': self.bwFactorSInner,
                  'lstd.discountFactor': self.lstd.discountFactor,
                  'reps.bwFactorS': self.bwFactorS,
                  'reps.epsilonAction': self.reps.epsilonAction,
                  'gp.MinVariance': self.policy.GPMinVariance,
                  'gp.Regularizer': self.policy.GPRegularizer,
                  'gp.bwFactorOuter': self.policy.bwFactorOuter,
                  'gp.bwFactorInner': self.policy.bwFactorInner
                  }

        with open(fileName, 'w') as f:
            f.write(pprint.pformat(params, width=1))

    def learn(self, savePrefix, numSampleIt, numLearnIt = 1,
            startEpsilon = 0.0, epsilonFactor = 1.0):

        """ sampling """
        self.numEpisodes = 50
        self.numStepsPerEpisode = 50

        self.stepsPerSec = 4096

        """ LSTD """
        self.lstd.discountFactor = 0.95

        factor = 2.0
        factorKb = 2.0
        weightNonKb = 1.0

        self.bwFactorNonKbSA = factor
        self.bwFactorKbSA = factorKb
        self.weightNonKbSA = weightNonKb

        self.bwFactorNonKbS = factor
        self.bwFactorKbS = factorKb
        self.weightNonKbS = weightNonKb

        self.numFeatures = 100

        """ REPS """
        self.reps.epsilonAction = 0.5

        """ GP """
        self.policy.GPMinVariance = 0.0
        self.policy.GPRegularizer = 0.05

        self.numSamplesSubsetGP = 100

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


        for i in range(numSampleIt):
            # get new samples
            St, At, Rt, S_t = self._getSamples()
            print('sum reward for last samples: {}'.format(Rt.sum()))

            # add samples
            self.S = r_[self.S, St]
            self.A = r_[self.A, At]
            self.R = r_[self.R, Rt]
            self.S_ = r_[self.S_, S_t]

            # only keep 5000 samples
            SARS = c_[self.S, self.A, self.R, self.S_]
            SARS = Helper.getRandomSubset(SARS, 5000)
            #SARS = Helper.getRepresentativeRows(SARS, 5000, self.normalizeRepRows)

            self.S, self.A, self.R, self.S_ = self._unpackSARS(SARS)

            self._updateKernelParameters(self.S, self.A, random=True,
                    normalize=True)
            #MazeLearner.plotFeatures(self.MuS)
            #input('press key...')


            self.PHI_S = self.kernelS.getGramMatrix(self.S, self.MuS)

            SA = self._getStateActionMatrix(self.S, self.A)
            self.PHI_SA = self.kernelSA.getGramMatrix(SA, self.MuSA)

            for j in range(numLearnIt):
                t = time.time()

                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                self.PHI_SA_ = self._getFeatureExpectation(self.S_, 5, self.MuSA)
                self.theta = self.lstd.learnLSTD(self.PHI_SA, self.PHI_SA_, self.R)

                # AC-REPS
                self.Q = self.PHI_SA * self.theta
                self.w = self.reps.computeWeighting(self.Q, self.PHI_S)

                # show weights for light positions
                #plt.scatter(self.S[:, 0], self.S[:, 1], c=self.w.flat)
                #plt.show()
                #input('press key')

                # GP
                Ssub = self._getSubsetForGP(self.S, random=True, normalize=True)
                self._updateBandwidthsGP(Ssub)
                self.policy.train(self.S, self.A, self.w, Ssub)

                self.it += 1
                print('finished learning iteration {}'.format(self.it))
                print('took: {}s'.format(time.time() - t))

                # plot save results
                figV = self.getValueFunctionFigure(100, 50, 4)
                figV.show()
                input('press key')

                #figP = self.getPolicyFigure(20, 10)

                #if savePath != '':
                #    figV.savefig(os.path.join(savePath, 'V_{}.svg'.format(self.it)))
                #    figP.savefig(os.path.join(savePath, 'P_{}.svg'.format(self.it)))

                #if self.it % 5 == 0:
                #    self.savePolicyAndSamples(os.path.join(savePath,
                #        'policy_samples_{}'.format(self.it)))

            self.epsilon *= self.epsilonFactor

            gc.collect()

    @staticmethod
    def plotFeatures(X):
        colors = cm.rainbow(linspace(0, 1, X.shape[0]))

        for i, c in zip(range(X.shape[0]), colors):
            L = asmatrix(X[i, 0:2])
            KB = asmatrix(X[i, 2:])
            KB = c_[KB.flat[0::2].T, KB.flat[1::2].T]

            plt.plot(L[0, 0], L[0, 1], 'x', color=c, markersize=10)
            plt.plot(KB[:, 0], KB[:, 1], 'o', color=c)

        plt.show()


    def getValueFunctionFigure(self, stepsX = 100, stepsY = 50, N = 10):
        [X, Y] = meshgrid(linspace(-2.0, 4.0, stepsX), linspace(-2.0, 3.0, stepsY))
        X = X.flatten()
        Y = 1.0 - Y.flatten()

        lightX = X
        lightY = Y

        # kilobots at light position
        KB = c_[X, Y, X, Y, X, Y, X, Y]

        Srep = repeat(c_[lightX, lightY, KB], N, axis=0)
        Arep = 0.05 * random.random((X.size * N, 2)) # TODO
        #Srep, Arep = self.policy.sampleActions(c_[X, Y], N)

        SA = self._getStateActionMatrix(Srep, Arep)
        PHI_SA_rep = self.kernelSA.getGramMatrix(SA, self.MuSA)
        Qrep = PHI_SA_rep * self.theta

        # max over each N rows
        V = asarray(Qrep).reshape(-1, N, Qrep.shape[1]).max(1).reshape(stepsY, stepsX)

        #S = c_[X, Y, KB]
        #A = c_[0 * ones((S.shape[0], 1)), 0.01 * ones((S.shape[0], 1))]

        #PHI_SA = self.kernelSA.getGramMatrix(c_[S, A], self.MuSA)

        fig = plt.figure()
        plt.imshow(V)

        plt.colorbar()
        plt.title('value function, iteration {}'.format(self.it))

        return fig

    """
    def getPolicyFigure(self, stepsX = 50, stepsY = 25):
        [X, Y] = meshgrid(linspace(0.0, 2.0, stepsX), linspace(0.0, 1.0, stepsY))
        X = X.flatten()
        Y = Y.flatten()

        A = asarray(self.policy.getMeanAction(c_[X, Y]))
        A /= linalg.norm(A, axis=1).reshape((A.shape[0], 1))

        U = A[:, 0]
        V = A[:, 1]

        fig = plt.figure()

        plt.quiver(X, Y, U, V)
        plt.title('policy, iteration {}'.format(self.it))

        return fig"""

learner = MazeLearner(2357)
learner.connect()
