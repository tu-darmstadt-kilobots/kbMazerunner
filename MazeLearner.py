from numpy import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

import os
import gc
import pprint

from zmq import Context, PAIR
import pickle

from Helper import Helper
from LeastSquaresTD import LeastSquaresTD
from AC_REPS import AC_REPS
from Kernel import ExponentialQuadraticKernel, KernelOverKernel
from SparseGPPolicy import SparseGPPolicy


class MazeLearner:
    def __init__(self, port):
        self.context = Context()
        self.socket = self.context.socket(PAIR)
        self.port = port

        # initial random range for actions
        aRange = array([[-0.1, 0.1], [-0.1, 0.1]])
        self.policy = SparseGPPolicy(aRange)

        # kernels used for LSTD
        self.kernelS = KernelOverKernel(ExponentialQuadraticKernel(5),
                ExponentialQuadraticKernel(20))
        self.kernelSA = KernelOverKernel(ExponentialQuadraticKernel(7),
                ExponentialQuadraticKernel(20))

        self.lstd = LeastSquaresTD()
        self.reps = AC_REPS()

        self.S, self.A, self.R, self.S_ = empty((0, 25)), empty((0, 2)),\
                empty((0, 1)), empty((0, 25))

        self.it = 0

        # default parameters
        self.numEpisodes = 200
        self.numStepsPerEpisode = 40

        self.stepsPerSec = 4096

        self.goalReward = 1.0
        self.wallPunishment = 1.0

        self.normalizeRepRows = False

        self.bwFactorSAOuter = 5.0
        self.bwFactorSAInner = 5.0

        self.bwFactorSOuter = 5.0
        self.bwFactorSInner = 5.0

        self.policy.bwFactorOuter = 5.0
        self.policy.bwFactorInner = 5.0

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
               'goalReward': 0.0,
               'wallPunishment': 0.0,
               'epsilon': 0.0,
               'useMean': True}
        self.socket.send(pickle.dumps(msg, protocol=2))

        # discard result
        self.socket.recv()

    def _updateKernelParameters(self):
        self.MuSA = Helper.getRepresentativeRows(c_[self.S, self.A], 500,
                self.normalizeRepRows)
        self.MuS = Helper.getRepresentativeRows(self.S, 500, self.normalizeRepRows)

        bwSAOuter = Helper.getBandwidth(self.MuSA[:, 0:7], 500,
                self.bwFactorSAOuter)
        bwSAInner = Helper.getBandwidth(self.MuSA[:, 7:], 500,
                self.bwFactorSAInner)

        bwSOuter = Helper.getBandwidth(self.MuS[:, 0:5], 500,
                self.bwFactorSOuter)
        bwSInner = Helper.getBandwidth(self.MuS[:, 5:], 500,
                self.bwFactorSInner)

        self.kernelS.setBandwidth(bwSOuter, bwSInner)
        self.kernelSA.setBandwidth(bwSAOuter, bwSAInner)

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
        self.S = SARS[:, 0:25]
        self.A = SARS[:, 25:27]
        self.R = SARS[:, 27:28]
        self.S_ = SARS[:, 28:53]

    def saveParams(self, fileName):
        params = {'numEpisodes': self.numEpisodes,
                  'numStepsPerEpisode': self.numStepsPerEpisode,
                  'goalReward': self.goalReward,
                  'wallPunishment': self.wallPunishment,
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
        self.lstd.discountFactor = 0.95

        self.reps.epsilonAction = 0.5

        self.policy.GPMinVariance = 0.0
        self.policy.GPRegularizer = 0.005

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

            # only keep 20000 samples
            SARS = c_[self.S, self.A, self.R, self.S_]
            SARS = Helper.getRepresentativeRows(SARS, 20000,
                    self.normalizeRepRows)

            self.S = SARS[:, 0:25]
            self.A = SARS[:, 25:27]
            self.R = SARS[:, 27:28]
            self.S_ = SARS[:, 28:53]

            self._updateKernelParameters()

            self.PHI_S = self.kernelS.getGramMatrix(self.S, self.MuS)
            self.PHI_SA = self.kernelSA.getGramMatrix(c_[self.S, self.A], self.MuSA)

            for j in range(numLearnIt):
                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                self.PHI_SA_ = Helper.getFeatureExpectation(self.S_, 5,
                        self.policy, self.kernelSA, self.MuSA)
                self.theta = self.lstd.learnLSTD(self.PHI_SA, self.PHI_SA_, self.R)

                # AC-REPS
                self.Q = self.PHI_SA * self.theta
                self.w = self.reps.computeWeighting(self.Q, self.PHI_S)

                # GP
                self.policy.train(self.S, self.A, self.w)

                self.it += 1
                print('finished learning iteration {}'.format(self.it))

                # plot save results
                #figV = self.getValueFunctionFigure()
                #figP = self.getPolicyFigure(20, 10)

                #if savePath != '':
                #    figV.savefig(os.path.join(savePath, 'V_{}.svg'.format(self.it)))
                #    figP.savefig(os.path.join(savePath, 'P_{}.svg'.format(self.it)))

                #if self.it % 5 == 0:
                #    self.savePolicyAndSamples(os.path.join(savePath,
                #        'policy_samples_{}'.format(self.it)))

            self.epsilon *= self.epsilonFactor

            gc.collect()

    def getValueFunctionFigure(self, stepsX = 100, stepsY = 50, N = 10):
        [X, Y] = meshgrid(linspace(0.0, 2.0, stepsX), linspace(0.0, 1.0, stepsY))
        X = X.flatten()
        Y = 1.0 - Y.flatten()

        Srep = repeat(c_[X, Y], N, axis=0)
        Arep = 0.05 * random.random((X.size * N, 2)) # TODO
        #Srep, Arep = self.policy.sampleActions(c_[X, Y], N)

        PHI_SA_rep = self.kernelSA.getGramMatrix(c_[Srep, Arep], self.MuSA)
        Qrep = PHI_SA_rep * self.theta

        # max over each N rows
        V = asarray(Qrep).reshape(-1, N, Qrep.shape[1]).max(1).reshape(stepsY, stepsX)

        fig = plt.figure()
        plt.imshow(V)

        ax = plt.gca()

        sx = stepsX / 2.0
        sy = stepsY / 1.0

        # wall near goal
        ax.add_patch(Rectangle((0.49 * sx,  0.5 * sy), 0.01 * sx, 0.5 * sy,
            facecolor='grey'))

        # walls near start
        ax.add_patch(Rectangle((0.99 * sx, 0.0), 0.01 * sx, 0.5 * sy,
            facecolor='grey'))
        ax.add_patch(Rectangle((1.00 * sx,  0.49 * sy), 0.5 * sx, 0.01 * sy,
            facecolor='grey'))

        plt.colorbar()
        plt.title('value function, iteration {}'.format(self.it))

        return fig

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

        return fig

learner = MazeLearner(2357)
learner.connect()
