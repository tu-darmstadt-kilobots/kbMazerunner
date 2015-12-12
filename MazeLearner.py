from numpy import array, asarray, zeros, c_, r_, repeat, sqrt, median, \
        newaxis, square, transpose, random, linspace, meshgrid, empty

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

import gc

from zmq import Context, PAIR
import pickle

from Helper import Helper
from LeastSquaresTD import LeastSquaresTD
from FeatureFunction import RBFFeatureFunction
from AC_REPS import AC_REPS
from Kernel import ExponentialQuadraticKernel
from SparseGPPolicy import SparseGPPolicy


class MazeLearner:
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

        self.S, self.A, self.R, self.S_ = empty((0, 2)), empty((0, 2)),\
                empty((0, 1)), empty((0, 2))

        self.it = 0

    def _sendPolicyModules(self):
        msg = {'message': 'sentPolicyModules',
               'modules': [
                   ('Helper.py', open('Helper.py').read()),
                   ('Kernel.py', open('Kernel.py').read()),
                   ('SparseGPPolicy.py', open('SparseGPPolicy.py').read())
                   ],
               'policyModule': 'SparseGPPolicy'}
        self.socket.send(pickle.dumps(msg, protocol=2))

    def _getSamples(self, epsilon, useMean):
        msg = {'message': 'getSamples',
               'policyDict': self.policy.getSerializableDict(),
               'epsilon': epsilon,
               'useMean': useMean}
        self.socket.send(pickle.dumps(msg, protocol=2))

        msg = pickle.loads(self.socket.recv(), encoding='latin1')
        if not msg['message'] == 'sentSamples':
            print('received unexpected message')
        else:
            return msg['samples']

    def _updateRBFParameters(self):
        MuSA = Helper.getRepresentativeRows(c_[self.S, self.A], 500, True)

        MuS = Helper.getRepresentativeRows(self.S, 500, True)

        bwSA = Helper.getBandwidth(MuSA, 500, 0.5) # 3.0 0.5
        bwS = Helper.getBandwidth(MuS, 500, 0.5) # 2.5 0.5

        self.rbf = RBFFeatureFunction(MuSA, bwSA, MuS, bwS)

    def connect(self):
        self.socket.bind('tcp://*:{}s'.format(self.port))

        self._sendPolicyModules()

    def savePolicy(self, fileName):
        d = self.policy.getSerializableDict()
        s = pickle.dumps(d)
        with open(fileName, 'wb') as f:
            f.write(s)

    def loadPolicy(self, fileName):
        with open(fileName, 'rb') as f:
            s = f.read()
        d = pickle.loads(s)
        self.policy = SparseGPPolicy.fromSerializableDict(d)

    def learn(self, startEpsilon, numIt, numLearnIt):
        self.lstd.discountFactor = 0.95

        self.policy.GPMinVariance = 0.0
        self.policy.GPRegularizer = 0.005
        self.policy.bwFactor = 0.5 # 2.5 0.5

        self.reps.epsilonAction = 0.5

        self.epsilon = startEpsilon
        epsilonFactor = 0.9;

        for i in range(numIt):
            # get new samples
            St, At, Rt, S_t = self._getSamples(self.epsilon, False)
            print('sum reward for last samples: {}'.format(Rt.sum()))

            # add samples
            self.S = r_[self.S, St]
            self.A = r_[self.A, At]
            self.R = r_[self.R, Rt]
            self.S_ = r_[self.S_, S_t]

            SARS = Helper.getRepresentativeRows(c_[self.S, self.A, self.R, self.S_],
                    20000, True)
            self.S = SARS[:, 0:2]
            self.A = SARS[:, 2:4]
            self.R = SARS[:, 4:5]
            self.S_ = SARS[:, 5:7]

            self._updateRBFParameters()

            self.PHI_SA = self.rbf.getStateActionFeatureMatrix(self.S, self.A)
            self.PHI_S = self.rbf.getStateFeatureMatrix(self.S)

            for j in range(numLearnIt):
                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                # TODO memory
                self.PHI_SA_ = Helper.getFeatureExpectation(self.S_, 5,
                        self.policy, self.rbf)
                self.theta = self.lstd.learnLSTD(self.PHI_SA, self.PHI_SA_, self.R)

                # AC-REPS
                self.Q = self.PHI_SA * self.theta
                self.w = self.reps.computeWeighting(self.Q, self.PHI_S)

                # GP
                self.policy.train(self.S, self.A, self.w)

                """ DEBUG """

                #self.plotV(100, 50, 10)
                #self.plotPolicyState2D(50, 25)
                #input('Press key to continue...')

            self.epsilon *= epsilonFactor

            self.it += 1
            gc.collect()

            print(self.it)

    def plotV(self, stepsX, stepsY, N):
        [X, Y] = meshgrid(linspace(0.0, 2.0, stepsX), linspace(0.0, 1.0, stepsY))
        X = X.flatten()
        Y = 1.0 - Y.flatten()

        Srep = repeat(c_[X, Y], N, axis=0)
        Arep = 0.05 * random.random((X.size * N, 2)) # TODO
        #Srep, Arep = self.policy.sampleActions(c_[X, Y], N)

        PHI_SA_rep = self.rbf.getStateActionFeatureMatrix(Srep, Arep)
        Qrep = PHI_SA_rep * self.theta

        # max over each N rows
        V = asarray(Qrep).reshape(-1, N, Qrep.shape[1]).max(1).reshape(stepsY, stepsX)

        plt.figure()
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
        plt.title('V, iteration: {}'.format(self.it))
        plt.show(block=False)


    def plotPolicyState2D(self, stepsX, stepsY):
        [X, Y] = meshgrid(linspace(0.0, 2.0, stepsX), linspace(0.0, 1.0, stepsY))
        X = X.flatten()
        Y = Y.flatten()
        A = asarray(self.policy.getMeanAction(c_[X, Y]))
        U = A[:, 0]
        V = A[:, 1]

        plt.figure()
        plt.quiver(X, Y, U, V)
        plt.title('iteration: {}'.format(self.it))
        plt.show(block=False)

learner = MazeLearner(2357)
learner.connect()
