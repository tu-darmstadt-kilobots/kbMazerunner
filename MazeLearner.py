from numpy import array, asarray, zeros, c_, r_
from scipy.spatial.distance import cdist

from LeastSquaresTD import LeastSquaresTD
from FeatureFunction import RBFFeatureFunction
from AC_REPS import AC_REPS
from Kernel import ExponentialQuadraticKernel
from SparseGPPolicy import SparseGPPolicy
from zmq import Context, PAIR
import pickle


class MazeLearner:
    def __init__(self, port):
        self.context = Context()
        self.socket = self.context.socket(PAIR)
        self.socket.bind('tcp://*:{}s'.format(port))

        # GP policy
        kernel = ExponentialQuadraticKernel(array([0.1, 0.1, 0.1, 0.1]))
        aRange = array([[-1e-1, 1e-1], [-1e-1, 1e-1]])
        self.policy = SparseGPPolicy(kernel, aRange)

        self._sendPolicyModules()

    def _sendPolicyModules(self):
        msg = {'message': 'sentPolicyModules',
               'modules': [
                   ('Kernel.py', open('Kernel.py').read()),
                   ('SparseGPPolicy.py', open('SparseGPPolicy.py').read())
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

    def _getRepresentativeRows(self, X, N):
        Y = zeros((N, X.shape[1]))

        for i in range(N):
            Y[i, :] = X[cdist(X, Y).min(axis=1).argmax(), :]

        return Y

    def _getFeatureExpectation(self, S, N):
        Srep, Arep = self.policy.sampleActions(S, N)
        PHI_SA_rep = self.rbf.getStateActionFeatureMatrix(Srep, Arep)

        # mean over each N rows
        return asarray(PHI_SA_rep).reshape(-1, N, PHI_SA_rep.shape[1]).mean(1)

    def learn(self):
        S, A, R, S_ = self._getSamples()

        MuSA = self._getRepresentativeRows(c_[S, A], 500)
        MuS = self._getRepresentativeRows(S, 500)
        bwSA = array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        bwS = array([0.1, 0.1, 0.1, 0.1])

        self.rbf = RBFFeatureFunction(MuSA, bwSA, MuS, bwS)

        lstd = LeastSquaresTD()
        reps = AC_REPS()

        for i in range(10):
            PHI_SA = self.rbf.getStateActionFeatureMatrix(S, A)

            for j in range(2):
                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                PHI_SA_ = self._getFeatureExpectation(S_, 200)

                theta = lstd.learnLSTD(PHI_SA, PHI_SA_, R)

                # AC-REPS
                Q = PHI_SA * theta
                PHI_S = self.rbf.getStateFeatureMatrix(S)
                w = reps.computeWeighting(Q, PHI_S)

                # GP
                self.policy.train(S, A, w)

            St, At, Rt, S_t = self._getSamples()
            S = r_[S, St]
            A = r_[A, At]
            R = r_[R, Rt]
            S_ = r_[S_, S_t]

            MuSA = self._getRepresentativeRows(c_[S, A], 500)
            MuS = self._getRepresentativeRows(S, 500)
            self.rbf.setParameters(MuSA, bwSA, MuS, bwS)


if __name__ == '__main__':
    learner = MazeLearner(2357)
    learner.learn()
