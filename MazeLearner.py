from numpy import asarray

from LeastSquaresTD import LeastSquaresTD
from FeatureFunction import RBFFeatureFunction
from AC_REPS import AC_REPS
from Kernel import ExponentialQuadraticKernel
from SparseGPPolicy import SparseGPPolicy


class MazeLearner:
    def _getMu(self):
        pass

    def _getBw(self):
        pass

    def _getKernel(self):
        pass

    def _getInitialTheta(self):
        pass

    def _getSamples(self):
        # use self.policy
        pass

    def _getFeatureExpectation(self, S, N):
        Srep, Arep = self.policy.sampleActions(S, N)
        PHI_SA_rep = self.rbf.getStateActionFeatureMatrix(Srep, Arep)

        # mean over each N rows
        return asarray(PHI_SA_rep).reshape(-1, N, PHI_SA_rep.shape[1]).mean(1)

    def learn(self):
        MuSA, MuS = self._getMu()
        bwSA, bwS = self._getBw()
        self.rbf = RBFFeatureFunction(MuSA, bwSA, MuS, bwS)

        lstd = LeastSquaresTD()
        reps = AC_REPS()
        self.policy = SparseGPPolicy(self._getKernel())

        samplingIterations = 100
        learningIterations = 10
        numActionSamples = 100

        theta = self._getInitialTheta()

        for i in range(1, samplingIterations):
            # TODO add samples? / change Mu, bw?
            S, A, S_, R = self._getSamples()

            PHI_SA = self.rbf.getStateActionFeatureMatrix(S, A)

            for j in range(1, learningIterations):
                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                PHI_SA_ = self.getFeatureExpectation(S_, numActionSamples)
                theta = lstd.learnLSTD(PHI_SA, PHI_SA_, R)

                # AC-REPS
                Q = PHI_SA * theta
                PHI_S = self.rbf.getStateFeatureMatrix(S)
                w = reps.computeWeighting(Q, PHI_S)

                # GP
                self.policy.train(S, A, w)
