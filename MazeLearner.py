from numpy import c_, repeat

from LeastSquaresTD import LeastSquaresTD
from FeatureFunction import RBFFeatureFunction
from REPS import REPS
from GPPolicy import GPPolicy


class MazeLearner:
    def get_Mu(self):
        pass

    def get_bw(self):
        pass

    def get_initial_theta(self):
        pass

    def get_samples(self):
        # use self.policy
        pass

    def get_feature_expectation(self, S, N):
        S = repeat(S, N, axis=0)
        A = self.policy.evaluate(S)

        PHI = self.rbf.getFeatureMatrix(c_[S, A])
        # mean over each N rows
        return PHI.reshape(-1, N, S.shape[1] + A.shape[1]).mean(1)

    def learn(self):
        Mu = self.get_Mu()
        bw = self.get_bw()
        self.rbf = RBFFeatureFunction(Mu, bw)

        lstd = LeastSquaresTD()
        reps = REPS()
        self.policy = GPPolicy()

        samplingIterations = 100
        learningIterations = 10
        numActionSamples = 100

        for i in range(1, samplingIterations):
            # TODO add samples? / change Mu, bw?
            S, A, S_, R = self.get_samples()

            PHI = self.rbf.getFeatureMatrix(c_[S, A])
            theta = self.get_initial_theta()

            for j in range(1, learningIterations):
                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                PHI_ = self.get_feature_expectation(S_, numActionSamples)
                theta = lstd.learnLSTD(PHI, PHI_, R)

                # REPS ...
                Q = PHI * theta

                # GP ...
