from numpy import c_, repeat

from LeastSquaresTD import LeastSquaresTD
from FeatureFunction import RBFFeatureFunction
from AC_REPS import AC_REPS
from GPPolicy import GPPolicy


class MazeLearner:
    def getMu(self):
        pass

    def getBw(self):
        pass

    def getInitialTheta(self):
        pass

    def getSamples(self):
        # use self.policy
        pass

    def getFeatureExpectation(self, S, N):
        S = repeat(S, N, axis=0)
        A = self.policy.evaluate(S)

        PHI_SA = self.rbf.getStateActionFeatureMatrix(c_[S, A])
        # mean over each N rows
        return PHI_SA.reshape(-1, N, S.shape[1] + A.shape[1]).mean(1)

    def learn(self):
        Mu = self.getMu()
        bw = self.getBw()
        self.rbf = RBFFeatureFunction(Mu, bw)

        lstd = LeastSquaresTD()
        reps = REPS()
        self.policy = GPPolicy()

        samplingIterations = 100
        learningIterations = 10
        numActionSamples = 100

        for i in range(1, samplingIterations):
            # TODO add samples? / change Mu, bw?
            S, A, S_, R = self.getSamples()

            PHI_SA = self.rbf.getStateActionFeatureMatrix(S, A)
            theta = self.getInitialTheta()

            for j in range(1, learningIterations):
                # LSTD to estimate Q function / Q(s,a) = phi(s, a).T * theta
                PHI_SA_ = self.getFeatureExpectation(S_, numActionSamples)
                theta = lstd.learnLSTD(PHI_SA, PHI_SA_, R)

                # AC-REPS
                Q = PHI_SA * theta
                PHI_S = self.rbf.getStateFeatureMatrix(S)


                # GP ...
