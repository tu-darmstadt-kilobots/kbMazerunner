from numpy import eye, asmatrix, r_, multiply as mul
from numpy.linalg import solve, inv


class LeastSquaresTD:
    def __init__(self):
        self.discountFactor = 0.98
        self.lstdRegularizationFactor = 1e-8
        self.lstdProjectionRegularizationFactor = 1e-6

    def learnLSTD(self, stateActionFeatures, nextStateActionFeatures, reward):
        phi = asmatrix(stateActionFeatures)
        phi_ = asmatrix(nextStateActionFeatures)

        A_ = phi.T * (phi - self.discountFactor * phi_)
        b_ = mul(phi, reward).sum(0).T

        n = phi.shape[1]
        C = phi * inv(phi.T * phi + self.lstdRegularizationFactor * eye(n))
        X = C * (A_ + self.lstdRegularizationFactor * eye(n))
        y = C * b_

        return solve(X.T * X + self.lstdProjectionRegularizationFactor * eye(n),
                X.T * y)
