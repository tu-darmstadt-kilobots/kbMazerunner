from numpy import eye, asmatrix
from numpy.linalg import solve


class LeastSquaresTD:
    discountFactor = 0.98
    lstdRegularizationFactor = 1e-8
    lstdProjectionRegularizationFactor = 1e-6

    def learnLSTD(self, stateActionFeatures, nextStateActionFeatures, reward):
        phi = asmatrix(stateActionFeatures)
        phi_ = asmatrix(nextStateActionFeatures)

        regMat1 = eye(phi.shape[1]) * self.lstdRegularizationFactor
        regMat2 = eye(phi.shape[1]) * self.lstdProjectionRegularizationFactor

        projector = solve(phi.T * phi + regMat2, phi.T * phi_)
        M = phi - self.discountFactor * phi * projector

        return solve(M.T * M + regMat1, M.T * reward)
