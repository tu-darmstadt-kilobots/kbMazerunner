from numpy import asmatrix, square, diagflat, sqrt, pi, multiply as mul
from numexpr import evaluate as ev


class FeatureFunction:
    """
        A: m x d

        return m x k feature matrix, where k is the feature vector dimension
    """
    def getFeatureMatrix(A):
        raise NotImplementedError("getFeatureMatrix not implemented")


class RBFFeatureFunction(FeatureFunction):
    def __init__(self, Mu, bw):
        self.setParameters(Mu, bw)

    def setParameters(self, Mu, bw):
        self.Mu = asmatrix(Mu)
        self.bw2 = square(bw)

    def getFeatureMatrix(self, A):
        X = asmatrix(A)

        Q = diagflat(1.0 / self.bw2)
        XQ = X * Q
        MuQ = self.Mu * Q

        B = mul(XQ, X).sum(1) + mul(MuQ, self.Mu).sum(1).T
        C = XQ * self.Mu.T
        s = sqrt(self.bw2.prod() * (2 * pi) ** (self.bw2.size))

        return asmatrix(ev("exp(-0.5 * (B - 2 * C)) / s"))
