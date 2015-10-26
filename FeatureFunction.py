from numpy import asmatrix, square, diagflat, sqrt, pi, c_, multiply as mul
from numexpr import evaluate as ev


class FeatureFunction:
    """
        S: m x d1
        A: m x d2

        return m x k feature matrix, where k is the feature vector dimension
    """
    def getStateActionFeatureMatrix(self, S, A):
        raise NotImplementedError("getStateActionFeatureMatrix not implemented")

    def getStateFeatureMatrix(self, S):
        raise NotImplementedError("getStateFeatureMatrix not implemented")


class RBFFeatureFunction(FeatureFunction):
    # TODO seperate bandwidths for phi(s, a) and phi(s)?
    def __init__(self, MuSA, MuS, bw):
        self.setParameters(MuSA, MuS, bw)

    def setParameters(self, MuSA, MuS, bw):
        self.MuSA = asmatrix(MuSA)
        self.MuS = asmatrix(MuS)
        self.bw2 = square(bw)

    def _computeFeatureMatrix(X, Mu, bw2):
        X = asmatrix(X)

        Q = diagflat(1.0 / bw2)
        XQ = X * Q
        MuQ = Mu * Q

        B = mul(XQ, X).sum(1) + mul(MuQ, Mu).sum(1).T
        C = XQ * Mu.T
        s = sqrt(bw2.prod() * (2 * pi) ** (bw2.size))

        return asmatrix(ev("exp(-0.5 * (B - 2 * C)) / s"))

    def getStateActionFeatureMatrix(self, S, A):
        return _computeFeatureMatrix(c_[S, A], self.MuSA, self.bw2)

    def getStateFeatureMatrix(self, S):
        return _computeFeatureMatrix(S, self.MuS, self.bw2)
