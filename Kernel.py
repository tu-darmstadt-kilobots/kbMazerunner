from numpy import asmatrix, square, diagflat, sqrt, pi, ones, multiply as mul
from numexpr import evaluate as ev


class Kernel:
    """
        A: m1 x d
        B: m2 x d

        returns m1 x m2 gram matrix
    """
    def getGramMatrix(self, A, B):
        raise NotImplementedError("getGramMatrix() not implemented.")

    def getGramDiag(self, A):
        raise NotImplementedError("getGramDiag() not implemented")


class ExponentialQuadraticKernel(Kernel):
    normalize = False

    def __init__(self, bw):
        self.bw2 = square(bw)

    def getGramMatrix(self, A, B):
        Q = asmatrix(diagflat(1.0 / self.bw2))

        AQ = A * Q
        K = mul(AQ, A).sum(1) + mul(B * Q, B).sum(1).T
        K -= 2.0 * AQ * B.T
        K = ev("exp(-0.5 * K)")

        if self.normalize:
            K /= sqrt(self.bw2.prod() * (2.0 * pi) ** (self.bw2.size))

        return K

    def getGramDiag(self, A):
        return ones((A.shape[0], 1))
