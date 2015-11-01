from numpy import asmatrix, square, diagflat, sqrt, pi, ones, array, \
        multiply as mul
from numexpr import evaluate as ev
import pickle


class Kernel:
    """
        returns a dict, which can be serialized with json.dumps
    """
    def getSerializableDict(self):
        raise NotImplementedError("getSerializableDict() not implemented")

    """
        loads data from dict created using getSerializableDict
    """
    @classmethod
    def fromSerializableDict(d):
        if d['type'] == 'ExpQuad':
            return ExponentialQuadraticKernel.fromSerializableDict(d)

    """
        A: m1 x d
        B: m2 x d

        returns m1 x m2 gram matrix
    """
    def getGramMatrix(self, A, B):
        raise NotImplementedError('getGramMatrix() not implemented.')

    def getGramDiag(self, A):
        raise NotImplementedError('getGramDiag() not implemented')


class ExponentialQuadraticKernel(Kernel):
    def __init__(self, bw):
        self.normalize = False

        self.bw2 = square(bw)

    def getSerializableDict(self):
        d = self.__dict__.copy()
        d['type'] = 'ExpQuad'
        d['bw2'] = pickle.dumps(self.bw2, protocol=2)
        return d

    @classmethod
    def fromSerializableDict(d):
        pass


    def getGramMatrix(self, A, B):
        Q = asmatrix(diagflat(1.0 / self.bw2))

        AQ = A * Q
        K = mul(AQ, A).sum(1) + mul(B * Q, B).sum(1).T
        K -= 2.0 * AQ * B.T
        K = ev('exp(-0.5 * K)')

        if self.normalize:
            K /= sqrt(self.bw2.prod() * (2.0 * pi) ** (self.bw2.size))

        return K

    def getGramDiag(self, A):
        return ones((A.shape[0], 1))
